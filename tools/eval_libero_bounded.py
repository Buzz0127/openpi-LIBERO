#!/usr/bin/env python3
"""Run a bounded, auditable subset of LIBERO episodes against an OpenPI server.

This tool intentionally lives outside the upstream OpenPI checkout. It keeps the
official observation/action preprocessing semantics while adding explicit task
and initial-state selection, collision-safe outputs, per-request timing, GPU
sampling, structured results, and deterministic environment cleanup.

The script must be executed with the LIBERO Python environment and the same
runtime variables used by the verified no-Docker EGL setup. It never starts the
policy server itself.
"""

import argparse
import dataclasses
import datetime
import hashlib
import json
import logging
import math
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


class ResourcePressureError(RuntimeError):
    """Raised when the selected GPU crosses the configured memory threshold."""


@dataclasses.dataclass
class GpuSample:
    timestamp_utc: str
    physical_gpu: int
    utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float
    temperature_c: float
    power_w: Optional[float]

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


class GpuSampler:
    """Bounded background sampler for one physical GPU."""

    def __init__(
        self,
        physical_gpu: int,
        output_path: pathlib.Path,
        interval_seconds: float,
        max_memory_fraction: float,
    ) -> None:
        self.physical_gpu = physical_gpu
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.max_memory_fraction = max_memory_fraction
        self.samples = []  # type: List[GpuSample]
        self.errors = []  # type: List[str]
        self.pressure_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval_seconds * 2.0))

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                sample = query_gpu(self.physical_gpu)
                self.samples.append(sample)
                with self.output_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(sample.as_dict(), sort_keys=True) + "\n")
                if sample.memory_total_mib > 0:
                    fraction = sample.memory_used_mib / sample.memory_total_mib
                    if fraction >= self.max_memory_fraction:
                        self.pressure_event.set()
            except Exception as exc:  # Monitoring failure is recorded, not hidden.
                self.errors.append("{}: {}".format(type(exc).__name__, exc))
            self.stop_event.wait(self.interval_seconds)

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {
                "physical_gpu": self.physical_gpu,
                "sample_count": 0,
                "errors": self.errors,
            }
        return {
            "physical_gpu": self.physical_gpu,
            "sample_count": len(self.samples),
            "peak_utilization_percent_sampled": max(x.utilization_percent for x in self.samples),
            "peak_memory_used_mib_sampled": max(x.memory_used_mib for x in self.samples),
            "memory_total_mib": self.samples[-1].memory_total_mib,
            "peak_temperature_c": max(x.temperature_c for x in self.samples),
            "peak_power_w": max(
                (x.power_w for x in self.samples if x.power_w is not None),
                default=None,
            ),
            "pressure_threshold_fraction": self.max_memory_fraction,
            "pressure_triggered": self.pressure_event.is_set(),
            "errors": self.errors,
        }


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_initial_state_spec(spec: str) -> List[int]:
    """Parse comma-separated indices and half-open ranges such as 0:10,12."""
    result = []  # type: List[int]
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            fields = part.split(":")
            if len(fields) not in (2, 3):
                raise argparse.ArgumentTypeError("invalid range: {}".format(part))
            try:
                start = int(fields[0])
                stop = int(fields[1])
                step = int(fields[2]) if len(fields) == 3 else 1
            except ValueError as exc:
                raise argparse.ArgumentTypeError("invalid integer in {}".format(part)) from exc
            if start < 0 or stop < 0 or step <= 0 or stop <= start:
                raise argparse.ArgumentTypeError("range must satisfy 0 <= start < stop and step > 0")
            result.extend(range(start, stop, step))
        else:
            try:
                index = int(part)
            except ValueError as exc:
                raise argparse.ArgumentTypeError("invalid index: {}".format(part)) from exc
            if index < 0:
                raise argparse.ArgumentTypeError("indices must be non-negative")
            result.append(index)
    if not result:
        raise argparse.ArgumentTypeError("at least one initial-state index is required")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("initial-state indices must be unique")
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded LIBERO evaluation subset against an existing OpenPI server."
    )
    parser.add_argument("--suite", choices=sorted(MAX_STEPS_BY_SUITE), default="libero_spatial")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument(
        "--initial-states",
        type=parse_initial_state_spec,
        required=True,
        metavar="SPEC",
        help="Comma-separated indices and half-open ranges, for example 0:10 or 0,2,4.",
    )
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-wait-seconds", type=float, default=30.0)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--max-control-steps", type=int, default=None)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--mujoco-egl-device-id", type=int, required=True)
    parser.add_argument("--egl-vendor-file", type=pathlib.Path, required=True)
    parser.add_argument("--gpu-sample-interval", type=float, default=1.0)
    parser.add_argument("--max-gpu-memory-fraction", type=float, default=0.90)
    parser.add_argument("--max-baseline-gpu-utilization", type=float, default=10.0)
    parser.add_argument("--max-baseline-gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--policy-config", choices=["pi0_libero"], default="pi0_libero")
    parser.add_argument("--openpi-root", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--expected-openpi-commit", required=True)
    parser.add_argument("--expected-libero-commit", required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=1024 * 1024 * 1024,
        help="Hard limit for this run directory; defaults to 1 GiB.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append safely and skip initial-state indices already present in results.jsonl.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.task_id < 0:
        parser.error("--task-id must be non-negative")
    if args.max_episodes <= 0:
        parser.error("--max-episodes must be positive")
    if len(args.initial_states) > args.max_episodes:
        parser.error("requested initial states exceed --max-episodes")
    if args.replan_steps <= 0:
        parser.error("--replan-steps must be positive")
    if args.num_steps_wait < 0:
        parser.error("--num-steps-wait must be non-negative")
    if args.resize_size <= 0:
        parser.error("--resize-size must be positive")
    if args.server_wait_seconds <= 0:
        parser.error("--server-wait-seconds must be positive")
    if args.gpu_sample_interval <= 0:
        parser.error("--gpu-sample-interval must be positive")
    if not 0.0 < args.max_gpu_memory_fraction <= 1.0:
        parser.error("--max-gpu-memory-fraction must be in (0, 1]")
    if not 0.0 <= args.max_baseline_gpu_utilization <= 100.0:
        parser.error("--max-baseline-gpu-utilization must be in [0, 100]")
    if not 0.0 < args.max_baseline_gpu_memory_fraction <= 1.0:
        parser.error("--max-baseline-gpu-memory-fraction must be in (0, 1]")
    if args.max_output_bytes <= 0:
        parser.error("--max-output-bytes must be positive")
    return args


def query_gpu(physical_gpu: int) -> GpuSample:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, timeout=10)
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6 or int(fields[0]) != physical_gpu:
            continue
        try:
            power = float(fields[5])
        except ValueError:
            power = None
        return GpuSample(
            timestamp_utc=utc_now(),
            physical_gpu=physical_gpu,
            utilization_percent=float(fields[1]),
            memory_used_mib=float(fields[2]),
            memory_total_mib=float(fields[3]),
            temperature_c=float(fields[4]),
            power_w=power,
        )
    raise RuntimeError("physical GPU {} not found in nvidia-smi output".format(physical_gpu))


def wait_for_server(host: str, port: int, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = None  # type: Optional[Exception]
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError("server {}:{} unavailable: {}".format(host, port, last_error))


def atomic_write_json(path: pathlib.Path, value: Dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def append_jsonl(path: pathlib.Path, value: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root: pathlib.Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root)] + list(arguments),
        text=True,
        timeout=30,
    ).strip()


def collect_identity(args: argparse.Namespace) -> Dict[str, Any]:
    openpi_root = args.openpi_root.expanduser().resolve()
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    libero_root = openpi_root / "third_party/libero"
    if not openpi_root.is_dir() or not libero_root.is_dir():
        raise FileNotFoundError("OpenPI or LIBERO checkout not found under {}".format(openpi_root))
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError("checkpoint not found: {}".format(checkpoint_dir))

    openpi_commit = git_output(openpi_root, "rev-parse", "HEAD")
    libero_commit = git_output(libero_root, "rev-parse", "HEAD")
    if openpi_commit != args.expected_openpi_commit:
        raise RuntimeError(
            "OpenPI commit mismatch: expected {}, found {}".format(
                args.expected_openpi_commit, openpi_commit
            )
        )
    if libero_commit != args.expected_libero_commit:
        raise RuntimeError(
            "LIBERO commit mismatch: expected {}, found {}".format(
                args.expected_libero_commit, libero_commit
            )
        )

    checkpoint_files = [path for path in checkpoint_dir.rglob("*") if path.is_file()]
    norm_stats = checkpoint_dir / "assets/physical-intelligence/libero/norm_stats.json"
    if not norm_stats.is_file():
        raise FileNotFoundError("checkpoint norm stats not found: {}".format(norm_stats))
    return {
        "tool_path": str(pathlib.Path(__file__).resolve()),
        "tool_sha256": sha256_file(pathlib.Path(__file__).resolve()),
        "openpi_root": str(openpi_root),
        "openpi_commit": openpi_commit,
        "openpi_branch": git_output(openpi_root, "branch", "--show-current"),
        "libero_commit": libero_commit,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file_count": len(checkpoint_files),
        "checkpoint_total_bytes": sum(path.stat().st_size for path in checkpoint_files),
        "norm_stats_sha256": sha256_file(norm_stats),
    }


def load_completed_indices(path: pathlib.Path) -> List[int]:
    if not path.exists():
        return []
    completed = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            completed.append(int(record["initial_state_index"]))
    return completed


def directory_size(path: pathlib.Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def validate_resume_config(
    output_dir: pathlib.Path,
    plan: Dict[str, Any],
    identity: Dict[str, Any],
) -> None:
    path = output_dir / "run_config.json"
    if not path.is_file():
        raise RuntimeError("--resume requires an existing run_config.json in {}".format(output_dir))
    with path.open(encoding="utf-8") as handle:
        previous = json.load(handle)
    comparisons = {
        "suite": plan["suite"],
        "task_id": plan["task_id"],
        "seed": plan["seed"],
        "policy_config": plan["policy_config"],
        "physical_gpu": plan["physical_gpu"],
        "mujoco_egl_device_id": plan["mujoco_egl_device_id"],
    }
    for key, expected in comparisons.items():
        if previous.get(key) != expected:
            raise RuntimeError(
                "resume mismatch for {}: previous={!r}, requested={!r}".format(
                    key, previous.get(key), expected
                )
            )
    previous_identity = previous.get("identity", {})
    for key in ("openpi_commit", "libero_commit", "checkpoint_dir", "norm_stats_sha256"):
        if previous_identity.get(key) != identity.get(key):
            raise RuntimeError(
                "resume identity mismatch for {}: previous={!r}, current={!r}".format(
                    key, previous_identity.get(key), identity.get(key)
                )
            )


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timing_summary(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def quat_to_axis_angle(quat: Any, np: Any) -> Any:
    quat = np.array(quat, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / denominator


def prepare_output(args: argparse.Namespace) -> pathlib.Path:
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not args.resume:
        raise FileExistsError(
            "output directory already exists; choose a new path or pass --resume: {}".format(output_dir)
        )
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_runtime_mapping(args: argparse.Namespace) -> Dict[str, str]:
    physical_gpu = args.physical_gpu
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be exactly {!r}; got {!r}".format(str(physical_gpu), visible)
        )
    egl_device = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if egl_device != str(args.mujoco_egl_device_id):
        raise RuntimeError(
            "MUJOCO_EGL_DEVICE_ID must be exactly {!r}; got {!r}".format(
                str(args.mujoco_egl_device_id), egl_device
            )
        )
    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("MUJOCO_GL must be exactly 'egl'")
    if os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("PYOPENGL_PLATFORM must be exactly 'egl'")
    expected_vendor = args.egl_vendor_file.expanduser().resolve()
    actual_vendor_raw = os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "")
    actual_vendor = pathlib.Path(actual_vendor_raw).expanduser().resolve() if actual_vendor_raw else None
    if actual_vendor != expected_vendor or not expected_vendor.is_file():
        raise RuntimeError(
            "EGL vendor file mismatch or missing: expected {}, found {}".format(
                expected_vendor, actual_vendor
            )
        )
    return {
        "physical_gpu": str(physical_gpu),
        "cuda_visible_devices": visible,
        "mujoco_egl_device_id": egl_device,
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM", ""),
        "egl_vendor_file": str(actual_vendor),
    }


def run_episode(
    args: argparse.Namespace,
    output_dir: pathlib.Path,
    task: Any,
    initial_state: Any,
    initial_state_index: int,
    client: Any,
    sampler: GpuSampler,
    dependencies: Dict[str, Any],
) -> Dict[str, Any]:
    np = dependencies["np"]
    imageio = dependencies["imageio"]
    image_tools = dependencies["image_tools"]
    OffScreenRenderEnv = dependencies["OffScreenRenderEnv"]
    get_libero_path = dependencies["get_libero_path"]

    task_description = str(task.language)
    bddl_path = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = None
    replay_images = []  # type: List[Any]
    request_wall_ms = []  # type: List[float]
    policy_infer_ms = []  # type: List[float]
    server_infer_ms = []  # type: List[float]
    server_prev_total_ms = []  # type: List[float]
    executed_actions = []  # type: List[List[float]]
    action_plan = None
    success = False
    failure_reason = "max_control_steps"
    exception_type = None
    exception_message = None
    control_steps = 0
    wait_steps = 0
    policy_requests = 0
    episode_started = time.monotonic()

    try:
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_path,
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(initial_state)

        for _ in range(args.num_steps_wait):
            if sampler.pressure_event.is_set():
                raise ResourcePressureError("selected GPU crossed memory pressure threshold")
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            wait_steps += 1
            if done:
                success = True
                failure_reason = None
                break

        import collections

        action_plan = collections.deque()
        max_control_steps = args.max_control_steps or MAX_STEPS_BY_SUITE[args.suite]
        while not success and control_steps < max_control_steps:
            if sampler.pressure_event.is_set():
                raise ResourcePressureError("selected GPU crossed memory pressure threshold")

            image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(image, args.resize_size, args.resize_size)
            )
            wrist_image = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_image, args.resize_size, args.resize_size)
            )
            replay_images.append(image)

            if not action_plan:
                element = {
                    "observation/image": image,
                    "observation/wrist_image": wrist_image,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            quat_to_axis_angle(obs["robot0_eef_quat"], np),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": task_description,
                }
                request_started = time.monotonic()
                response = client.infer(element)
                request_wall_ms.append((time.monotonic() - request_started) * 1000.0)
                policy_requests += 1

                timing = response.get("policy_timing", {})
                server_timing = response.get("server_timing", {})
                if "infer_ms" in timing:
                    policy_infer_ms.append(float(timing["infer_ms"]))
                if "infer_ms" in server_timing:
                    server_infer_ms.append(float(server_timing["infer_ms"]))
                if "prev_total_ms" in server_timing:
                    server_prev_total_ms.append(float(server_timing["prev_total_ms"]))

                actions = np.asarray(response["actions"])
                if actions.ndim != 2 or actions.shape[1] != 7:
                    raise ValueError("unexpected action shape: {}".format(actions.shape))
                if len(actions) < args.replan_steps:
                    raise ValueError(
                        "policy returned {} steps, fewer than replan_steps={}".format(
                            len(actions), args.replan_steps
                        )
                    )
                if not np.isfinite(actions).all():
                    raise ValueError("policy returned non-finite actions")
                action_plan.extend(actions[: args.replan_steps])

            action = action_plan.popleft()
            action_list = action.tolist()
            executed_actions.append(action_list)
            obs, _, done, _ = env.step(action_list)
            control_steps += 1
            if done:
                success = True
                failure_reason = None
                break

    except Exception as exc:
        failure_reason = "exception"
        exception_type = type(exc).__name__
        exception_message = str(exc)[:1000]
        logging.error("episode %s failed: %s", initial_state_index, traceback.format_exc())
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                logging.warning("env.close() failed for episode %s: %s", initial_state_index, exc)

    status = "success" if success else "failure"
    video_name = "task_{:02d}_init_{:02d}_{}.mp4".format(args.task_id, initial_state_index, status)
    video_path = output_dir / video_name
    video_error = None
    if replay_images:
        try:
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
        except Exception as exc:
            video_error = "{}: {}".format(type(exc).__name__, exc)
            logging.error("video write failed for episode %s: %s", initial_state_index, video_error)

    episode_seconds = time.monotonic() - episode_started
    action_trace_name = "task_{:02d}_init_{:02d}_actions.json".format(
        args.task_id, initial_state_index
    )
    action_trace_path = output_dir / action_trace_name
    atomic_write_json(
        action_trace_path,
        {
            "schema_version": 1,
            "suite": args.suite,
            "task_id": args.task_id,
            "initial_state_index": initial_state_index,
            "action_dim": 7,
            "actions": executed_actions,
        },
    )
    result = {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "suite": args.suite,
        "task_id": args.task_id,
        "initial_state_index": initial_state_index,
        "seed": args.seed,
        "task_description": task_description,
        "success": success,
        "failure_reason": failure_reason,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "wait_steps": wait_steps,
        "control_steps": control_steps,
        "saved_frames": len(replay_images),
        "policy_requests": policy_requests,
        "episode_wall_seconds": episode_seconds,
        "request_timings_ms_raw": {
            "client_request_wall": request_wall_ms,
            "policy_infer": policy_infer_ms,
            "server_infer": server_infer_ms,
            "server_prev_total": server_prev_total_ms,
        },
        "timing_ms": {
            "client_request_wall": timing_summary(request_wall_ms),
            "policy_infer": timing_summary(policy_infer_ms),
            "server_infer": timing_summary(server_infer_ms),
            "server_prev_total": timing_summary(server_prev_total_ms),
        },
        "video": {
            "filename": video_name if video_path.exists() else None,
            "bytes": video_path.stat().st_size if video_path.exists() else None,
            "sha256": sha256_file(video_path) if video_path.exists() else None,
            "write_error": video_error,
        },
        "action_trace": {
            "filename": action_trace_name,
            "bytes": action_trace_path.stat().st_size,
            "sha256": sha256_file(action_trace_path),
            "steps": len(executed_actions),
            "action_dim": 7,
        },
    }
    atomic_write_json(
        output_dir / "task_{:02d}_init_{:02d}_result.json".format(args.task_id, initial_state_index),
        result,
    )
    append_jsonl(output_dir / "results.jsonl", result)
    return result


def load_planning_dependencies() -> Dict[str, Any]:
    from libero.libero import benchmark
    import numpy as np

    return {
        "benchmark": benchmark,
        "np": np,
    }


def load_runtime_dependencies() -> Dict[str, Any]:
    import imageio
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy

    return {
        "imageio": imageio,
        "get_libero_path": get_libero_path,
        "OffScreenRenderEnv": OffScreenRenderEnv,
        "image_tools": image_tools,
        "websocket_client_policy": websocket_client_policy,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    output_dir = prepare_output(args)
    identity = collect_identity(args)
    dependencies = load_planning_dependencies()
    benchmark = dependencies["benchmark"]

    np = dependencies["np"]
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    if args.task_id >= suite.n_tasks:
        raise ValueError("task-id {} outside [0, {})".format(args.task_id, suite.n_tasks))
    task = suite.get_task(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    invalid = [index for index in args.initial_states if index >= len(initial_states)]
    if invalid:
        raise ValueError(
            "initial-state indices {} outside [0, {})".format(invalid, len(initial_states))
        )

    plan = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "dry_run": args.dry_run,
        "suite": args.suite,
        "suite_task_count": suite.n_tasks,
        "task_id": args.task_id,
        "task_description": str(task.language),
        "available_initial_states": len(initial_states),
        "requested_initial_states": args.initial_states,
        "episode_count": len(args.initial_states),
        "max_control_steps": args.max_control_steps or MAX_STEPS_BY_SUITE[args.suite],
        "replan_steps": args.replan_steps,
        "seed": args.seed,
        "policy_config": args.policy_config,
        "server": "{}:{}".format(args.host, args.port),
        "physical_gpu": args.physical_gpu,
        "mujoco_egl_device_id": args.mujoco_egl_device_id,
        "output_dir": str(output_dir),
        "max_output_bytes": args.max_output_bytes,
        "identity": identity,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        print("dry_run=ok", flush=True)
        return 0

    runtime_mapping = validate_runtime_mapping(args)
    dependencies.update(load_runtime_dependencies())
    baseline_gpu = query_gpu(args.physical_gpu)
    if baseline_gpu.utilization_percent > args.max_baseline_gpu_utilization:
        raise ResourcePressureError(
            "selected GPU baseline utilization {:.1f}% exceeds {:.1f}%".format(
                baseline_gpu.utilization_percent, args.max_baseline_gpu_utilization
            )
        )
    baseline_memory_fraction = baseline_gpu.memory_used_mib / baseline_gpu.memory_total_mib
    if baseline_memory_fraction > args.max_baseline_gpu_memory_fraction:
        raise ResourcePressureError(
            "selected GPU baseline memory fraction {:.3f} exceeds {:.3f}".format(
                baseline_memory_fraction, args.max_baseline_gpu_memory_fraction
            )
        )
    if args.resume:
        validate_resume_config(output_dir, plan, identity)
    completed = set(load_completed_indices(output_dir / "results.jsonl")) if args.resume else set()
    pending_indices = [index for index in args.initial_states if index not in completed]
    if not pending_indices:
        raise RuntimeError("no pending initial-state indices remain")

    run_config = dict(plan)
    run_config.update(
        {
            "dry_run": False,
            "runtime_mapping": runtime_mapping,
            "baseline_gpu": baseline_gpu.as_dict(),
            "resume": args.resume,
            "already_completed_indices": sorted(completed),
            "pending_initial_states": pending_indices,
            "max_gpu_memory_fraction": args.max_gpu_memory_fraction,
        }
    )
    atomic_write_json(output_dir / "run_config.json", run_config)

    sampler = GpuSampler(
        physical_gpu=args.physical_gpu,
        output_path=output_dir / "gpu_samples.jsonl",
        interval_seconds=args.gpu_sample_interval,
        max_memory_fraction=args.max_gpu_memory_fraction,
    )
    sampler.start()
    client = None
    results = []  # type: List[Dict[str, Any]]
    try:
        wait_for_server(args.host, args.port, args.server_wait_seconds)
        client = dependencies["websocket_client_policy"].WebsocketClientPolicy(args.host, args.port)
        for initial_state_index in pending_indices:
            if sampler.pressure_event.is_set():
                raise ResourcePressureError("selected GPU crossed memory pressure threshold before episode")
            result = run_episode(
                args=args,
                output_dir=output_dir,
                task=task,
                initial_state=initial_states[initial_state_index],
                initial_state_index=initial_state_index,
                client=client,
                sampler=sampler,
                dependencies=dependencies,
            )
            results.append(result)
            output_bytes = directory_size(output_dir)
            print(
                "episode_result initial_state={} success={} control_steps={} requests={} output_bytes={}".format(
                    initial_state_index,
                    result["success"],
                    result["control_steps"],
                    result["policy_requests"],
                    output_bytes,
                ),
                flush=True,
            )
            if output_bytes > args.max_output_bytes:
                raise ResourcePressureError(
                    "output directory {} bytes exceeds limit {}".format(
                        output_bytes, args.max_output_bytes
                    )
                )
    finally:
        if client is not None:
            try:
                client._ws.close()  # The current OpenPI client exposes no public close method.
            except Exception as exc:
                logging.warning("WebSocket close failed: %s", exc)
        sampler.stop()

    all_records = []  # type: List[Dict[str, Any]]
    with (output_dir / "results.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                all_records.append(json.loads(line))
    successes = sum(bool(record["success"]) for record in all_records)
    summary = {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "suite": args.suite,
        "task_id": args.task_id,
        "requested_initial_states": args.initial_states,
        "completed_initial_states": [record["initial_state_index"] for record in all_records],
        "episodes_completed": len(all_records),
        "successes": successes,
        "failures": len(all_records) - successes,
        "success_rate": successes / len(all_records) if all_records else None,
        "total_policy_requests": sum(record["policy_requests"] for record in all_records),
        "total_control_steps": sum(record["control_steps"] for record in all_records),
        "gpu": sampler.summary(),
    }
    atomic_write_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    print("bounded_evaluation=completed", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("interrupted_by_user=true", file=sys.stderr, flush=True)
        sys.exit(130)

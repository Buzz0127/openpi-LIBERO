# Original pi0_libero: LIBERO spatial task 0, initial states 0-9

## Scope

- Model: original `pi0_libero`, not the default pi0.5 policy
- OpenPI commit: `15a9616a00943ada6c20a0f158e3adb39df2ccac`
- LIBERO submodule commit: `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`
- Suite/task: `libero_spatial`, task 0
- Task: pick up the black bowl between the plate and the ramekin and place it on the plate
- Initial states: 0 through 9
- Seed: 7
- Execution: sequential, one physical GPU, `replan_steps=5`, maximum 220 control steps per episode

This is a bounded 10-initial-state result for one LIBERO task. It is not a full LIBERO suite benchmark.

## Result

- Episodes: 10
- Successes: 9
- Failures: 1
- Success rate: 90%
- Total control steps: 899
- Total policy requests: 185

| Initial state | Result | Control steps | Policy requests | Wall time (s) | Failure reason |
| ---: | --- | ---: | ---: | ---: | --- |
| 0 | success | 77 | 16 | 25.733 | - |
| 1 | failure | 220 | 44 | 15.907 | `max_control_steps` |
| 2 | success | 76 | 16 | 7.865 | - |
| 3 | success | 81 | 17 | 8.181 | - |
| 4 | success | 69 | 14 | 9.054 | - |
| 5 | success | 82 | 17 | 9.445 | - |
| 6 | success | 67 | 14 | 9.150 | - |
| 7 | success | 79 | 16 | 9.349 | - |
| 8 | success | 77 | 16 | 8.552 | - |
| 9 | success | 71 | 15 | 8.641 | - |

The first episode includes the first-inference JAX compilation cost. Its first client request took about 16.27 seconds; later policy requests were generally around one tenth of a second.

## GPU protection and isolation

- Physical GPU: 0
- `CUDA_VISIBLE_DEVICES=0`; JAX and PyTorch each saw one process-local `cuda:0`
- MuJoCo EGL context was verified on physical GPU 0
- JAX default memory preallocation disabled
- Guard SHA-256: `ba9f7c74b843a61dc7cdf2f48b182390edb59022b5a2e76a489ce9d7048c4594`
- Guard peak sampled utilization: 92%
- Guard minimum sampled free VRAM: 37.10%
- Evaluator peak sampled used VRAM: 61,569 MiB of 97,887 MiB total
- Guard pauses/resumes/emergencies/monitor errors: 0/0/0/0
- Policy server, evaluator, and guard exited normally; port 8000 was released

## Evidence integrity

- `run_summary.json`: `304f560c5d2fcf83a7bedc45d47201b99b3af3aa2dc173a3968aa45dfd29c782`
- `results.jsonl`: `00ef6c3713cad43309b54daf7e71204938a0b4bd890d345d4ade011a9d34aee5`
- Ten episode result JSON files, ten action traces, and ten MP4 videos are present.

The `evaluation/` directory contains the model outputs. The `control/` directory contains the baseline samples, exact guarded command, guard event log, server/evaluator logs, and recorded task PIDs.

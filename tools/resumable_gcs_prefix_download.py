#!/usr/bin/env python3
"""Fetch a public GCS prefix manifest and download it safely with resume support."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import fcntl


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def opener(proxy: str | None) -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy} if proxy else {})
    )


def request_json(client: urllib.request.OpenerDirector, url: str) -> dict:
    with client.open(urllib.request.Request(url, headers={"User-Agent": "openpi-lora-audit/1"}), timeout=60) as response:
        return json.load(response)


def fetch_manifest(bucket: str, prefix: str, proxy: str | None) -> dict:
    if not prefix or prefix.startswith("/") or not prefix.endswith("/"):
        raise ValueError("prefix must be a non-empty relative path ending in /")
    client = opener(proxy)
    items: list[dict] = []
    token: str | None = None
    while True:
        query = {"prefix": prefix, "fields": "items(name,generation,size,md5Hash,crc32c,etag),nextPageToken"}
        if token:
            query["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{urllib.parse.quote(bucket, safe='')}/o?{urllib.parse.urlencode(query)}"
        payload = request_json(client, url)
        for raw in payload.get("items", []):
            name = raw["name"]
            relative = PurePosixPath(name).relative_to(PurePosixPath(prefix))
            if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
                raise ValueError(f"unsafe object name: {name}")
            if not raw.get("md5Hash") or not raw.get("generation"):
                raise ValueError(f"object lacks immutable identity fields: {name}")
            items.append({
                "name": name,
                "relative_path": relative.as_posix(),
                "generation": str(raw["generation"]),
                "size": int(raw["size"]),
                "md5_b64": raw["md5Hash"],
                "crc32c_b64": raw.get("crc32c"),
                "etag": raw.get("etag"),
            })
        token = payload.get("nextPageToken")
        if not token:
            break
    if not items:
        raise ValueError("GCS prefix is empty")
    items.sort(key=lambda item: item["name"])
    return {
        "schema_version": 1,
        "bucket": bucket,
        "prefix": prefix,
        "fetched_at_utc": utc_now(),
        "object_count": len(items),
        "total_bytes": sum(item["size"] for item in items),
        "objects": items,
    }


def md5_b64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def validate_file(path: Path, item: dict) -> bool:
    return path.is_file() and path.stat().st_size == item["size"] and md5_b64(path) == item["md5_b64"]


def media_url(bucket: str, item: dict) -> str:
    encoded = urllib.parse.quote(item["name"], safe="")
    return f"https://storage.googleapis.com/download/storage/v1/b/{urllib.parse.quote(bucket, safe='')}/o/{encoded}?alt=media&generation={urllib.parse.quote(item['generation'], safe='')}"


def download_one(client: urllib.request.OpenerDirector, bucket: str, item: dict, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if target.exists():
        if validate_file(target, item):
            return
        raise RuntimeError(f"existing completed file failed identity check: {target}")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > item["size"]:
        raise RuntimeError(f"partial is larger than source object: {partial}")
    if offset == item["size"]:
        if md5_b64(partial) != item["md5_b64"]:
            raise RuntimeError(f"complete-sized partial failed MD5: {partial}")
        os.replace(partial, target)
        return
    headers = {"User-Agent": "openpi-lora-audit/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(media_url(bucket, item), headers=headers)
    with client.open(request, timeout=120) as response:
        if offset:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError(f"server did not honor resume offset {offset} for {item['name']}")
        elif response.status not in (200, 206):
            raise RuntimeError(f"unexpected HTTP status {response.status} for {item['name']}")
        with partial.open("ab") as handle:
            while True:
                chunk = response.read(8 << 20)
                if not chunk:
                    break
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    if partial.stat().st_size != item["size"] or md5_b64(partial) != item["md5_b64"]:
        raise RuntimeError(f"downloaded object failed size/MD5: {item['name']}")
    os.replace(partial, target)


def run_download(manifest: dict, scratch: Path, final: Path, proxy: str | None, retries: int, report_path: Path) -> dict:
    if final.exists():
        raise FileExistsError(f"final target already exists: {final}")
    scratch.mkdir(parents=True, exist_ok=True)
    client = opener(proxy)
    for index, item in enumerate(manifest["objects"], start=1):
        target = scratch / item["relative_path"]
        for attempt in range(1, retries + 1):
            try:
                download_one(client, manifest["bucket"], item, target)
                print(json.dumps({"event": "object_verified", "index": index, "count": manifest["object_count"], "name": item["name"], "size": item["size"]}), flush=True)
                break
            except (OSError, urllib.error.URLError, RuntimeError) as error:
                if attempt == retries:
                    raise
                delay = min(60.0, (2 ** (attempt - 1)) + random.random())
                print(json.dumps({"event": "retry", "attempt": attempt, "delay_seconds": delay, "name": item["name"], "error": repr(error)}), flush=True)
                time.sleep(delay)
    for item in manifest["objects"]:
        if not validate_file(scratch / item["relative_path"], item):
            raise RuntimeError(f"final verification failed: {item['name']}")
    report = {
        "schema_version": 1,
        "completed_at_utc": utc_now(),
        "object_count": manifest["object_count"],
        "total_bytes": manifest["total_bytes"],
        "all_size_and_md5_verified": True,
        "scratch": str(scratch),
        "final": str(final),
    }
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(scratch, final)
    atomic_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    manifest = sub.add_parser("manifest")
    manifest.add_argument("--bucket", required=True)
    manifest.add_argument("--prefix", required=True)
    manifest.add_argument("--proxy")
    manifest.add_argument("--output", required=True, type=Path)
    download = sub.add_parser("download")
    download.add_argument("--manifest", required=True, type=Path)
    download.add_argument("--scratch", required=True, type=Path)
    download.add_argument("--final", required=True, type=Path)
    download.add_argument("--proxy")
    download.add_argument("--retries", type=int, default=8)
    download.add_argument("--report", required=True, type=Path)
    download.add_argument("--lock", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "manifest":
        if args.output.exists():
            raise FileExistsError(args.output)
        value = fetch_manifest(args.bucket, args.prefix, args.proxy)
        atomic_json(args.output, value)
        print(json.dumps({"object_count": value["object_count"], "total_bytes": value["total_bytes"]}))
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.retries < 1:
            raise ValueError("retries must be positive")
        if args.report.exists():
            raise FileExistsError(args.report)
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        with args.lock.open("a+b") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            result = run_download(manifest, args.scratch, args.final, args.proxy, args.retries, args.report)
        print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

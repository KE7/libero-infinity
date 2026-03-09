#!/usr/bin/env python3
"""Asset parity audit: compare a local asset tree against the pinned HF snapshot.

Downloads lerobot/libero-assets@HF_REVISION to a local cache if not already
present, then computes SHA-256 checksums for every file in both trees and
produces a machine-readable parity report.

Usage::

    python scripts/run_asset_parity.py
    python scripts/run_asset_parity.py --local-assets /path/to/assets
    python scripts/run_asset_parity.py --hf-cache ~/.cache/libero_infinity/assets  # use existing cache
    python scripts/run_asset_parity.py --out audit/my_report.json
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import shutil
import sys
import tempfile

HF_REPO_ID = "lerobot/libero-assets"
HF_REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"  # full SHA
HF_FULL_SHA = HF_REVISION

DEFAULT_CACHE = pathlib.Path.home() / ".cache" / "libero_infinity" / "assets"
DEFAULT_OUT   = pathlib.Path(__file__).resolve().parents[1] / "audit" / f"asset_parity_{HF_REPO_ID.replace('/', '_')}_{HF_FULL_SHA[:8]}.json"


# ---------------------------------------------------------------------------
# Checksum helpers
# ---------------------------------------------------------------------------

def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _is_hf_internal(rel: str) -> bool:
    """Return True for huggingface_hub's own cache/metadata files.

    snapshot_download() writes .metadata sidecar files and a .cache/ directory
    alongside the real assets. These are internal bookkeeping and must be
    excluded from content comparison.
    """
    parts = pathlib.PurePosixPath(rel).parts
    return (
        parts[0] in (".cache", ".huggingface")
        or rel.endswith(".metadata")
        or rel == ".gitattributes"
    )


def inventory(root: pathlib.Path, *, verbose: bool = False, exclude_hf_internal: bool = False) -> dict[str, str]:
    """Return {relative_path_str: sha256} for every file under *root*."""
    result = {}
    files = sorted(p for p in root.rglob("*") if p.is_file())
    total = len(files)
    skipped = 0
    for i, p in enumerate(files, 1):
        rel = str(p.relative_to(root))
        if exclude_hf_internal and _is_hf_internal(rel):
            skipped += 1
            continue
        if verbose and i % 200 == 0:
            print(f"  hashing {i}/{total} …", flush=True)
        result[rel] = sha256_file(p)
    if skipped:
        print(f"  (skipped {skipped} huggingface_hub internal files)")
    return result


# ---------------------------------------------------------------------------
# HF download
# ---------------------------------------------------------------------------

def download_hf(cache_dir: pathlib.Path, *, force: bool = False) -> pathlib.Path:
    if cache_dir.is_dir() and not force:
        print(f"Using existing HF cache: {cache_dir}")
        return cache_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: uv pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    parent = cache_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp(dir=parent, prefix=".parity_dl_"))
    try:
        print(f"Downloading {HF_REPO_ID}@{HF_REVISION} → {cache_dir} …")
        snapshot_download(
            repo_id=HF_REPO_ID,
            revision=HF_REVISION,
            repo_type="dataset",
            local_dir=str(tmp),
            local_dir_use_symlinks=False,
        )
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        tmp.rename(cache_dir)
        print("Download complete.")
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return cache_dir


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(
    local_inv: dict[str, str],
    hf_inv: dict[str, str],
    *,
    local_root: pathlib.Path,
    hf_root: pathlib.Path,
) -> dict:
    local_keys = set(local_inv)
    hf_keys    = set(hf_inv)

    missing_in_hf   = sorted(local_keys - hf_keys)   # in local but not HF
    added_in_hf     = sorted(hf_keys - local_keys)    # in HF but not local
    common          = local_keys & hf_keys
    hash_match      = sorted(k for k in common if local_inv[k] == hf_inv[k])
    hash_mismatch   = sorted(k for k in common if local_inv[k] != hf_inv[k])

    status = "PASS" if not missing_in_hf and not added_in_hf and not hash_mismatch else "FAIL"

    return {
        "status": status,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "hf_repo": HF_REPO_ID,
        "hf_revision": HF_REVISION,
        "hf_full_sha": HF_FULL_SHA,
        "local_root": str(local_root),
        "hf_root": str(hf_root),
        "summary": {
            "local_file_count": len(local_inv),
            "hf_file_count": len(hf_inv),
            "files_matching": len(hash_match),
            "files_missing_in_hf": len(missing_in_hf),
            "files_added_in_hf": len(added_in_hf),
            "files_hash_mismatch": len(hash_mismatch),
        },
        "missing_in_hf": missing_in_hf,
        "added_in_hf": added_in_hf,
        "hash_mismatches": [
            {"path": k, "local_sha256": local_inv[k], "hf_sha256": hf_inv[k]}
            for k in hash_mismatch
        ],
        "approved_exceptions": [],
    }


def print_human_summary(report: dict) -> None:
    s = report["summary"]
    print()
    print("=" * 60)
    print(f"  Asset Parity Report — {report['status']}")
    print("=" * 60)
    print(f"  HF source  : {report['hf_repo']}@{report['hf_revision']}")
    print(f"  Local root : {report['local_root']}")
    print(f"  Local files: {s['local_file_count']}")
    print(f"  HF files   : {s['hf_file_count']}")
    print(f"  Matching   : {s['files_matching']}")
    print(f"  Missing in HF  : {s['files_missing_in_hf']}")
    print(f"  Added in HF    : {s['files_added_in_hf']}")
    print(f"  Hash mismatches: {s['files_hash_mismatch']}")
    print()

    if report["missing_in_hf"]:
        print("FILES MISSING IN HF (in local but not HF):")
        for f in report["missing_in_hf"]:
            print(f"  - {f}")
        print()

    if report["added_in_hf"]:
        print("FILES ADDED IN HF (not in local):")
        for f in report["added_in_hf"][:20]:
            print(f"  + {f}")
        if len(report["added_in_hf"]) > 20:
            print(f"  … and {len(report['added_in_hf']) - 20} more")
        print()

    if report["hash_mismatches"]:
        print("HASH MISMATCHES:")
        for m in report["hash_mismatches"][:20]:
            print(f"  ~ {m['path']}")
            print(f"      local: {m['local_sha256']}")
            print(f"      hf   : {m['hf_sha256']}")
        if len(report["hash_mismatches"]) > 20:
            print(f"  … and {len(report['hash_mismatches']) - 20} more")
        print()

    print(f"Verdict: {report['status']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parity audit: local libero assets vs HF snapshot")
    parser.add_argument("--local-assets", type=pathlib.Path, required=True,
                        help="Local asset tree root to compare against the pinned HF snapshot")
    parser.add_argument("--hf-cache", type=pathlib.Path, default=DEFAULT_CACHE,
                        help=f"Where to store/read the HF snapshot (default: {DEFAULT_CACHE})")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT,
                        help=f"Output JSON report path (default: {DEFAULT_OUT})")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download from HF even if cache exists")
    args = parser.parse_args()

    if not args.local_assets.is_dir():
        print(f"ERROR: local asset directory not found: {args.local_assets}", file=sys.stderr)
        sys.exit(1)

    # Download HF snapshot (if needed)
    hf_root = download_hf(args.hf_cache, force=args.force_download)

    # Hash both trees
    print(f"\nHashing local tree: {args.local_assets} …")
    local_inv = inventory(args.local_assets, verbose=True)
    print(f"  {len(local_inv)} files hashed.")

    print(f"\nHashing HF tree: {hf_root} …")
    hf_inv = inventory(hf_root, verbose=True, exclude_hf_internal=True)
    print(f"  {len(hf_inv)} files hashed.")

    # Build and write report
    report = build_report(local_inv, hf_inv, local_root=args.local_assets, hf_root=hf_root)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to: {args.out}")

    print_human_summary(report)
    sys.exit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

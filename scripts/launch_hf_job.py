#!/usr/bin/env python
"""
Launcher for Hugging Face Jobs.

Two jobs are available:
  embeddings  QZhou sentence embeddings    (default, a10g-small is enough)
  surprisal   novelty/transience/resonance (gemma-4-31b, needs a100-large)
  endpoint    endpoint predictability      (gemma-4-31b, needs a100-large)

Jobs using sentence units also upload the canonical sentence index, so remote
segmentation -- and the misalignment it caused -- cannot happen.

Usage:
  export HF_TOKEN="hf_xxxxxxxxxxxxxxxx"

  # sentence embeddings
  python scripts/launch_hf_job.py \
    --job embeddings \
    --repo username/penpal-qzhou-embeddings \
    --flavor a10g-small

  # endpoint predictability on an A100
  python scripts/launch_hf_job.py \
    --job endpoint \
    --repo username/penpal-endpoint \
    --flavor a100-large \
    --timeout 4h

After completion, pull the artifacts down:
  python scripts/launch_hf_job.py --download --job endpoint --repo username/penpal-endpoint
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, snapshot_download, run_uv_job
except ImportError:
    print("[ERROR] huggingface_hub package is required.")
    print("Please install it via: pip install huggingface_hub")
    sys.exit(1)


# Per-job settings: remote script, default hardware/timeout, default download
# directory, and the environment the remote script reads its parameters from.
JOBS = {
    "embeddings": {
        "script": "01_compute_embeddings_hf_job.py",
        "name": "qzhou-embeddings-job",
        "flavor": "a10g-small",
        "timeout": "1h",
        "outdir": "data/interim/embeddings",
        "needs_sentence_index": False,
        "env": {
            "MODEL_NAME": "Kingsoft-LLM/QZhou-Embedding",
            "BATCH_SIZE": "16",
        },
    },
    "surprisal": {
        "script": "03_compute_surprisal_metrics_hf_job.py",
        "name": "surprisal-ntr-job",
        # gemma-4-31b in bf16 needs an A100; ~3 forward passes per unit.
        "flavor": "a100-large",
        "timeout": "6h",
        "outdir": "data/processed/surprisal",
        # Sentence units come from the canonical index, which the launcher
        # uploads, so this job never segments text itself.
        "needs_sentence_index": True,
        "env": {
            "MODEL_NAME": "google/gemma-4-31b",
            "UNIT": "sentence",
            "WINDOW_WORDS": "14",
            "MIN_WINDOW_WORDS": "3",
            "COMMON_INDEX_RANGE": "1",
        },
    },
    "endpoint": {
        "script": "04_compute_endpoint_predictability_hf_job.py",
        "name": "endpoint-predictability-job",
        # gemma-4-31b in bf16 needs an A100; ~n+1 forward passes per story.
        "flavor": "a100-large",
        "timeout": "4h",
        "outdir": "data/processed/endpoint",
        "needs_sentence_index": False,
        "env": {
            "MODEL_NAME": "google/gemma-4-31b",
            "TARGET_MODE": "last_sentences",
            "N_TARGET_SENTENCES": "2",
            "TARGET_FRACTION": "0.1",
            "MIN_BODY_SENTENCES": "5",
            "TAIL_FRACTION": "0.2",
        },
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Launch or download an HF Job.")
    p.add_argument("--job", default="embeddings", choices=sorted(JOBS),
                   help="which computation to run remotely (default: embeddings)")
    p.add_argument("--repo", required=True, help="Target HF Dataset repo id (e.g. username/penpal-embeddings)")
    p.add_argument("--input", default="data/interim/annotations/penpal_annotations_final.csv", help="Input CSV file")
    p.add_argument("--flavor", default=None,
                   choices=["a100-large", "a10g-large", "a10g-small", "l4x1", "l4x4", "a10g-largex2", "a10g-largex4", "t4-medium", "t4-small"],
                   help="HF Jobs hardware GPU flavor (default: per-job)")
    p.add_argument("--timeout", default=None, help="Job timeout, e.g. 1h, 30m (default: per-job)")
    p.add_argument("--model", default=None, help="override MODEL_NAME for the remote script")
    p.add_argument("--unit", default=None, choices=["sentence", "window"],
                   help="unit of analysis for the surprisal job (default: per-job)")
    p.add_argument("--sentence-index", default="data/interim/annotations/sentence_index.csv",
                   help="canonical sentence index uploaded for jobs that need it")
    p.add_argument("--outdir", default=None, help="Local directory to download results to (default: per-job)")
    p.add_argument("--download", action="store_true", help="Download completed artifacts from HF Dataset repo to local outdir")
    return p.parse_args()


def main():
    args = parse_args()
    job = JOBS[args.job]

    # Per-job defaults, overridable from the command line.
    flavor = args.flavor or job["flavor"]
    timeout = args.timeout or job["timeout"]
    outdir = args.outdir or job["outdir"]

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] HF_TOKEN environment variable is not set!")
        print("Set your write token via: export HF_TOKEN=\"hf_xxxx...\"")
        sys.exit(1)

    api = HfApi(token=token)

    if args.download:
        print(f"Downloading artifacts from HF Dataset '{args.repo}' into '{outdir}'...")
        out_path = Path(outdir)
        out_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=args.repo,
            repo_type="dataset",
            local_dir=str(out_path),
            token=token,
            ignore_patterns=["input.csv", ".git*"],
        )
        print(f"✅ Download complete! Files saved to {out_path.resolve()}")
        return

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"[ERROR] Input file '{input_file}' not found.")
        sys.exit(1)

    print(f"1. Preparing HF Dataset repository: {args.repo}...")
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=True, exist_ok=True)

    print(f"2. Uploading input file ({input_file}) as 'input.csv' to {args.repo}...")
    api.upload_file(
        path_or_fileobj=str(input_file),
        path_in_repo="input.csv",
        repo_id=args.repo,
        repo_type="dataset",
        token=token,
    )
    print("   Input uploaded successfully.")

    unit = args.unit or job["env"].get("UNIT")
    if job.get("needs_sentence_index") and unit != "window":
        index_file = Path(args.sentence_index)
        if not index_file.exists():
            print(f"[ERROR] Canonical sentence index '{index_file}' not found.")
            print("        Build it first:  python scripts/00b_build_sentence_index.py")
            print("        (or pass --unit window to use fixed-width windows instead)")
            sys.exit(1)
        print(f"2b. Uploading canonical sentence index ({index_file})...")
        api.upload_file(
            path_or_fileobj=str(index_file),
            path_in_repo="sentence_index.csv",
            repo_id=args.repo,
            repo_type="dataset",
            token=token,
        )
        print("   Sentence index uploaded — the remote job will not re-segment.")

    remote_script_path = Path(__file__).parent / job["script"]
    if not remote_script_path.exists():
        print(f"[ERROR] Remote script missing at {remote_script_path}")
        sys.exit(1)

    env = {"TARGET_HF_REPO": args.repo, **job["env"]}
    if args.model:
        env["MODEL_NAME"] = args.model
    if args.unit:
        env["UNIT"] = args.unit

    print(f"3. Submitting HF Job '{args.job}' "
          f"(script: {job['script']}, hardware: {flavor}, timeout: {timeout})...")
    print(f"   Model: {env.get('MODEL_NAME', 'n/a')}")

    try:
        submitted = run_uv_job(
            script=str(remote_script_path.resolve()),
            name=job["name"],
            flavor=flavor,
            timeout=timeout,
            secrets={"HF_TOKEN": token},
            env=env,
        )
        print(f"\n🚀 Job submitted successfully!")
        print(f"   Job ID: {getattr(submitted, 'id', submitted)}")
        print(f"   Track progress at: https://huggingface.co/jobs")
        print("\nOnce completed, download the artifacts locally by running:")
        print(f"   python scripts/launch_hf_job.py --job {args.job} "
              f"--repo {args.repo} --download\n")
    except Exception as e:
        print(f"[ERROR] Failed to launch HF job: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

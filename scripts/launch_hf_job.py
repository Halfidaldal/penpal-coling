#!/usr/bin/env python
"""
Launcher script for Hugging Face Jobs (QZhou Embedding generation).

Usage:
  export HF_TOKEN="hf_xxxxxxxxxxxxxxxx"
  python scripts/launch_hf_job.py \
    --input data/annotations/penpal_annotations_final.csv \
    --repo username/penpal-qzhou-embeddings \
    --flavor a10g-small

After job completion, pull the outputs into your local output/ directory:
  python scripts/launch_hf_job.py --download --repo username/penpal-qzhou-embeddings
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


def parse_args():
    p = argparse.ArgumentParser(description="Launch or download HF Job embeddings.")
    p.add_argument("--repo", required=True, help="Target HF Dataset repo id (e.g. username/penpal-embeddings)")
    p.add_argument("--input", default="data/interim/annotations/penpal_annotations_final.csv", help="Input CSV file")
    p.add_argument("--flavor", default="a10g-small", 
                   choices=["a100-large", "a10g-large", "a10g-small", "l4x1", "l4x4", "a10g-largex2", "a10g-largex4", "t4-medium", "t4-small"],
                   help="HF Jobs hardware GPU flavor")
    p.add_argument("--timeout", default="1h", help="Job timeout (e.g. 1h, 30m)")
    p.add_argument("--outdir", default="data/interim/embeddings", help="Local directory to download results to")
    p.add_argument("--download", action="store_true", help="Download completed artifacts from HF Dataset repo to local outdir")
    return p.parse_args()


def main():
    args = parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[ERROR] HF_TOKEN environment variable is not set!")
        print("Set your write token via: export HF_TOKEN=\"hf_xxxx...\"")
        sys.exit(1)

    api = HfApi(token=token)

    if args.download:
        print(f"Downloading artifacts from HF Dataset '{args.repo}' into '{args.outdir}'...")
        out_path = Path(args.outdir)
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

    # Use the script file path and specify a short name (<100 chars)
    remote_script_path = Path(__file__).parent / "01_compute_embeddings_hf_job.py"
    if not remote_script_path.exists():
        print(f"[ERROR] Remote script missing at {remote_script_path}")
        sys.exit(1)

    print(f"3. Submitting HF Job (Hardware: {args.flavor}, Timeout: {args.timeout})...")
    
    try:
        job = run_uv_job(
            script=str(remote_script_path.resolve()),
            name="qzhou-embeddings-job",
            flavor=args.flavor,
            timeout=args.timeout,
            secrets={"HF_TOKEN": token},
            env={
                "TARGET_HF_REPO": args.repo,
                "MODEL_NAME": "Kingsoft-LLM/QZhou-Embedding",
                "BATCH_SIZE": "16",
            },
        )
        print(f"\n🚀 Job submitted successfully!")
        print(f"   Job ID: {getattr(job, 'id', job)}")
        print(f"   Track progress at: https://huggingface.co/jobs")
        print("\nOnce completed, download the resulting embeddings locally by running:")
        print(f"   python scripts/launch_hf_job.py --repo {args.repo} --download\n")
    except Exception as e:
        print(f"[ERROR] Failed to launch HF job: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

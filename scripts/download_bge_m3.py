#!/usr/bin/env python
import os
from pathlib import Path

MODEL_NAME = "BAAI/bge-m3"
MODEL_DIR = Path(os.environ.get("BGE_M3_MODEL_PATH", "models/bge-m3"))

def download():
    try:
        from huggingface_hub import snapshot_download
        print(f"Downloading {MODEL_NAME} to {MODEL_DIR}...")
        snapshot_download(repo_id=MODEL_NAME, local_dir=str(MODEL_DIR), local_dir_use_symlinks=False)
        print("Download complete!")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    download()

"""Remove inactive anonymous demo workspaces older than the configured TTL."""
import argparse
import os
import shutil
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=int(os.environ.get("WORKSPACE_TTL_DAYS", "14")))
    args = parser.parse_args()
    root = os.path.abspath(os.path.join(os.environ.get("APP_DATA_ROOT", "/data"), "workspaces"))
    if not os.path.isdir(root):
        print("No workspace directory exists yet.")
        return
    cutoff = time.time() - max(args.days, 1) * 86400
    removed = 0
    for entry in os.scandir(root):
        if not entry.is_dir(follow_symlinks=False):
            continue
        try:
            if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                shutil.rmtree(entry.path)
                removed += 1
        except FileNotFoundError:
            continue
    print(f"Removed {removed} inactive workspace(s) older than {args.days} day(s).")

if __name__ == "__main__":
    main()

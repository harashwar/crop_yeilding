"""
download_model.py
-----------------
Downloads crop_yield_model.pkl from GitHub Releases if it doesn't
already exist locally. Called automatically during Render's build step.
"""

import os
import sys
import requests

MODEL_FILE = "crop_yield_model.pkl"

# ── Paste the GitHub Release direct-download URL below ──────────────
# After creating the GitHub Release and uploading the .pkl file,
# get the direct URL (right-click → Copy link) and paste it here.
MODEL_URL = os.getenv("MODEL_URL", "")   # can also be set as Render env var
# ─────────────────────────────────────────────────────────────────────

def download():
    if os.path.exists(MODEL_FILE):
        size_mb = os.path.getsize(MODEL_FILE) / 1e6
        print(f"[OK] {MODEL_FILE} already present ({size_mb:.1f} MB) — skipping download.")
        return

    if not MODEL_URL:
        print("[ERROR] MODEL_URL is not set. Set it as an environment variable in Render.")
        sys.exit(1)

    print(f"[→] Downloading {MODEL_FILE} ...")
    print(f"    URL: {MODEL_URL}")

    with requests.get(MODEL_URL, stream=True, allow_redirects=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(MODEL_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    {pct:.1f}%  ({downloaded/1e6:.1f}/{total/1e6:.1f} MB)", end="", flush=True)
        print()

    size_mb = os.path.getsize(MODEL_FILE) / 1e6
    print(f"[OK] Download complete — {size_mb:.1f} MB saved to '{MODEL_FILE}'")

if __name__ == "__main__":
    download()

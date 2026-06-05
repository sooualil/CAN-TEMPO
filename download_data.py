"""
Download and extract CHD and SAD datasets.

Two modes:

  Auto   — fetches the dataset page and extracts the Dropbox URL automatically:
    python download_data.py --dataset chd
    python download_data.py --dataset sad
    python download_data.py --dataset all

  Manual — provide the Dropbox URL directly (e.g. if the page structure changes):
    python download_data.py --dataset chd --url "https://www.dropbox.com/..."
    python download_data.py --dataset sad --url "https://www.dropbox.com/..."

Datasets are extracted to --data_root (default: ./data/raw/).
"""

import os
import re
import sys
import zipfile
import argparse
import requests

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

DATASET_PAGES = {
    "chd": "https://ocslab.hksecurity.net/Datasets/car-hacking-dataset",
    "sad": "https://ocslab.hksecurity.net/Datasets/survival-ids",
}

DATASET_TARGETS = {
    "chd": "Car-Hacking",
    "sad": "Survival",
}

DROPBOX_RE  = re.compile(r'href="(https://www\.dropbox\.com/[^"]+)"')
_PW_RE      = re.compile(r'\(PW:\s*([^)]+)\)')


def fetch_page_info(page_url):
    from html import unescape
    print(f"Fetching dataset page: {page_url} ...", flush=True)
    r = requests.get(page_url, timeout=30)
    r.raise_for_status()

    matches = DROPBOX_RE.findall(r.text)
    if not matches:
        raise RuntimeError(f"No Dropbox URL found on {page_url}")
    url = unescape(matches[0])
    url = re.sub(r'[?&]dl=\d', '', url)
    url += ("&dl=1" if "?" in url else "?dl=1")
    print(f"  Found: {url}", flush=True)

    pw_match = _PW_RE.search(r.text)
    pwd = pw_match.group(1).encode() if pw_match else None
    if pwd:
        print(f"  Password found on page.", flush=True)
    return url, pwd


def download(url, dest):
    print(f"Downloading {os.path.basename(dest)} ...", flush=True)
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(
            f"HTTP {r.status_code}. The dataset may require a password or the link expired.\n"
            f"Try providing the URL manually with --url."
        )
    total = int(r.headers.get("content-length", 0))
    if HAS_TQDM and total:
        bar = tqdm(total=total, unit="B", unit_scale=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            if HAS_TQDM and total:
                bar.update(len(chunk))
    if HAS_TQDM and total:
        bar.close()


def extract(zip_path, target_dir, pwd=None):
    print(f"Extracting to {target_dir} ...", flush=True)
    os.makedirs(target_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir, pwd=pwd)
    os.remove(zip_path)
    print("  Done.", flush=True)


def restructure_survival(sad_dir):
    """Lift vehicle folders out of the nested dataset/ subdirectory."""
    dataset_dir = os.path.join(sad_dir, "dataset")
    if not os.path.isdir(dataset_dir):
        return
    import shutil
    for name in os.listdir(dataset_dir):
        src = os.path.join(dataset_dir, name)
        dst = os.path.join(sad_dir, name)
        shutil.move(src, dst)
    os.rmdir(dataset_dir)
    print("  Restructured Survival directory.", flush=True)


def generate_benign_csv(chd_dir):
    """Convert normal_run_data.txt to benign_dataset.csv (mirrors the paper's awk command)."""
    txt = os.path.join(chd_dir, "normal_run_data", "normal_run_data.txt")
    csv = os.path.join(chd_dir, "benign_dataset.csv")
    if not os.path.exists(txt):
        print(f"  Warning: {txt} not found — benign_dataset.csv not generated.", flush=True)
        return
    print(f"Generating benign_dataset.csv ...", flush=True)
    # line format: "Timestamp: <ts>  ID: <id>  000  DLC: <dlc>  b0 b1 .. b7"
    # awk fields:   $1       $2  $3   $4  $5   $6    $7    $8-$15
    with open(txt, "r") as fin, open(csv, "w") as fout:
        for line in fin:
            parts = line.split()
            if len(parts) < 15:
                continue
            row = [parts[1], parts[3], parts[6]] + parts[7:15] + ["R"]
            fout.write(",".join(row) + "\n")
    print(f"  Saved → {csv}", flush=True)


def process(key, url, data_root):
    tgt_dir  = os.path.join(data_root, DATASET_TARGETS[key])
    zip_path = os.path.join(data_root, f"{key}.zip")

    if os.path.isdir(tgt_dir):
        print(f"{tgt_dir} already exists — skipping.", flush=True)
        return

    try:
        page_url, pwd = fetch_page_info(DATASET_PAGES[key])
        if url is None:
            url = page_url
        download(url, zip_path)
        extract(zip_path, tgt_dir, pwd=pwd)
        if key == "sad":
            inner = os.path.join(tgt_dir, "survival.zip")
            if os.path.exists(inner):
                extract(inner, tgt_dir, pwd=pwd)
            restructure_survival(tgt_dir)
        if key == "chd":
            generate_benign_csv(tgt_dir)
    except Exception as e:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print(f"ERROR [{key}]: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   choices=["chd", "sad", "all"], default="all")
    p.add_argument("--url",       default=None,
                   help="Dropbox URL (manual mode — bypasses page fetch)")
    p.add_argument("--data_root", default="./data/raw")
    args = p.parse_args()

    if args.url and args.dataset == "all":
        p.error("--url requires a specific --dataset (chd or sad), not 'all'")

    os.makedirs(args.data_root, exist_ok=True)
    keys = ["chd", "sad"] if args.dataset == "all" else [args.dataset]
    for key in keys:
        process(key, args.url if args.dataset != "all" else None, args.data_root)


if __name__ == "__main__":
    main()

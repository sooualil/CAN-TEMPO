"""
Creates a virtual environment and installs all dependencies.

Usage:
  python setup_env.py
  python setup_env.py --env_dir ./venv
"""

import os
import sys
import argparse
import subprocess


def run(cmd):
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_dir", default=".venv",
                   help="Path for the virtual environment (default: .venv)")
    args = p.parse_args()

    env_dir = args.env_dir
    req     = os.path.join(os.path.dirname(__file__), "requirements.txt")

    print(f"Creating virtual environment at {env_dir} ...", flush=True)
    run([sys.executable, "-m", "venv", env_dir])

    pip = os.path.join(env_dir, "Scripts" if os.name == "nt" else "bin", "pip")

    print("Installing dependencies ...", flush=True)
    run([pip, "install", "--upgrade", "pip"])
    run([pip, "install", "-r", req])

    activate = (
        os.path.join(env_dir, "Scripts", "activate")
        if os.name == "nt"
        else f"source {os.path.join(env_dir, 'bin', 'activate')}"
    )
    print(f"\nDone. Activate with:\n  {activate}", flush=True)


if __name__ == "__main__":
    main()

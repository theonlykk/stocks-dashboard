import os
import subprocess
import sys

os.environ["PORT"] = "7860"

print("[*] Step 1/3 — Downloading price data from yfinance...")
subprocess.run([sys.executable, "download.py"], check=True)

print("[*] Step 2/3 — Precomputing cointegration metrics (single-threaded)...")
subprocess.run([sys.executable, "precompute.py", "--no-ray"], check=True)

print("[*] Step 3/3 — Starting Flask server on port 7860...")
subprocess.run([sys.executable, "server.py"])

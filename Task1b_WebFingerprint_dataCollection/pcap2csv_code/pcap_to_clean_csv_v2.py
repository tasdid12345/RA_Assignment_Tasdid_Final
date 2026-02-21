import os, glob, subprocess
from pathlib import Path
import pandas as pd

PCAP_DIR  = r"E:\Changgangwang sir\data\pcap"
RAW_DIR   = r"E:\Changgangwang sir\data\csv_raw"
CLEAN_DIR = r"E:\Changgangwang sir\data\csv_clean"

# Use the FULL tshark path (most reliable on Windows)
TSHARK = r"C:\Program Files\Wireshark\tshark.exe"

# Put your IPv4 here (from ipconfig) — based on your output it's likely 192.168.0.119
MY_IPV4 = "192.168.0.119"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEAN_DIR, exist_ok=True)

pcaps = sorted(glob.glob(os.path.join(PCAP_DIR, "*.pcapng"))) + sorted(glob.glob(os.path.join(PCAP_DIR, "*.pcap")))
print("PCAP files found:", len(pcaps))
if not pcaps:
    raise SystemExit("No pcap/pcapng files found. Check PCAP_DIR.")

for i, p in enumerate(pcaps, 1):
    base = Path(p).stem
    raw_csv = os.path.join(RAW_DIR, f"{base}.csv")
    clean_csv = os.path.join(CLEAN_DIR, f"{base}_clean.csv")

    # 1) PCAP -> RAW CSV
    cmd = [
        TSHARK, "-n",
        "-r", p,
        "-T", "fields",
        "-e", "frame.time_relative",
        "-e", "frame.len",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-E", "header=y",
        "-E", "separator=,",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)

    if res.returncode != 0:
        print(f"\n[ERROR tshark] {base}")
        print(res.stderr[:1200])
        continue

    if not res.stdout.strip():
        print(f"\n[EMPTY tshark stdout] {base}")
        print("stderr:", res.stderr[:300])
        continue

    with open(raw_csv, "w", encoding="utf-8", newline="") as f:
        f.write(res.stdout)

    # 2) RAW CSV -> CLEAN CSV
    df = pd.read_csv(raw_csv).dropna(subset=["frame.time_relative", "frame.len", "ip.src"])
    df["time"] = df["frame.time_relative"].astype(float)
    df["size"] = df["frame.len"].astype(float)
    df["direction"] = df["ip.src"].astype(str).apply(lambda x: 1.0 if x == MY_IPV4 else -1.0)

    df[["time", "size", "direction"]].to_csv(clean_csv, index=False)

    if i <= 1:
        print("\nPreview of first clean file:", clean_csv)
        print(pd.read_csv(clean_csv).head(8).to_string(index=False))

    print(f"[{i}/{len(pcaps)}] OK -> {base}")

print("\nDONE.")
print("RAW_DIR  :", RAW_DIR)
print("CLEAN_DIR:", CLEAN_DIR)

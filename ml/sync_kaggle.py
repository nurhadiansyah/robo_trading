"""
Helper utility to push, monitor, and pull Deep Learning artifacts from Kaggle
Target Kernel: nurhadiansyah/xauusd-deep-learning-1m-train
"""

import os
import sys
import time
import argparse
import subprocess

KERNEL_ID = "nurhadiansyah/xauusd-deep-learning-1m-train"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KAGGLE_DIR = os.path.join(BASE_DIR, "ml", "kaggle_train")
WEIGHTS_DIR = os.path.join(BASE_DIR, "ml", "weights")

def push_kernel():
    print("=" * 65)
    print(f"[Kaggle Sync] Mengunggah kernel ke Kaggle Cloud: {KERNEL_ID}")
    print(f"              Direktori lokal: {KAGGLE_DIR}")
    print("=" * 65)
    cmd = ["kaggle", "kernels", "push", "-p", KAGGLE_DIR]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("[Stderr]", res.stderr)
    return res.returncode == 0

def check_status():
    cmd = ["kaggle", "kernels", "status", KERNEL_ID]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout.strip()
    print(f"[Kaggle Status] {out}")
    return out

def pull_output():
    print("=" * 65)
    print(f"[Kaggle Sync] Mengunduh output bobot model dari Kaggle...")
    print(f"              Direktori tujuan: {WEIGHTS_DIR}")
    print("=" * 65)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    cmd = ["kaggle", "kernels", "output", KERNEL_ID, "-p", WEIGHTS_DIR]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("[Stderr]", res.stderr)
    return res.returncode == 0

def wait_for_completion(poll_interval=30, max_wait_sec=3600):
    print(f"[Kaggle Sync] Menunggu proses pelatihan di cloud selesai (cek tiap {poll_interval} detik)...")
    start = time.time()
    while True:
        status_str = check_status()
        if "complete" in status_str.lower():
            print("\n>>> [SUKSES] Pelatihan di Kaggle telah SELESAI! Mengunduh artefak bobot...")
            pull_output()
            break
        elif "error" in status_str.lower() or "failed" in status_str.lower():
            print("\n>>> [GAGAL] Pelatihan di Kaggle mengalami kesalahan.")
            break
        elif "cancel" in status_str.lower():
            print("\n>>> [DIBATALKAN] Kernel dibatalkan.")
            break
            
        elapsed = time.time() - start
        if elapsed > max_wait_sec:
            print(f"\n[Timeout] Batas waktu tunggu ({max_wait_sec} detik) tercapai.")
            break
        time.sleep(poll_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Model Synchronization Utility")
    parser.add_argument("--push", action="store_true", help="Push kernel ke Kaggle")
    parser.add_argument("--status", action="store_true", help="Cek status kernel")
    parser.add_argument("--pull", action="store_true", help="Tarik output bobot model dari Kaggle")
    parser.add_argument("--wait", action="store_true", help="Tunggu hingga selesai lalu unduh otomatis")
    args = parser.parse_args()

    if args.push:
        if push_kernel() and args.wait:
            wait_for_completion()
    elif args.status:
        check_status()
    elif args.pull:
        pull_output()
    elif args.wait:
        wait_for_completion()
    else:
        parser.print_help()

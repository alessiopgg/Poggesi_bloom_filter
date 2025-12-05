# main_iso_final.py
# Benchmark Scientifico Completo: workers × chunks × split × size
# Include: warmup, wall-clock + CPU time, min/max/std, system info.

import time
import math
import asyncio
import csv
import os
import gc
import statistics
import platform
from pathlib import Path
from sequential_bloom_filter import SequentialBloomFilter
from hybrid_bloom_filter import HybridBloomFilter


# ============================================================
# CONFIGURAZIONE
# ============================================================

BASE_DIR = Path("bench_data")

DATASET_SIZES = [2_000_000, 5_000_000, 10_000_000, 20_000_000]
SPLIT_COUNTS = [1, 2, 4, 8, 16]

NUM_ROUNDS = 10        # ripetizioni misurate
WARMUP = 2             # primi run ignorati
SPLIT_FACTOR = 4
DESIRED_FPR = 0.01

CSV_OUT = "results_final.csv"
SYSTEM_INFO_FILE = "system_info.txt"

WORKER_CANDIDATES = [1, 2, 4, 8, 16, 32]

CHUNK_SIZES = ["auto", 5000, 10000, 20000, 50000, 100000]


# ============================================================
# PARAMETRI OTTIMALI BLOOM FILTER
# ============================================================

def _get_optimal_params(n: int, p: float):
    if n == 0:
        return 1000, 3
    ln2 = math.log(2.0)
    m = int(math.ceil(-n * math.log(p) / (ln2 ** 2)))
    k = max(1, int(round((m / n) * ln2)))
    return m, k


def _dynamic_chunk(total_items: int, factor: int) -> int:
    cpu = os.cpu_count() or 1
    chunk = math.ceil(total_items / (cpu * factor))
    return max(5000, chunk)


# ============================================================
# RACCOLTA INFO SISTEMA
# ============================================================

def write_system_info():
    with open(SYSTEM_INFO_FILE, "w") as f:
        f.write("=== SYSTEM INFO ===\n")
        f.write(f"OS: {platform.system()} {platform.release()}\n")
        f.write(f"Python version: {platform.python_version()}\n\n")

        f.write("--- CPU ---\n")
        f.write(f"Cores (logical): {os.cpu_count()}\n")
        f.write(f"Processor: {platform.processor()}\n\n")

        try:
            import psutil
            ram = round(psutil.virtual_memory().total / (1024**3), 2)
            f.write(f"RAM: {ram} GB\n")
        except:
            f.write("RAM: psutil non installato\n")

    print(f"Saved system info → {SYSTEM_INFO_FILE}")


# ============================================================
# BENCHMARK SCIENTIFICO
# ============================================================

def run_benchmark():

    write_system_info()

    print(f"\n--- BENCHMARK COMPLETO ({NUM_ROUNDS} rounds, {WARMUP} warmup) ---")
    print(f"Output → {CSV_OUT}\n")

    if not BASE_DIR.exists():
        print("ERRORE: genera prima i dataset!")
        return

    cpu = os.cpu_count() or 1
    max_workers_allowed = cpu * 2
    worker_list = [w for w in WORKER_CANDIDATES if w <= max_workers_allowed]

    print(f"CPU = {cpu}")
    print(f"Testing WORKERS = {worker_list}")
    print(f"Testing CHUNK SIZES = {CHUNK_SIZES}\n")

    # ====================================================
    # CSV HEADER
    # ====================================================

    fieldnames = [
        "size", "split", "workers", "chunk",

        # Sequenziale
        "seq_build_mean", "seq_build_min", "seq_build_max", "seq_build_std",
        "seq_verify_mean", "seq_verify_min", "seq_verify_max", "seq_verify_std",
        "seq_total_mean",

        # Ibrido
        "hyb_build_mean", "hyb_build_min", "hyb_build_max", "hyb_build_std",
        "hyb_verify_mean", "hyb_verify_min", "hyb_verify_max", "hyb_verify_std",
        "hyb_total_mean",

        # CPU times
        "seq_cpu_mean", "hyb_cpu_mean",

        # Speedup
        "speedup",

        # Winner
        "winner"
    ]

    with open(CSV_OUT, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
        writer.writeheader()

        # ====================================================
        # MAIN TEST LOOPS
        # ====================================================

        for size in DATASET_SIZES:

            print(f"\n=== SIZE = {size} ===")

            m, k = _get_optimal_params(size, DESIRED_FPR)
            auto_chunk = _dynamic_chunk(size, SPLIT_FACTOR)

            size_dir = BASE_DIR / f"size_{size}"

            for split in SPLIT_COUNTS:

                split_dir = size_dir / f"split_{split}"
                train_files = sorted((split_dir / "train").glob("*.txt"))
                test_files = sorted((split_dir / "test").glob("*.txt"))

                if not train_files:
                    continue

                print(f"\n→ Split={split} ({len(train_files)} train files)")

                for workers in worker_list:
                    print(f"   Workers={workers}")

                    for chunk_choice in CHUNK_SIZES:

                        chunk = auto_chunk if chunk_choice == "auto" else int(chunk_choice)
                        print(f"      Chunk={chunk}")

                        # valori misurati
                        seq_build_times = []
                        seq_verify_times = []
                        seq_cpu_times = []

                        hyb_build_times = []
                        hyb_verify_times = []
                        hyb_cpu_times = []

                        # ---------------------------
                        # ROUNDS con warmup
                        # ---------------------------

                        for r in range(NUM_ROUNDS + WARMUP):

                            gc.collect()

                            # ========================================
                            # SEQUENZIALE
                            # ========================================
                            bf_seq = SequentialBloomFilter(size=m, hash_count=k)

                            t_cpu0 = time.process_time()

                            t0 = time.perf_counter()
                            bf_seq.build(train_files)
                            t_seq_b = time.perf_counter() - t0

                            t0 = time.perf_counter()
                            bf_seq.verify_from_paths(test_files)
                            t_seq_v = time.perf_counter() - t0

                            t_cpu = time.process_time() - t_cpu0

                            if r >= WARMUP:
                                seq_build_times.append(t_seq_b)
                                seq_verify_times.append(t_seq_v)
                                seq_cpu_times.append(t_cpu)

                            # ========================================
                            # IBRIDO
                            # ========================================
                            bf_hyb = HybridBloomFilter(size=m, hash_count=k)

                            t_cpu0 = time.process_time()

                            t0 = time.perf_counter()
                            asyncio.run(bf_hyb.build_async(train_files, chunk_size=chunk, max_workers=workers))
                            t_hyb_b = time.perf_counter() - t0

                            t0 = time.perf_counter()
                            asyncio.run(bf_hyb.verify_async(test_files, chunk_size=chunk, max_workers=workers))
                            t_hyb_v = time.perf_counter() - t0

                            t_cpu = time.process_time() - t_cpu0

                            if r >= WARMUP:
                                hyb_build_times.append(t_hyb_b)
                                hyb_verify_times.append(t_hyb_v)
                                hyb_cpu_times.append(t_cpu)

                        # ====================================================
                        # STATISTICHE
                        # ====================================================

                        seq_total_mean = statistics.mean(seq_build_times) + statistics.mean(seq_verify_times)
                        hyb_total_mean = statistics.mean(hyb_build_times) + statistics.mean(hyb_verify_times)

                        speedup = seq_total_mean / hyb_total_mean
                        winner = "HYB" if hyb_total_mean < seq_total_mean else "SEQ"

                        # ====================================================
                        # SALVATAGGIO CSV
                        # ====================================================

                        writer.writerow({

                            "size": size,
                            "split": split,
                            "workers": workers,
                            "chunk": chunk_choice,

                            # Sequenziale
                            "seq_build_mean": statistics.mean(seq_build_times),
                            "seq_build_min": min(seq_build_times),
                            "seq_build_max": max(seq_build_times),
                            "seq_build_std": statistics.stdev(seq_build_times),

                            "seq_verify_mean": statistics.mean(seq_verify_times),
                            "seq_verify_min": min(seq_verify_times),
                            "seq_verify_max": max(seq_verify_times),
                            "seq_verify_std": statistics.stdev(seq_verify_times),

                            "seq_total_mean": seq_total_mean,

                            # Ibrido
                            "hyb_build_mean": statistics.mean(hyb_build_times),
                            "hyb_build_min": min(hyb_build_times),
                            "hyb_build_max": max(hyb_build_times),
                            "hyb_build_std": statistics.stdev(hyb_build_times),

                            "hyb_verify_mean": statistics.mean(hyb_verify_times),
                            "hyb_verify_min": min(hyb_verify_times),
                            "hyb_verify_max": max(hyb_verify_times),
                            "hyb_verify_std": statistics.stdev(hyb_verify_times),

                            "hyb_total_mean": hyb_total_mean,

                            # CPU Time
                            "seq_cpu_mean": statistics.mean(seq_cpu_times),
                            "hyb_cpu_mean": statistics.mean(hyb_cpu_times),

                            # Speedup
                            "speedup": round(speedup, 3),

                            "winner": winner
                        })

                        csv_f.flush()

    print("\nBenchmark COMPLETATO con successo!")
    print(f"Risultati salvati in → {CSV_OUT}")
    print(f"System info salvate in → {SYSTEM_INFO_FILE}")


if __name__ == "__main__":
    run_benchmark()

# main.py
# Benchmark finale e stabile per Sequential vs Hybrid Bloom Filter
# Include:
# - misure end-to-end
# - misure compute-only
# - workers × chunk × split × size
# - warmup + misure multiple

import time
import math
import asyncio
import csv
import os
import gc
from pathlib import Path
from sequential_bloom_filter import SequentialBloomFilter
from hybrid_bloom_filter import HybridBloomFilter


# ===========================================================
# CONFIGURAZIONE GLOBALE
# ===========================================================

BASE_DIR = Path("bench_data")

DATASET_SIZES = [2_000_000, 5_000_000, 10_000_000, 20_000_000]
SPLIT_COUNTS = [1, 4, 16]

WORKER_LIST = [1, 4, 8, 16]
CHUNK_SIZES = ["auto", 20000, 50000]

WARMUP = 1
NUM_ROUNDS = 3

DESIRED_FPR = 0.01
CSV_OUT = "results_final_with_compute.csv"


# ===========================================================
# FUNZIONI DI SUPPORTO
# ===========================================================

def _get_optimal_params(n: int, p: float):
    if n == 0:
        return 1000, 3
    ln2 = math.log(2.0)
    m = int(math.ceil(-n * math.log(p) / (ln2 ** 2)))
    k = max(1, int(round((m / n) * ln2)))
    return m, k


def _dynamic_chunk(total_items: int):
    cpu = os.cpu_count() or 1
    chunk = total_items // (cpu * 4)
    return max(5000, chunk)


def save_system_info():
    """Salva informazioni sull'hardware per la relazione."""
    import platform
    with open("system_info.txt", "w") as f:
        f.write("=== SYSTEM INFO ===\n")
        f.write(f"CPU COUNT: {os.cpu_count()}\n")
        f.write(f"OS: {platform.platform()}\n")
        f.write(f"Python: {platform.python_version()}\n")
        f.write(f"Processor: {platform.processor()}\n")
    print("Saved system info -> system_info.txt")


# ===========================================================
# BENCHMARK
# ===========================================================

def run_benchmark():

    save_system_info()

    print(f"\n--- BENCHMARK COMPLETO ({NUM_ROUNDS} rounds, {WARMUP} warmup) ---")
    print(f"Output -> {CSV_OUT}\n")

    if not BASE_DIR.exists():
        print("ERRORE: dataset non trovato. Genera i dati prima.")
        return

    cpu = os.cpu_count() or 1
    print(f"CPU = {cpu}")
    print(f"Testing WORKERS = {WORKER_LIST}")
    print(f"Testing CHUNK SIZES = {CHUNK_SIZES}\n")

    # CSV HEADER
    fieldnames = [
        "size", "split", "workers", "chunk",

        # End-to-end
        "seq_build", "seq_verify", "seq_total",
        "hyb_build", "hyb_verify", "hyb_total",
        "speedup", "winner",

        # Compute-only
        "seq_build_compute", "seq_verify_compute", "seq_total_compute",
        "hyb_build_compute", "hyb_verify_compute", "hyb_total_compute",
        "speedup_compute", "winner_compute"
    ]

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # ======================================================
        # CICLO PRINCIPALE
        # ======================================================
        for size in DATASET_SIZES:

            print(f"\n=== SIZE = {size} ===\n")

            m, k = _get_optimal_params(size, DESIRED_FPR)
            auto_chunk = _dynamic_chunk(size)

            for split in SPLIT_COUNTS:

                split_dir = BASE_DIR / f"size_{size}" / f"split_{split}"
                train_files = sorted((split_dir / "train").glob("*.txt"))
                test_files = sorted((split_dir / "test").glob("*.txt"))

                if not train_files:
                    continue

                print(f"-> Split={split} ({len(train_files)} train files)")

                for workers in WORKER_LIST:

                    print(f"   Workers={workers}")

                    for chunk_option in CHUNK_SIZES:

                        chunk = auto_chunk if chunk_option == "auto" else chunk_option
                        print(f"      Chunk={chunk}")

                        # Somme end-to-end
                        sum_seq_b = sum_seq_v = 0.0
                        sum_hyb_b = sum_hyb_v = 0.0

                        # Somme compute-only
                        sum_seq_b_comp = sum_seq_v_comp = 0.0
                        sum_hyb_b_comp = sum_hyb_v_comp = 0.0

                        # ---------------------------
                        # WARMUP (non registrato)
                        # ---------------------------
                        gc.collect()
                        try:
                            bf_warm = HybridBloomFilter(size=m, hash_count=k)
                            asyncio.run(
                                bf_warm.build_async(
                                    train_files,
                                    chunk_size=chunk,
                                    max_workers=workers,
                                )
                            )
                            asyncio.run(
                                bf_warm.verify_async(
                                    test_files,
                                    chunk_size=chunk,
                                    max_workers=workers,
                                )
                            )
                        except Exception:
                            pass

                        # ---------------------------
                        # BENCHMARK UFFICIALE
                        # ---------------------------
                        for _ in range(NUM_ROUNDS):

                            gc.collect()

                            # === Sequential ===
                            bf_seq = SequentialBloomFilter(size=m, hash_count=k)

                            t0 = time.perf_counter()
                            bf_seq.build(train_files)
                            t_seq_b = time.perf_counter() - t0
                            t_seq_b_comp = getattr(bf_seq, "last_build_compute_time", 0.0)

                            t0 = time.perf_counter()
                            bf_seq.verify_from_paths(test_files)
                            t_seq_v = time.perf_counter() - t0
                            t_seq_v_comp = getattr(bf_seq, "last_verify_compute_time", 0.0)

                            sum_seq_b += t_seq_b
                            sum_seq_v += t_seq_v
                            sum_seq_b_comp += t_seq_b_comp
                            sum_seq_v_comp += t_seq_v_comp

                            # === Hybrid ===
                            bf_hyb = HybridBloomFilter(size=m, hash_count=k)

                            t0 = time.perf_counter()
                            asyncio.run(
                                bf_hyb.build_async(
                                    train_files,
                                    chunk_size=chunk,
                                    max_workers=workers,
                                )
                            )
                            t_hyb_b = time.perf_counter() - t0
                            t_hyb_b_comp = getattr(bf_hyb, "last_build_compute_time", 0.0)

                            t0 = time.perf_counter()
                            asyncio.run(
                                bf_hyb.verify_async(
                                    test_files,
                                    chunk_size=chunk,
                                    max_workers=workers,
                                )
                            )
                            t_hyb_v = time.perf_counter() - t0
                            t_hyb_v_comp = getattr(bf_hyb, "last_verify_compute_time", 0.0)

                            sum_hyb_b += t_hyb_b
                            sum_hyb_v += t_hyb_v
                            sum_hyb_b_comp += t_hyb_b_comp
                            sum_hyb_v_comp += t_hyb_v_comp

                        # ---------------------------
                        # CALCOLO MEDIE END-TO-END
                        # ---------------------------
                        seq_build = sum_seq_b / NUM_ROUNDS
                        seq_verify = sum_seq_v / NUM_ROUNDS
                        seq_total = seq_build + seq_verify

                        hyb_build = sum_hyb_b / NUM_ROUNDS
                        hyb_verify = sum_hyb_v / NUM_ROUNDS
                        hyb_total = hyb_build + hyb_verify

                        speedup = seq_total / hyb_total if hyb_total > 0 else 0.0
                        winner = "HYB" if hyb_total < seq_total else "SEQ"

                        # ---------------------------
                        # CALCOLO MEDIE COMPUTE-ONLY
                        # ---------------------------
                        seq_build_compute = sum_seq_b_comp / NUM_ROUNDS
                        seq_verify_compute = sum_seq_v_comp / NUM_ROUNDS
                        seq_total_compute = seq_build_compute + seq_verify_compute

                        hyb_build_compute = sum_hyb_b_comp / NUM_ROUNDS
                        hyb_verify_compute = sum_hyb_v_comp / NUM_ROUNDS
                        hyb_total_compute = hyb_build_compute + hyb_verify_compute

                        speedup_compute = (
                            seq_total_compute / hyb_total_compute
                            if hyb_total_compute > 0 else 0.0
                        )
                        winner_compute = (
                            "HYB" if hyb_total_compute < seq_total_compute else "SEQ"
                        )

                        writer.writerow({
                            "size": size,
                            "split": split,
                            "workers": workers,
                            "chunk": chunk_option,

                            # End-to-end
                            "seq_build": round(seq_build, 4),
                            "seq_verify": round(seq_verify, 4),
                            "seq_total": round(seq_total, 4),
                            "hyb_build": round(hyb_build, 4),
                            "hyb_verify": round(hyb_verify, 4),
                            "hyb_total": round(hyb_total, 4),
                            "speedup": round(speedup, 3),
                            "winner": winner,

                            # Compute-only
                            "seq_build_compute": round(seq_build_compute, 4),
                            "seq_verify_compute": round(seq_verify_compute, 4),
                            "seq_total_compute": round(seq_total_compute, 4),
                            "hyb_build_compute": round(hyb_build_compute, 4),
                            "hyb_verify_compute": round(hyb_verify_compute, 4),
                            "hyb_total_compute": round(hyb_total_compute, 4),
                            "speedup_compute": round(speedup_compute, 3),
                            "winner_compute": winner_compute,
                        })
                        f.flush()

    print("\nBenchmark completato.\n")


if __name__ == "__main__":
    run_benchmark()
# main_iso_avg.py
# Benchmark Scientifico: Media su N ripetizioni.

import time
import math
import asyncio
import csv
import os
import gc # Garbage Collector per pulire la RAM tra i test
from pathlib import Path
from sequential_bloom_filter import SequentialBloomFilter
from hybrid_bloom_filter import HybridBloomFilter

# --- CONFIGURAZIONE ---
BASE_DIR = Path("bench_data")
DATASET_SIZES = [50_000, 100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
SPLIT_COUNTS = [1, 2, 4, 8, 16]

# Parametri Scientifici
NUM_ROUNDS = 10   # Quante volte ripetere ogni test?
SPLIT_FACTOR = 4
DESIRED_FPR = 0.01
CSV_OUT = "results_iso_average_2.csv" # Nome file diverso per non sovrascrivere

def _get_optimal_params(n: int, p: float):
    if n == 0: return 1000, 3
    ln2 = math.log(2.0)
    m = int(math.ceil(-n * math.log(p) / (ln2 ** 2)))
    k = max(1, int(round((m / n) * ln2)))
    return m, k

def _dynamic_chunk(total_items: int, factor: int) -> int:
    cpu = os.cpu_count() or 1
    chunk = math.ceil(total_items / (cpu * factor))
    return max(5000, chunk)

def run_benchmark():
    print(f"--- AVVIO BENCHMARK MEDIATO ({NUM_ROUNDS} ROUNDS) ---")
    print(f"Output: {CSV_OUT}")

    if not BASE_DIR.exists():
        print("ERRORE: Esegui 'generate_iso.py' prima!")
        return

    # Intestazione CSV
    fieldnames = [
        "total_items", "num_files",
        "seq_build_avg", "seq_verify_avg", "seq_total_avg",
        "hyb_build_avg", "hyb_verify_avg", "hyb_total_avg",
        "speedup_avg", "winner"
    ]

    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for size in DATASET_SIZES:
            size_dir = BASE_DIR / f"size_{size}"
            if not size_dir.exists(): continue

            print(f"\n> SCENARIO: {size} Elementi (Train + Test)")

            m, k = _get_optimal_params(size, DESIRED_FPR)
            chunk = _dynamic_chunk(size, SPLIT_FACTOR)

            for split in SPLIT_COUNTS:
                split_dir = size_dir / f"split_{split}"

                train_files = sorted((split_dir / "train").glob("*.txt"))
                test_files = sorted((split_dir / "test").glob("*.txt"))

                if not train_files: continue

                print(f"   -> {split} File | Chunk: {chunk} | Avvio {NUM_ROUNDS} Round...")

                # Accumulatori per le medie
                sum_seq_build = 0.0
                sum_seq_verify = 0.0
                sum_hyb_build = 0.0
                sum_hyb_verify = 0.0

                # --- CICLO DI RIPETIZIONE (ROUNDS) ---
                for r in range(1, NUM_ROUNDS + 1):
                    # Pulizia memoria forzata prima di ogni round per evitare interferenze
                    gc.collect()

                    # 1. SEQUENZIALE
                    bf_seq = SequentialBloomFilter(size=m, hash_count=k)

                    t0 = time.perf_counter()
                    bf_seq.build(train_files)
                    t_seq_b = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    bf_seq.verify_from_paths(test_files)
                    t_seq_v = time.perf_counter() - t0

                    sum_seq_build += t_seq_b
                    sum_seq_verify += t_seq_v

                    # 2. IBRIDO
                    # Ricreiamo l'oggetto per pulizia totale
                    bf_hyb = HybridBloomFilter(size=m, hash_count=k)

                    t0 = time.perf_counter()
                    asyncio.run(bf_hyb.build_async(train_files, chunk_size=chunk, max_workers=None))
                    t_hyb_b = time.perf_counter() - t0

                    t0 = time.perf_counter()
                    asyncio.run(bf_hyb.verify_async(test_files, chunk_size=chunk, max_workers=None))
                    t_hyb_v = time.perf_counter() - t0

                    sum_hyb_build += t_hyb_b
                    sum_hyb_verify += t_hyb_v

                    # Feedback visivo semplice (un puntino per round)
                    print(f".", end="", flush=True)

                print(" Fatto!") # Vai a capo dopo i puntini

                # --- CALCOLO MEDIE ---
                avg_seq_build = sum_seq_build / NUM_ROUNDS
                avg_seq_verify = sum_seq_verify / NUM_ROUNDS
                avg_seq_tot = avg_seq_build + avg_seq_verify

                avg_hyb_build = sum_hyb_build / NUM_ROUNDS
                avg_hyb_verify = sum_hyb_verify / NUM_ROUNDS
                avg_hyb_tot = avg_hyb_build + avg_hyb_verify

                speedup = avg_seq_tot / avg_hyb_tot
                win = "HYB" if avg_hyb_tot < avg_seq_tot else "SEQ"

                # Stampa Risultati Mediati
                print(f"      [MEDIA] SEQ: {avg_seq_tot:.3f}s | HYB: {avg_hyb_tot:.3f}s | Speedup: {speedup:.2f}x")

                writer.writerow({
                    "total_items": size,
                    "num_files": split,
                    "seq_build_avg": round(avg_seq_build, 4),
                    "seq_verify_avg": round(avg_seq_verify, 4),
                    "seq_total_avg": round(avg_seq_tot, 4),
                    "hyb_build_avg": round(avg_hyb_build, 4),
                    "hyb_verify_avg": round(avg_hyb_verify, 4),
                    "hyb_total_avg": round(avg_hyb_tot, 4),
                    "speedup_avg": round(speedup, 2),
                    "winner": win
                })
                f.flush()

    print("\nBenchmark completato.")

if __name__ == "__main__":
    run_benchmark()
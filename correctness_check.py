import asyncio
import math
import os
from pathlib import Path
from typing import Iterable, List

from sequential_bloom_filter import SequentialBloomFilter
from hybrid_bloom_filter import HybridBloomFilter


# =========================
# CONFIGURAZIONE SEMPLICE
# =========================
BASE_DIR = Path("bench_data")
DATASET_SIZE = 2_000_000
SPLIT_COUNT = 1
WORKERS = 8
CHUNK = "auto"          # "auto" oppure un intero, es. 50000
KNOWN_POSITIVES = 1000   # quanti elementi del train usare per il check no-false-negatives

DESIRED_FPR = 0.01


# =========================
# HELPERS
# =========================
def get_optimal_params(n: int, p: float) -> tuple[int, int]:
    if n == 0:
        return 1000, 3
    ln2 = math.log(2.0)
    m = int(math.ceil(-n * math.log(p) / (ln2 ** 2)))
    k = max(1, int(round((m / n) * ln2)))
    return m, k


def dynamic_chunk(total_items: int) -> int:
    cpu = os.cpu_count() or 1
    chunk = total_items // (cpu * 4)
    return max(5000, chunk)


def sample_first_n_lines(paths: Iterable[Path], n: int) -> List[str]:
    sample: List[str] = []
    for p in paths:
        with Path(p).open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                item = line.strip()
                if item:
                    sample.append(item)
                    if len(sample) >= n:
                        return sample
    return sample


# =========================
# CHECK DI CORRETTEZZA
# =========================
async def run_check() -> None:
    split_dir = BASE_DIR / f"size_{DATASET_SIZE}" / f"split_{SPLIT_COUNT}"
    train_files = sorted((split_dir / "train").glob("*.txt"))
    test_files = sorted((split_dir / "test").glob("*.txt"))

    if not train_files or not test_files:
        raise FileNotFoundError(
            f"Dataset non trovato in {split_dir}. Genera prima i dati in bench_data/."
        )

    m, k = get_optimal_params(DATASET_SIZE, DESIRED_FPR)
    chunk_size = dynamic_chunk(DATASET_SIZE) if CHUNK == "auto" else int(CHUNK)

    print("=== CORRECTNESS CHECK ===")
    print(f"size={DATASET_SIZE}, split={SPLIT_COUNT}, workers={WORKERS}, chunk={chunk_size}")
    print(f"Bloom params -> m={m}, k={k}")

    # 1) Build sequenziale
    bf_seq = SequentialBloomFilter(size=m, hash_count=k)
    inserted_seq = bf_seq.build(train_files)
    print(f"[SEQ] inserted = {inserted_seq}")

    # 2) Build ibrida
    bf_hyb = HybridBloomFilter(size=m, hash_count=k)
    inserted_hyb = await bf_hyb.build_async(
        train_files,
        chunk_size=chunk_size,
        max_workers=WORKERS,
    )
    print(f"[HYB] inserted = {inserted_hyb}")

    # Check A: stesso numero di elementi letti
    same_insert_count = inserted_seq == inserted_hyb
    print(f"[CHECK A] same inserted count: {same_insert_count}")

    # Check B: bit array finale identico
    same_bitarray = bf_seq.bit_array == bf_hyb.bit_array
    print(f"[CHECK B] same final bit array: {same_bitarray}")

    # 3) Verify sui file di test
    seq_results = bf_seq.verify_from_paths(test_files)
    hyb_results = await bf_hyb.verify_async(
        test_files,
        chunk_size=chunk_size,
        max_workers=WORKERS,
    )

    same_len = len(seq_results) == len(hyb_results)
    same_results = seq_results == hyb_results
    print(f"[CHECK C] same verify result length: {same_len}")
    print(f"[CHECK D] identical verify results: {same_results}")

    # 4) Check empirico no false negatives su elementi sicuramente inseriti
    known_positive_items = sample_first_n_lines(train_files, KNOWN_POSITIVES)
    if not known_positive_items:
        raise RuntimeError("Impossibile leggere campioni dal training set.")

    seq_known = bf_seq.contains(known_positive_items)
    hyb_known = bf_hyb.contains(known_positive_items)

    # i metodi restituiscono list[bool] se l'input è una lista
    seq_no_false_neg = all(seq_known)
    hyb_no_false_neg = all(hyb_known)
    print(f"[CHECK E] sequential has no false negatives on known positives: {seq_no_false_neg}")
    print(f"[CHECK F] hybrid has no false negatives on known positives: {hyb_no_false_neg}")

    all_ok = all([
        same_insert_count,
        same_bitarray,
        same_len,
        same_results,
        seq_no_false_neg,
        hyb_no_false_neg,
    ])

    print("\n=== FINAL RESULT ===")
    if all_ok:
        print("PASS: la versione ibrida è coerente con la sequenziale sulla configurazione scelta.")
    else:
        print("FAIL: trovata una discrepanza tra versione sequenziale e ibrida.")
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(run_check())

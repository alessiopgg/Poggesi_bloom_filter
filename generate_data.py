# generate_iso.py
# Genera dataset Train e Test IDENTICI per dimensione, ma frammentati diversamente.

import os
import random
import string
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Dimensioni totali del dataset (Righe)
DATASET_SIZES = [50_000, 100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]

# In quanti file spezzare il carico?
SPLIT_COUNTS = [1, 2, 4, 8, 16]

BASE_DIR = Path("bench_data")

def generate_chunk(num_lines):
    """Genera dati casuali in memoria."""
    chars = string.ascii_lowercase + string.digits
    data = []
    # Bufferizzazione veloce
    for _ in range(num_lines):
        length = random.randint(10, 20)
        data.append(''.join(random.choices(chars, k=length)))
    return data

def write_file(path, lines):
    path.write_text("\n".join(lines), encoding="utf-8")

def main():
    if BASE_DIR.exists():
        print("Pulizia vecchia cartella bench_data...")
        shutil.rmtree(BASE_DIR)
    BASE_DIR.mkdir()

    print("--- INIZIO GENERAZIONE ISO-WORKLOAD (TRAIN + TEST) ---")

    for size in DATASET_SIZES:
        print(f"\n> Generazione Master Dataset: {size} righe...")

        # 1. Creiamo DUE master dataset diversi in RAM
        print("  - Generazione Train Data in RAM...")
        master_train = generate_chunk(size)
        print("  - Generazione Test Data in RAM...")
        master_test = generate_chunk(size)

        size_dir = BASE_DIR / f"size_{size}"
        size_dir.mkdir()

        for split in SPLIT_COUNTS:
            # Creiamo la struttura cartelle
            split_dir = size_dir / f"split_{split}"
            train_dir = split_dir / "train"
            test_dir = split_dir / "test"

            train_dir.mkdir(parents=True)
            test_dir.mkdir(parents=True)

            lines_per_file = size // split
            remainder = size % split

            print(f"   -> Scrittura configurazione {split} file...")

            idx = 0
            futures = []
            with ProcessPoolExecutor() as executor:
                for i in range(split):
                    count = lines_per_file + (1 if i < remainder else 0)

                    # Slicing dei dati
                    sub_train = master_train[idx : idx + count]
                    sub_test = master_test[idx : idx + count]
                    idx += count

                    # Task scrittura Train
                    p_train = train_dir / f"train_{i:02d}.txt"
                    futures.append(executor.submit(write_file, p_train, sub_train))

                    # Task scrittura Test
                    p_test = test_dir / f"test_{i:02d}.txt"
                    futures.append(executor.submit(write_file, p_test, sub_test))

                # Attesa completamento scrittura fisica
                for f in futures: f.result()

    print(f"\n[OK] Dati pronti in {BASE_DIR}")

if __name__ == "__main__":
    main()
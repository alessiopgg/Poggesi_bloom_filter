# plot_results.py
# Generazione grafici scientifici dai risultati del benchmark
# Richiede: pandas, matplotlib, seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

CSV_FILE = "results_final.csv"
OUT_DIR = "plots"

sns.set(style="whitegrid")


def ensure_out():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)


# ============================================================
# 1. SPEEDUP VS WORKERS
# ============================================================

def plot_speedup_vs_workers(df):
    for size in df["size"].unique():
        subset = df[(df["size"] == size) & (df["chunk"] == "auto")]

        plt.figure(figsize=(9, 6))
        sns.lineplot(
            data=subset,
            x="workers",
            y="speedup",
            hue="split",
            marker="o",
            palette="tab10"
        )
        plt.title(f"Speedup vs Workers (size={size}, chunk=auto)")
        plt.xlabel("Numero di worker")
        plt.ylabel("Speedup")
        plt.savefig(f"{OUT_DIR}/speedup_workers_size{size}.png", dpi=200)
        plt.close()


# ============================================================
# 2. HEATMAP WORKER × CHUNK (speedup)
# ============================================================

def plot_heatmap(df):
    for size in df["size"].unique():
        for split in df["split"].unique():

            subset = df[(df["size"] == size) & (df["split"] == split)]

            pivot = subset.pivot_table(
                index="workers",
                columns="chunk",
                values="speedup",
                aggfunc="mean"
            )

            if pivot.empty:
                continue

            plt.figure(figsize=(10, 7))
            sns.heatmap(pivot, annot=True, cmap="viridis")
            plt.title(f"Heatmap Speedup (size={size}, split={split})")
            plt.savefig(f"{OUT_DIR}/heatmap_size{size}_split{split}.png", dpi=200)
            plt.close()


# ============================================================
# 3. TEMPI TOTALI SEQ vs HYB
# ============================================================

def plot_total_times(df):
    for size in df["size"].unique():
        subset = df[(df["size"] == size) & (df["chunk"] == "auto")]

        plt.figure(figsize=(9, 6))
        sns.lineplot(
            data=subset,
            x="workers",
            y="seq_total_mean",
            label="Sequenziale",
            marker="o"
        )
        sns.lineplot(
            data=subset,
            x="workers",
            y="hyb_total_mean",
            label="Ibrido",
            marker="o"
        )
        plt.title(f"Tempo totale seq vs hyb (size={size}, chunk=auto)")
        plt.xlabel("Numero di worker")
        plt.ylabel("Tempo totale (s)")
        plt.legend()
        plt.savefig(f"{OUT_DIR}/total_times_size{size}.png", dpi=200)
        plt.close()


# ============================================================
# 4. VARIABILITÀ: STANDARD DEVIATION
# ============================================================

def plot_std_dev(df):
    for size in df["size"].unique():

        subset = df[(df["size"] == size) & (df["chunk"] == "auto")]

        plt.figure(figsize=(9, 6))
        sns.lineplot(
            data=subset,
            x="workers",
            y="hyb_build_std",
            marker="o",
            label="Ibrido - Build std"
        )
        sns.lineplot(
            data=subset,
            x="workers",
            y="seq_build_std",
            marker="o",
            label="Sequenziale - Build std"
        )

        plt.title(f"Variabilità tempi di build (std dev) — size={size}")
        plt.xlabel("Workers")
        plt.ylabel("Std Dev (s)")
        plt.legend()
        plt.savefig(f"{OUT_DIR}/stddev_size{size}.png", dpi=200)
        plt.close()


# ============================================================
# 5. STRONG SCALING
# ============================================================

def plot_strong_scaling(df):
    for size in df["size"].unique():
        subset = df[(df["size"] == size) & (df["chunk"] == "auto")]

        plt.figure(figsize=(9, 6))
        sns.lineplot(
            data=subset,
            x="workers",
            y="hyb_total_mean",
            marker="o",
            label="Hybrid"
        )
        sns.lineplot(
            data=subset,
            x="workers",
            y="seq_total_mean",
            marker="o",
            label="Sequential"
        )
        plt.title(f"Strong Scaling (size={size}, chunk=auto)")
        plt.xlabel("Workers")
        plt.ylabel("Tempo totale (s)")
        plt.legend()
        plt.savefig(f"{OUT_DIR}/strong_scaling_size{size}.png", dpi=200)
        plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_out()

    print("Caricamento CSV...")
    df = pd.read_csv(CSV_FILE)

    print("Generazione grafici...")
    plot_speedup_vs_workers(df)
    plot_heatmap(df)
    plot_total_times(df)
    plot_std_dev(df)
    plot_strong_scaling(df)

    print("Grafici salvati in:", OUT_DIR)


if __name__ == "__main__":
    main()

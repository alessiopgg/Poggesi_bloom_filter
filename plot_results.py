# plot_results.py
# Generazione dei grafici principali per la relazione:
# 1) Speedup vs workers (end-to-end)
# 2) Speedup vs workers (compute-only)
# 3) Confronto speedup end-to-end vs compute-only
# 4) Execution time vs dataset size (end-to-end)
# 5) Effect of chunk size on speedup (end-to-end)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "results_final_with_compute.csv"
OUT_DIR = "plots_selected"

# Configurazioni rappresentative
MAIN_SPLIT = 1
MAIN_CHUNK = "auto"
BEST_WORKERS = 8

# Puoi cambiare queste due se vuoi usare 20M invece di 10M
COMPARISON_SIZE = 10_000_000
CHUNK_EFFECT_SIZE = 10_000_000

sns.set(style="whitegrid", context="talk")


def ensure_out():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)


def normalize_chunk_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["chunk_str"] = df["chunk"].astype(str)
    return df


def size_to_label(size: int) -> str:
    return f"{int(size/1_000_000)}M"


def chunk_sort_key(chunk_value: str) -> int:
    order = {
        "20000": 0,
        "50000": 1,
        "auto": 2
    }
    return order.get(str(chunk_value), 99)


def format_size_ticks(ax, sizes):
    ax.set_xticks(sizes)
    ax.set_xticklabels([size_to_label(s) for s in sizes])


# ============================================================
# 1. SPEEDUP VS WORKERS (END-TO-END)
# ============================================================

def plot_speedup_vs_workers(df: pd.DataFrame):
    subset = df[
        (df["split"] == MAIN_SPLIT) &
        (df["chunk_str"] == str(MAIN_CHUNK))
        ].copy()

    if subset.empty:
        return

    subset["size_label"] = subset["size"].apply(size_to_label)

    plt.figure(figsize=(9, 6))
    ax = sns.lineplot(
        data=subset,
        x="workers",
        y="speedup",
        hue="size_label",
        marker="o"
    )
    ax.set_title("Speedup vs Workers")
    ax.set_xlabel("Number of worker processes")
    ax.set_ylabel("Speedup")
    ax.legend(title="Dataset size")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/01_speedup_vs_workers_end_to_end.png", dpi=220, bbox_inches="tight")
    plt.close()


# ============================================================
# 2. SPEEDUP VS WORKERS (COMPUTE-ONLY)
# ============================================================

def plot_speedup_compute_vs_workers(df: pd.DataFrame):
    subset = df[
        (df["split"] == MAIN_SPLIT) &
        (df["chunk_str"] == str(MAIN_CHUNK))
        ].copy()

    if subset.empty:
        return

    subset["size_label"] = subset["size"].apply(size_to_label)

    plt.figure(figsize=(9, 6))
    ax = sns.lineplot(
        data=subset,
        x="workers",
        y="speedup_compute",
        hue="size_label",
        marker="o"
    )
    ax.set_title("Compute-only Speedup vs Workers")
    ax.set_xlabel("Number of worker processes")
    ax.set_ylabel("Compute-only speedup")
    ax.legend(title="Dataset size")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_speedup_vs_workers_compute_only.png", dpi=220, bbox_inches="tight")
    plt.close()


# ============================================================
# 3. CONFRONTO SPEEDUP END-TO-END vs COMPUTE-ONLY
# ============================================================

def plot_speedup_comparison(df: pd.DataFrame):
    subset = df[
        (df["size"] == COMPARISON_SIZE) &
        (df["split"] == MAIN_SPLIT) &
        (df["chunk_str"] == str(MAIN_CHUNK))
        ].copy()

    if subset.empty:
        return

    plt.figure(figsize=(9, 6))
    ax = sns.lineplot(
        data=subset,
        x="workers",
        y="speedup",
        marker="o",
        label="End-to-end speedup"
    )
    sns.lineplot(
        data=subset,
        x="workers",
        y="speedup_compute",
        marker="o",
        label="Compute-only speedup",
        ax=ax
    )
    ax.set_title(f"End-to-end vs Compute-only Speedup ({size_to_label(COMPARISON_SIZE)})")
    ax.set_xlabel("Number of worker processes")
    ax.set_ylabel("Speedup")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_speedup_comparison_size_{COMPARISON_SIZE}.png", dpi=220, bbox_inches="tight")
    plt.close()


# ============================================================
# 4. EXECUTION TIME VS DATASET SIZE (END-TO-END)
# ============================================================

def plot_execution_time_vs_size(df: pd.DataFrame):
    subset = df[
        (df["workers"] == BEST_WORKERS) &
        (df["split"] == MAIN_SPLIT) &
        (df["chunk_str"] == str(MAIN_CHUNK))
        ].copy()

    if subset.empty:
        return

    subset = subset.sort_values("size")
    sizes = subset["size"].tolist()

    plt.figure(figsize=(9, 6))
    ax = sns.lineplot(
        data=subset,
        x="size",
        y="seq_total",
        marker="o",
        label="Sequential"
    )
    sns.lineplot(
        data=subset,
        x="size",
        y="hyb_total",
        marker="o",
        label="Hybrid",
        ax=ax
    )
    ax.set_title("Execution Time vs Dataset Size")
    ax.set_xlabel("Dataset size")
    ax.set_ylabel("Total execution time (s)")
    format_size_ticks(ax, sizes)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/04_execution_time_vs_size_end_to_end.png", dpi=220, bbox_inches="tight")
    plt.close()


# ============================================================
# 5. EFFECT OF CHUNK SIZE (END-TO-END)
# ============================================================

def plot_chunk_effect(df: pd.DataFrame):
    subset = df[
        (df["size"] == CHUNK_EFFECT_SIZE) &
        (df["workers"] == BEST_WORKERS) &
        (df["split"] == MAIN_SPLIT)
        ].copy()

    if subset.empty:
        return

    subset["chunk_order"] = subset["chunk_str"].apply(chunk_sort_key)
    subset = subset.sort_values("chunk_order")

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=subset,
        x="chunk_str",
        y="speedup"
    )
    ax.set_title(f"Effect of Chunk Size on Speedup ({size_to_label(CHUNK_EFFECT_SIZE)})")
    ax.set_xlabel("Chunk size")
    ax.set_ylabel("Speedup")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/05_chunk_effect_end_to_end.png", dpi=220, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_out()

    print("Loading CSV...")
    df = pd.read_csv(CSV_FILE)
    df = normalize_chunk_column(df)

    print("Generating selected plots...")
    plot_speedup_vs_workers(df)
    plot_speedup_compute_vs_workers(df)
    plot_speedup_comparison(df)
    plot_execution_time_vs_size(df)
    plot_chunk_effect(df)

    print("Plots saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
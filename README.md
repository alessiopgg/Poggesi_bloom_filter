# Bloom Filter Acceleration in Python

This project implements and compares two versions of a Bloom Filter in Python:

- a **sequential baseline**
- a **hybrid parallel implementation** based on **AsyncIO** and **multiprocessing**

The goal of the project is to evaluate how much a hybrid strategy can improve Bloom Filter construction and membership-query performance on large synthetic datasets, while also analyzing the effect of dataset fragmentation, worker count, and chunk size.

The repository includes:

- the full source code
- a synthetic dataset generator
- a benchmark script
- a plotting script for the main experimental figures

The benchmark has been extended to report **both end-to-end timings and compute-only timings**, so the analysis can distinguish full application cost from the pure computational and parallelization cost.

---

## Project Structure

.
├── bloom_interface.py
├── sequential_bloom_filter.py
├── hybrid_bloom_filter.py
├── generate_data.py
├── main.py
├── plot_results.py
├── README.md
├── bench_data/                 # generated datasets
├── results_final_with_compute.csv
├── system_info.txt
└── plots_selected/

### Main files

- **bloom_interface.py**  
  Abstract interface for Bloom Filter implementations.

- **sequential_bloom_filter.py**  
  Sequential Bloom Filter implementation.  
  It also records internal **compute-only** timings for build and verify phases.

- **hybrid_bloom_filter.py**  
  Hybrid Bloom Filter implementation using:
  - `asyncio` for asynchronous file handling and coordination
  - `ProcessPoolExecutor` for CPU-bound parallel work

  It also records internal **compute-only** timings for build and verify phases.

- **generate_data.py**  
  Synthetic dataset generator. It creates independent train and test datasets and stores them under `bench_data/` using different fragmentation levels.

- **main.py**  
  Final benchmark script. It runs the sequential and hybrid implementations over multiple configurations and stores both:
  - **end-to-end** timings
  - **compute-only** timings

- **plot_results.py**  
  Generates the selected plots used in the report.

---

## Requirements

The project uses Python and the following external packages:

- `bitarray`
- `mmh3`
- `pandas`
- `matplotlib`
- `seaborn`

Standard-library modules such as `asyncio`, `csv`, `math`, `gc`, `os`, `time`, and `concurrent.futures` are also used.

You can install the external dependencies with:

```bash
pip install bitarray mmh3 pandas matplotlib seaborn
Dataset Generation

The datasets are fully synthetic.

For each configuration, the generator creates:

a training dataset, used to build the Bloom Filter
a test dataset, used for membership queries

Each sample is a random alphanumeric string with length uniformly distributed between 10 and 20 characters.

Train and test sets have the same size but contain independent data, with no intentional overlap.

The datasets are stored with the following structure:

bench_data/
 └── size_<N>/
     └── split_<S>/
         ├── train/
         │    ├── train_00.txt
         │    ├── train_01.txt
         │    └── ...
         └── test/
              ├── test_00.txt
              ├── test_01.txt
              └── ...
Generate datasets
python generate_data.py
Benchmark Configuration

The benchmark currently tests the following parameters:

Dataset sizes: 2_000_000, 5_000_000, 10_000_000, 20_000_000
Split counts: 1, 4, 16
Worker counts: 1, 4, 8, 16
Chunk sizes: auto, 20000, 50000

Other settings:

Warm-up runs: 1
Measured runs: 3
Target false positive rate: 1%

Bloom Filter parameters (m and k) are computed automatically from the dataset size and target false positive rate.

Measurement Methodology

The benchmark collects two complementary timing perspectives.

1. End-to-end timing

This includes the full workflow:

file access
parsing
Bloom Filter construction
membership queries
coordination overhead
2. Compute-only timing

This excludes file reading and parsing when possible and isolates the algorithmic and parallelization cost.

In the sequential version, compute-only timing measures the cost of hashing and bit-array operations during build and verify.
In the hybrid version, compute-only timing excludes asynchronous file reading and parsing, but still includes:
chunking
task submission
worker execution
synchronization
aggregation of partial results

This makes it possible to compare full application performance with the scalability of the computational core.

Run the Benchmark
python main.py

The benchmark generates:

results_final_with_compute.csv
system_info.txt

The CSV contains, for each tested configuration:

end-to-end times:
seq_build, seq_verify, seq_total
hyb_build, hyb_verify, hyb_total
speedup
compute-only times:
seq_build_compute, seq_verify_compute, seq_total_compute
hyb_build_compute, hyb_verify_compute, hyb_total_compute
speedup_compute
Generate Plots
python plot_results.py

The plotting script creates the selected figures in:

plots_selected/

Currently, it generates:

Speedup vs Workers (end-to-end)
Speedup comparison: end-to-end vs compute-only
Execution Time vs Dataset Size
Effect of Chunk Size on Speedup

The plotting script is configured with representative settings such as:

main split = 1
main chunk = auto
best workers = 8

and uses the final benchmark CSV as input.

Parallelization Strategy

The hybrid implementation separates the workflow into two parts:

I/O-bound part
Managed through asyncio, used for file reading and task coordination.
CPU-bound part
Managed through ProcessPoolExecutor, used to parallelize:
hash computation
local Bloom Filter construction
membership queries
Build phase

During construction:

input lines are divided into chunks
each worker creates a local Bloom Filter
local bit arrays are returned to the main process
the final global Bloom Filter is obtained through bitwise OR aggregation
Query phase

During membership queries:

the Bloom Filter is serialized once
each worker receives a chunk of queries
workers perform membership checks independently
results are collected and merged in the main process

This design avoids fine-grained synchronization on a shared bit array and follows a map-reduce style approach.

Main Findings

The updated benchmark shows that:

the hybrid implementation consistently outperforms the sequential baseline for most relevant configurations
the best configuration is typically:
8 worker processes
automatic chunk sizing
end-to-end speedup reaches a little above 4× on the largest workloads
compute-only speedup exceeds 5.5× on large datasets

This confirms that the computational core scales better than the full application workflow, and that part of the remaining overhead is due to file I/O, parsing, and coordination costs.

Notes

This project was developed as part of a university assignment on parallel programming and performance analysis.
The final version focuses on a clean comparison between:

sequential execution
hybrid AsyncIO + multiprocessing execution

with explicit analysis of:

strong scaling
chunk-size effects
end-to-end vs compute-only timing
scalability limits and bottlenecks

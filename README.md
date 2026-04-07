"Hybrid Bloom Filter in Python

This project was developed for a Parallel Programming course and presents a comparison between a sequential Bloom Filter implementation and a hybrid parallel version written in Python.

The hybrid implementation combines:
- asyncio for asynchronous file I/O
- multiprocessing through ProcessPoolExecutor for CPU-bound computation

The goal of the project is to evaluate performance under different configurations by varying:
- dataset size
- number of worker processes
- chunk size
- dataset fragmentation across multiple files

Project Structure

- bloom_interface.py
  Abstract interface for the Bloom Filter
- sequential_bloom_filter.py
  Sequential Bloom Filter implementation
- hybrid_bloom_filter.py
  Hybrid parallel Bloom Filter implementation
- generate_data.py
  Synthetic dataset generator
- main.py
  Benchmark runner


Requirements

Install the required Python packages:

pip install bitarray mmh3 pandas matplotlib seaborn

How to Run

1. Generate the datasets

python generate_data.py

2. Run the benchmark

python main.py


Tested Parameters

- Dataset sizes: 2M, 5M, 10M, 20M elements
- Worker processes: 1, 4, 8, 16
- Chunk sizes: auto, 20000, 50000
- Dataset splits: 1, 4, 16 files


Datasets

Datasets are synthetic and generated automatically by generate_data.py.

Each element is a random alphanumeric string with length between 10 and 20 characters.
For each configuration, the generator creates:
- a train dataset used to build the Bloom Filter
- a test dataset used for membership queries

Generated data is stored in:

bench_data/

Output

The benchmark produces a CSV file containing:
- build time
- verification time
- total execution time
- speedup
- best-performing configuration


Notes

This project focuses on the evaluation of a hybrid Python strategy for mixed I/O-bound and CPU-bound workloads, comparing it against a sequential baseline.

The repository is intended to provide:
- a clear separation between sequential and parallel implementations
- reproducible benchmark execution
- a compact experimental setup for performance analysis

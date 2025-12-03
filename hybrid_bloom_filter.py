# hybrid_bloom_filter.py

import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence, List, Union
import mmh3
from bitarray import bitarray

# Fallback silenzioso per l'interfaccia
try:
    from bloom_interface import BloomFilterInterface
except ImportError:
    class BloomFilterInterface:
        pass


# --- WORKERS (Pure Functions) ---

def _worker_build_partial(items: List[str], size: int, k: int) -> bytes:
    """Crea un bitarray locale per il chunk e lo ritorna come bytes."""
    local_ba = bitarray(size)
    local_ba.setall(0)
    for s in items:
        if not s: continue
        for seed in range(k):
            local_ba[mmh3.hash(s, seed) % size] = 1
    return local_ba.tobytes()


def _worker_verify(items: List[str], size: int, k: int, ba_bytes: bytes) -> List[bool]:
    """Verifica items contro un bitarray ricostruito dai bytes."""
    ba = bitarray()
    ba.frombytes(ba_bytes)
    if len(ba) > size: ba = ba[:size]  # Clip padding

    results = []
    for s in items:
        if not s:
            results.append(False)
            continue
        # Verifica veloce: True solo se tutti i bit sono 1
        results.append(all(ba[mmh3.hash(s, seed) % size] for seed in range(k)))
    return results


# --- CLASSE PRINCIPALE ---

class HybridBloomFilter(BloomFilterInterface):
    def __init__(self, size: int = 1000, hash_count: int = 3):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    # --- Sincroni ---

    def add(self, item: str) -> None:
        for seed in range(self.hash_count):
            self.bit_array[mmh3.hash(item, seed) % self.size] = 1

    def might_contain(self, item: str) -> bool:
        if not item: return False
        return all(self.bit_array[mmh3.hash(item, seed) % self.size] for seed in range(self.hash_count))

    def contains(self, items: Union[str, Sequence[str]]) -> Union[bool, List[bool]]:
        if isinstance(items, str):
            return self.might_contain(items)
        return [self.might_contain(i) for i in items]

    def clear(self) -> None:
        self.bit_array.setall(0)

    # --- Asincroni (I/O + Processi) ---

    async def build_async(self, paths: Iterable[Union[str, Path]], chunk_size: int = 100_000,
                          max_workers: int = None) -> int:

        loop = asyncio.get_running_loop()
        total = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for p in paths:

                path = Path(p)
                if not path.exists():
                    continue

                # ----------------------------------------------------
                # 1. Lettura file in un thread (non blocca event loop)
                # ----------------------------------------------------
                text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="ignore")

                # Rimuove righe vuote + strip
                lines = [s for s in (line.strip() for line in text.splitlines()) if s]
                if not lines:
                    continue

                total += len(lines)

                # ----------------------------------------------------
                # 2. Submit dei task CPU-bound ai processi
                # ----------------------------------------------------
                futures = [loop.run_in_executor(executor, _worker_build_partial, lines[i:i + chunk_size], self.size,
                                                self.hash_count)
                           for i in range(0, len(lines), chunk_size)
                           ]

                # ----------------------------------------------------
                # 3. Raccolta dei risultati con as_completed
                #    (i risultati arrivano uno alla volta)
                # ----------------------------------------------------
                for f in asyncio.as_completed(futures):
                    partial_bytes = await f

                    # Converti in bitarray
                    partial_ba = bitarray()
                    partial_ba.frombytes(partial_bytes)

                    # Tronca se eccede
                    if len(partial_ba) > self.size:
                        partial_ba = partial_ba[:self.size]

                    # OR bitwise con il Bloom Filter principale
                    self.bit_array |= partial_ba

        return total

    async def verify_async(self, paths: Iterable[Union[str, Path]], chunk_size: int = 50_000,
                           max_workers: int = None) -> List[bool]:

        loop = asyncio.get_running_loop()
        results = []

        # Serializzazione una volta sola (ottimo)
        ba_bytes = self.bit_array.tobytes()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for p in paths:

                path = Path(p)
                if not path.exists():
                    continue

                # Lettura file su thread
                text = await asyncio.to_thread(path.read_text, encoding="utf-8", errors="ignore")

                lines = [s for s in (line.strip() for line in text.splitlines()) if s]
                if not lines:
                    continue

                # Submit dei task CPU-bound
                futures = [
                    loop.run_in_executor(executor, _worker_verify, lines[i:i + chunk_size], self.size, self.hash_count,
                                         ba_bytes)
                    for i in range(0, len(lines), chunk_size)
                ]

                # Raccolta progressiva dei risultati
                for f in asyncio.as_completed(futures):
                    chunk_res = await f
                    results.extend(chunk_res)

        return results

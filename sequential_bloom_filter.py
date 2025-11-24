# sequential_bloom_filter.py (versione estesa)

from bloom_interface import BloomFilterInterface
from bitarray import bitarray
import mmh3
from pathlib import Path
from typing import Iterable, Sequence, List, Union


class SequentialBloomFilter(BloomFilterInterface):
    def __init__(self, size: int = 1000, hash_count: int = 3):
        """Costruttore del Bloom Filter."""
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def _hashes(self, item: str) -> list[int]:
        """Calcola 'hash_count' posizioni nel bit array per un dato elemento."""
        return [mmh3.hash(item, seed) % self.size for seed in range(self.hash_count)]

    def add(self, item: str) -> None:
        """Aggiunge un elemento al filtro, impostando a 1 i bit corrispondenti."""
        for index in self._hashes(item):
            self.bit_array[index] = 1

    def might_contain(self, item: str) -> bool:
        """
        Ritorna True se l'elemento potrebbe essere nel set.
        Questo è il metodo 'core' richiesto dall'interfaccia.
        """
        if not item:
            return False
        return all(self.bit_array[index] for index in self._hashes(item))

        # --- METODO UNIFICATO (Comodo) ---
    def contains(self, items: Union[str, Sequence[str]]) -> Union[bool, List[bool]]:
        """
        Wrapper intelligente:
        - Se passi una stringa -> chiama might_contain (ritorna bool)
        - Se passi una lista -> chiama might_contain per ogni elemento (ritorna List[bool])
        """
        if isinstance(items, str):
            return self.might_contain(items)

        # List comprehension che riusa la logica di base
        return [self.might_contain(i) for i in items]
    def clear(self) -> None:
        """Resetta il filtro (tutti i bit a 0)."""
        self.bit_array.setall(0)

    def build(self, members_path: str) -> int:
        """
        Legge il file 'members_path' e inizializza il filtro
        aggiungendo tutti gli elementi (uno per riga).
        Ritorna quanti elementi sono stati inseriti.
        """
        count = 0
        path = Path(members_path)
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                item = line.strip()
                if not item:
                    continue
                self.add(item)
                count += 1
        return count

    # NEW: build da più file (sequenziale)
    def build_from_paths(self, paths: Iterable[Path]) -> int:
        total = 0
        for p in paths:
            total += self.build(str(p))
        return total

    # NEW: verifica membership su 1+ file (sequenziale)
    def verify_from_paths(self, paths: Iterable[Path]) -> List[bool]:
        """Verifica elementi da più file usando il metodo contains."""
        results: List[bool] = []
        for p in paths:
            if not Path(p).exists():
                continue
            with Path(p).open("r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]
                # Qui contains ritornerà una List[bool]
                batch_result = self.contains(lines)
                if isinstance(batch_result, list):
                    results.extend(batch_result)
        return results

if __name__ == "__main__":
    # Esempio rapido
    MEMBERS_PATH = "data/processed/members.txt"
    size = 1920000
    hashes = 70
    bf = SequentialBloomFilter(size=size, hash_count=hashes)
    n = bf.build(MEMBERS_PATH)
    print(f"\nFiltro costruito con {n} elementi.")
    print(f"Dimensione array: {size}, numero hash: {hashes}")
    test_items = ["anna", "gioele32", "qualcosa_inventato"]
    print("\nRisultati dei test:")
    for item in test_items:
        result = bf.contains(item)
        print(f" - {item}: {'Forse presente' if result else 'Assente'}")

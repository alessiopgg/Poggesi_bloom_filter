# sequential_bloom_filter.py

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
        """Aggiunge un elemento al filtro."""
        for index in self._hashes(item):
            self.bit_array[index] = 1

    def might_contain(self, item: str) -> bool:
        """True se l'elemento è potenzialmente presente (metodo interfaccia)."""
        if not item: return False
        return all(self.bit_array[index] for index in self._hashes(item))

    def contains(self, items: Union[str, Sequence[str]]) -> Union[bool, List[bool]]:
        """
        Metodo smart: accetta sia stringa singola che lista di stringhe.
        """
        if isinstance(items, str):
            return self.might_contain(items)
        return [self.might_contain(i) for i in items]

    def clear(self) -> None:
        """Resetta il filtro."""
        self.bit_array.setall(0)

    # --- METODO BUILD UNIFICATO (RISOLVE IL TUO ERRORE) ---
    def build(self, source: Union[str, Path, Iterable[Union[str, Path]]]) -> int:
        """
        Carica elementi nel filtro.
        Accetta:
         - Un singolo percorso (str o Path) -> Carica quel file.
         - Una lista di percorsi (Iterable) -> Carica tutti i file nella lista.
        Ritorna: Il numero totale di elementi aggiunti.
        """

        # 1. Se è una LISTA (o iterabile), usiamo la ricorsione
        # Controllo che non sia str/Path, ma sia iterabile
        if not isinstance(source, (str, Path)) and isinstance(source, Iterable):
            total = 0
            for single_path in source:
                total += self.build(single_path)
            return total

        # 2. Se è un SINGOLO FILE (str o Path)
        path = Path(source)
        if not path.exists():
            print(f"Attenzione: File {path} non trovato.")
            return 0

        count = 0
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                item = line.strip()
                if item:
                    self.add(item)
                    count += 1
        return count

    # Manteniamo verify_from_paths separato per chiarezza nel main
    def verify_from_paths(self, paths: Iterable[Path]) -> List[bool]:
        results: List[bool] = []
        for p in paths:
            path_obj = Path(p)
            if not path_obj.exists():
                continue
            with path_obj.open("r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]
                # Sfruttiamo il contains smart
                batch_res = self.contains(lines)
                if isinstance(batch_res, list):
                    results.extend(batch_res)
        return results

if __name__ == "__main__":
    # Test rapido
    bf = SequentialBloomFilter(size=100, hash_count=3)
    # Test lista file (simulata)
    print("Test build lista vuota:", bf.build([]))
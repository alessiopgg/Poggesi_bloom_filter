from bloom_interface import BloomFilterInterface
import mmh3
from bitarray import bitarray


class SequentialBloomFilter(BloomFilterInterface):
    def __init__(self, size: int = 1000, hash_count: int = 3):
        """
        Costruttore del Bloom Filter.

        :param size: Dimensione dell'array di bit
        :param hash_count: Numero di funzioni hash simulate
        """
        self.size = size
        self.hash_count = hash_count
        self.bit_array = bitarray(size)
        self.bit_array.setall(0)

    def _hashes(self, item: str) -> list[int]:
        """
        Calcola 'hash_count' posizioni nel bit array per un dato elemento.

        :param item: Elemento da hashare
        :return: Lista di indici nell'array di bit
        """
        return [mmh3.hash(item, seed) % self.size for seed in range(self.hash_count)]

    def add(self, item: str) -> None:
        """
        Aggiunge un elemento al filtro, impostando a 1 i bit corrispondenti.
        """
        for index in self._hashes(item):
            self.bit_array[index] = 1

    def might_contain(self, item: str) -> bool:
        """
        Verifica se l'elemento potrebbe essere presente nel filtro.
        Restituisce True se tutti i bit corrispondenti sono a 1.
        """
        return all(self.bit_array[index] for index in self._hashes(item))

    def clear(self) -> None:
        """
        Resetta il filtro, impostando tutti i bit a 0.
        """
        self.bit_array.setall(0)

from abc import ABC, abstractmethod


class BloomFilterInterface(ABC):
    @abstractmethod
    def add(self, item: str) -> None:
        """Aggiunge un elemento al filtro."""
        pass

    @abstractmethod
    def might_contain(self, item: str) -> bool:
        """Restituisce True se l'elemento potrebbe essere presente, False se sicuramente non lo è."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Resetta il filtro (opzionale)."""
        pass

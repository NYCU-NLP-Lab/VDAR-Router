from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts import CanonicalRow


class SourceAdapter(ABC):
    adapter_name: str

    @abstractmethod
    def load_rows(self) -> list[CanonicalRow]:
        raise NotImplementedError

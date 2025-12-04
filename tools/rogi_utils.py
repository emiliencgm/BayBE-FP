from __future__ import annotations

from enum import auto
from typing import NamedTuple, Optional
from enum import Enum
from typing import Union
import numpy as np


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    def __str__(self) -> str:
        return self.value

    @classmethod
    def get(cls, name: Union[str, AutoName]) -> AutoName:
        if isinstance(name, cls):
            return name

        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unsupported alias! got: {name}. expected one of: {cls.keys()}")

    @classmethod
    def keys(cls) -> list[str]:
        return [e.value for e in cls]


class flist(list):
    def __format__(self, format_spec):
        fmt = lambda xs: ", ".join(f"{x:{format_spec}}" for x in xs)  # noqa: E731
        if len(self) >= 6:
            s = f"[{fmt(self[:3])}, ..., {fmt(self[-3:])}]"
        else:
            s = f"[{fmt(self)}]"

        return s

    def __str__(self) -> str:
        return f"{self}"

class Fingerprint(AutoName):
    MORGAN = auto()
    TOPOLOGICAL = auto()


class Metric(AutoName):
    DICE = auto()
    TANIMOTO = auto()
    EUCLIDEAN = auto()
    COSINE = auto()
    CITYBLOCK = auto()
    MAHALANOBIS = auto()
    PRECOMPUTED = auto()


class IntegrationDomain(AutoName):
    THRESHOLD = auto()
    CLUSTER_RATIO = auto()
    LOG_CLUSTER_RATIO = auto()
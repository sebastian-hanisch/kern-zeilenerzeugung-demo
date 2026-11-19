"""Zufallsstrom für alle Befunde der Demo: Blöcke aus SplitMix64 (kzl_rng, wörtliche Kopie), kein `random`, kein `numpy.random`.

Ein `Stream` gibt Blöcke gleichverteilter Zahlen auf [0, 1) aus. Jeder Aufruf von `block` zieht einen frischen Teilstrom (Teil-Seed aus Seed und Aufrufzähler,
beide über SplitMix64 durchmischt); so ist die Folge der Blöcke bei gleichem Seed auf jeder Plattform gleich (reine Ganzzahl-Arithmetik, kein libm)."""

import numpy as np

from kzl_rng import SplitMix64, uniform_block

_MASK = (1 << 64) - 1
_ZAEHLER_FAKTOR = 0xA24BAED4963EE407


class Stream:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self.aufrufe = 0
        self._basis = SplitMix64(self.seed).next()

    def block(self, count: int) -> np.ndarray:
        """`count` gleichverteilte Zahlen auf [0, 1) aus einem frischen Teilstrom."""
        self.aufrufe += 1
        teil_seed = SplitMix64((self._basis ^ (self.aufrufe * _ZAEHLER_FAKTOR)) & _MASK).next()
        return uniform_block(teil_seed, int(count))

    def matrix(self, rows: int, cols: int) -> np.ndarray:
        return self.block(rows * cols).reshape(rows, cols)

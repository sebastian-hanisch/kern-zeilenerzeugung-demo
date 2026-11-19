"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kzl_games as G  # noqa: E402


@pytest.fixture
def hand4():
    """Gewichte 4, 3, 2, 1, Quote 6 (Gewinner: {1,2}, {1,3}, {2,3,4} und Obermengen). Von Hand: ε* = 1/3 mit x = (1/3, 1/3, 1/3, 0):
    die drei Ungleichungen x1+x2, x1+x3, x2+x3+x4 >= 1 - ε summieren sich zu 3ε >= 1 + x4 >= 1."""
    return G.GewichtsSpiel([4, 3, 2, 1], 6)


@pytest.fixture
def drei_gleiche():
    """3 gleiche Spieler, Quote 2: v(S) = 1 für |S| >= 2. Von Hand: ε* = 1 - 2/3 = 1/3 bei x = (1/3, 1/3, 1/3)."""
    return G.GewichtsSpiel([1, 1, 1], 2)


@pytest.fixture
def zwei_kunden():
    """2 Kunden, 1 Standort mit Fixkosten 10, Wege 2 und 4: c({1}) = 12, c({2}) = 14, c(N) = 16.
    Von Hand: x1 - ε <= 12, x2 - ε <= 14, x1 + x2 = 16 -> 2ε >= 16 - 26, also ε* = -5 mit x = (7, 9): der Kern hat ein Inneres."""
    return G.StandortSpiel(np.array([[2.0], [4.0]]), np.array([10.0]))


@pytest.fixture
def dreieck3():
    """Dreiecks-Typen mit n = 3, f = 30: c = 30 (Einer), 30 (Paare), 60 (N); ε* = 10 (die drei Paar-Ungleichungen x_i + x_j <= 30 + ε summieren sich zu 2 * 60 <= 90 + 3ε)."""
    d = np.zeros((3, 3))
    d[np.arange(3), np.arange(3)] = 1000.0
    return G.StandortSpiel(d, np.full(3, 30.0))

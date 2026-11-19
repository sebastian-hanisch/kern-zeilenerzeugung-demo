"""SplitMix64-Kopie: wörtlich wie im Original, bekannte Werte, Block gleich Strom, keine anderen Importe; Teilströme des Stream sind reproduzierbar."""

import ast
from pathlib import Path

import numpy as np
import pytest

from kzl_rng import SplitMix64, uniform_block
from kzl_stream import Stream

ROOT = Path(__file__).resolve().parent.parent
RNG_SOURCE = ("ab-test-demo", "abt_rng.py")


def test_rng_copy_is_verbatim_apart_from_the_header():
    folder, original = RNG_SOURCE
    source = ROOT.parent / folder / original
    mine = (ROOT / "kzl_rng.py").read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_copied_generator_has_the_documented_stream():
    """SplitMix64 mit Seed 0: die bekannten ersten Werte der Konvention (Vigna)."""
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_block_equals_the_scalar_stream():
    for seed in (0, 1, 35, 2 ** 63 + 5):
        rng = SplitMix64(seed)
        assert np.array_equal(uniform_block(seed, 100), np.array([rng.uniform() for _ in range(100)]))


def test_rng_module_has_no_other_imports_than_numpy_and_math():
    tree = ast.parse((ROOT / "kzl_rng.py").read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert names == {"math", "numpy"}


def test_demo_code_uses_no_python_random_and_no_numpy_random():
    """Befunde kommen nur aus SplitMix64: im Laufzeitcode weder `import random` noch `numpy.random` noch `default_rng` (per AST, Kommentare zählen nicht)."""
    for path in ROOT.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert "random" not in {a.name.split(".")[0] for a in node.names}, path.name
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "random", path.name
            elif isinstance(node, ast.Attribute):
                assert node.attr not in ("random", "default_rng"), (path.name, node.lineno)
            elif isinstance(node, ast.Name):
                assert node.id != "default_rng", path.name


def test_stream_blocks_are_reproducible_uniform_and_distinct():
    a, b = Stream(11), Stream(11)
    x = a.block(200_000)
    assert np.array_equal(x, b.block(200_000)) and 0.0 <= x.min() and x.max() < 1.0 and abs(x.mean() - 0.5) < 4 * (1 / 12) ** 0.5 / 200_000 ** 0.5
    assert not np.array_equal(Stream(11).block(50), Stream(12).block(50))
    assert Stream(11).matrix(3, 4).shape == (3, 4)

"""Tests de regresión: todos los módulos deben importarse sin error.

Este test habría cazado el bug del `IndentationError` en `data_loader.py`
en el primer commit en el que se introdujo.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

CORE_DIR = Path(__file__).parent.parent / "app" / "core"


def _core_modules() -> list[str]:
    """Lista de módulos importables de app.core."""
    modules = []
    for py in sorted(CORE_DIR.glob("*.py")):
        if py.name.startswith("_"):
            continue
        modules.append(f"app.core.{py.stem}")
    return modules


def test_core_directory_exists():
    assert CORE_DIR.is_dir(), f"No existe {CORE_DIR}"


@pytest.mark.parametrize("module_name", _core_modules())
def test_core_module_importable(module_name: str):
    """Cada módulo de app.core debe importarse sin error."""
    try:
        importlib.import_module(module_name)
    except Exception as e:
        pytest.fail(f"Fallo al importar {module_name}: {e!r}")


def test_all_pages_syntax_valid():
    """Cada página de app/pages debe compilar sin error de sintaxis."""
    import py_compile

    pages_dir = Path(__file__).parent.parent / "app" / "pages"
    if not pages_dir.is_dir():
        pytest.skip("No hay carpeta app/pages")

    for py in sorted(pages_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        try:
            py_compile.compile(str(py), doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"Error de sintaxis en {py.name}: {e}")


def test_data_loader_first_char():
    """Regresión específica: la línea 1 de data_loader.py debe empezar en col 0."""
    path = CORE_DIR / "data_loader.py"
    if not path.exists():
        pytest.skip("data_loader.py no existe")

    with path.open("rb") as f:
        first_bytes = f.read(4)

    # El archivo debe empezar por `"""` (comilla doble comilla doble comilla doble)
    assert first_bytes[:1] == b'"', (
        f"data_loader.py empieza con {first_bytes[:1]!r}, no con '\"'. "
        "Probablemente hay un espacio o BOM al inicio."
    )
    assert first_bytes[:3] == b'"""', (
        f"data_loader.py no empieza con triple comilla: {first_bytes[:3]!r}"
    )

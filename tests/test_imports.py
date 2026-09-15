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


def test_no_duplicate_pages():
    """Regresión: cada número de página (1-8) debe tener exactamente UN
    archivo en app/pages/.

    Subir archivos con nombres que llevan emoji desde algunos gestores
    de archivos móviles puede corromper la codificación del nombre
    (mojibake) sin tocar el contenido, dejando un archivo NUEVO en vez
    de sobrescribir el existente. Streamlit renderiza todo archivo de
    app/pages/, así que el resultado es una página duplicada (una
    versión desactualizada conviviendo con la correcta) sin que ningún
    error de sintaxis o de import lo detecte — este test comprueba el
    recuento por número de prefijo, no solo que cada archivo compile.
    """
    pages_dir = Path(__file__).parent.parent / "app" / "pages"
    if not pages_dir.is_dir():
        pytest.skip("No hay carpeta app/pages")

    prefixes: dict[str, list[str]] = {}
    for py in sorted(pages_dir.glob("*.py")):
        if py.name.startswith("_"):
            continue
        prefix = py.name.split("_", 1)[0]
        prefixes.setdefault(prefix, []).append(py.name)

    duplicates = {p: names for p, names in prefixes.items() if len(names) > 1}
    assert not duplicates, (
        f"Hay páginas duplicadas por número de prefijo: {duplicates}. "
        "Probablemente un archivo con nombre corrupto (mojibake) "
        "convive con el archivo correcto — borra el que no tenga el "
        "emoji legible en el nombre."
    )
    assert len(prefixes) >= 8, (
        f"Se esperaban al menos 8 páginas, hay {len(prefixes)}: {sorted(prefixes)}. "
        "Si has borrado una página a propósito, baja este mínimo."
    )

"""QuickLabel launcher.

A tiny wrapper that locates the project's Python venv and starts
``python -m backend.server`` from the QuickLabel folder. Packaged by PyInstaller
into ``QuickLabel.exe`` so the user can double-click to run the app without
opening a terminal. The heavy dependencies (torch / sam2 / sam3 / opencv) stay
in the venv on disk — the .exe is small (~10 MB) and does not duplicate them.
"""
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path


def _here() -> Path:
    """Folder containing this launcher (PyInstaller-aware)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _find_python(root: Path) -> Path | None:
    """Resolve a Python interpreter: env override → local .venv → parent .venv."""
    override = os.environ.get("QUICKLABEL_PYTHON", "").strip()
    for candidate in [
        Path(override) if override else None,
        root / ".venv" / "Scripts" / "python.exe",
        root.parent / ".venv" / "Scripts" / "python.exe",
    ]:
        if candidate and candidate.is_file():
            return candidate
    return None


def main() -> int:
    root = _here()
    py = _find_python(root)

    if py is None:
        print("=" * 60)
        print(" QuickLabel: Python venv не найден")
        print("=" * 60)
        print(f" Искал в: {root / '.venv'}")
        print(f"          {root.parent / '.venv'}")
        print(" Запустите setup.ps1 из папки QuickLabel, чтобы создать venv,")
        print(" затем запустите QuickLabel.exe снова.")
        input("\nНажмите Enter для выхода…")
        return 1

    os.chdir(root)
    port = os.environ.get("QUICKLABEL_PORT", "8765")
    url = f"http://127.0.0.1:{port}"

    print(f"QuickLabel — запуск на {url}")
    print(f"Python: {py}")
    print("Закройте это окно, чтобы остановить сервер.")
    print()

    # NB: the browser is opened by backend.server.main() itself — the launcher
    # must NOT open it too, or two tabs appear.

    # Inherit stdout/stderr so server logs land in this console.
    try:
        proc = subprocess.Popen([str(py), "-m", "backend.server"], cwd=str(root))
        return proc.wait()
    except KeyboardInterrupt:
        print("\nОстановка…")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Ошибка запуска: {exc}")
        input("\nНажмите Enter для выхода…")
        return 2


if __name__ == "__main__":
    sys.exit(main())

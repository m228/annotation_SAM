"""QuickLabel frozen-bundle entry point.

Used by ``app.spec`` (PyInstaller) as the top-level entry when building the
full one-folder server bundle. Calls ``backend.server.main()`` directly so
PyInstaller can analyse the import graph and bundle all dependencies.

This file is NOT used when running from source (use run.ps1 / run.bat instead);
it is only the PyInstaller-visible entry so the bundler can walk all imports.
"""
from backend.server import main

if __name__ == "__main__":
    main()

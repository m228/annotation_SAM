"""QuickLabel self-update from GitHub Releases.

Three steps:
  1. check    — compare current version with latest GitHub release
  2. download — fetch the release zip into DATA_DIR/.update/staged
  3. apply    — launch detached PowerShell that waits for app exit,
               replaces files, and restarts the exe

apply works only in a frozen bundle: a running exe cannot overwrite itself,
so the actual replacement is done by an external PowerShell script after
the process exits. From source-checkout, only check is meaningful.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .config import DATA_DIR, GITHUB_REPO, read_version

_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
_ASSET_PREFIX = "QuickLabel_"

_STAGING = DATA_DIR / ".update"
_STAGED  = _STAGING / "staged"


def _is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _app_dir() -> Path:
    return Path(sys.executable).resolve().parent


def _norm(v: str) -> str:
    return (v or "").strip().lstrip("vV")


def _fetch_meta() -> tuple:
    """Return (tag, asset_dict|None, release_notes) from GitHub API."""
    req = urllib.request.Request(_API, headers={
        "User-Agent": f"QuickLabel-update/{read_version()}",
        "Accept": "application/vnd.github+json",
    })
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))
    asset = next(
        (a for a in data.get("assets", [])
         if str(a.get("name", "")).startswith(_ASSET_PREFIX) and a["name"].endswith(".zip")),
        None,
    )
    return data.get("tag_name") or "", asset, data.get("body") or ""


def _find_source_dir() -> Path | None:
    """Return the directory inside _STAGED that contains QuickLabel.exe."""
    direct = _STAGED / "QuickLabel.exe"
    if direct.is_file():
        return _STAGED
    sub = _STAGED / "QuickLabel" / "QuickLabel.exe"
    if sub.is_file():
        return _STAGED / "QuickLabel"
    return None


# ── Public API ──────────────────────────────────────────────────────────────

def check_latest() -> dict:
    """Compare current version with latest GitHub release. Safe from source."""
    current = read_version()
    try:
        tag, asset, notes = _fetch_meta()
    except Exception as exc:
        return {
            "current_version": current,
            "latest_version": None,
            "update_available": False,
            "error": "Не удалось связаться с GitHub: " + str(exc),
        }
    latest = _norm(tag)
    available = bool(latest) and asset is not None and _norm(current) != latest
    return {
        "current_version": current,
        "latest_version": latest or None,
        "update_available": available,
        "has_asset": asset is not None,
        "notes": notes,
    }


def download_latest() -> dict:
    """Download and unpack the latest release zip (frozen bundle only)."""
    if not _is_frozen():
        return {"ok": False, "error": "Обновление доступно только в собранной версии (.exe)"}

    try:
        tag, asset, _ = _fetch_meta()
    except Exception as exc:
        return {"ok": False, "error": "Не удалось связаться с GitHub: " + str(exc)}

    if asset is None:
        return {"ok": False, "error": f"В последнем релизе нет файла {_ASSET_PREFIX}*.zip"}

    if _STAGING.exists():
        shutil.rmtree(_STAGING, ignore_errors=True)
    _STAGING.mkdir(parents=True, exist_ok=True)
    zip_path = _STAGING / str(asset.get("name", "QuickLabel_new.zip"))

    try:
        req = urllib.request.Request(
            asset["browser_download_url"],
            headers={"User-Agent": "QuickLabel-update"},
        )
        with urllib.request.urlopen(req, timeout=300) as r, open(zip_path, "wb") as f:
            shutil.copyfileobj(r, f)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(_STAGED)
    except Exception as exc:
        return {"ok": False, "error": "Ошибка скачивания: " + str(exc)}

    src = _find_source_dir()
    if src is None:
        return {"ok": False, "error": "Архив скачан, но QuickLabel.exe не найден внутри"}

    staged_ver = ""
    for vf in (src / "VERSION", src / "_internal" / "VERSION"):
        if vf.is_file():
            try:
                staged_ver = vf.read_text(encoding="utf-8").strip()
                break
            except Exception:
                pass

    return {"ok": True, "version": staged_ver or _norm(tag)}


def apply_update() -> dict:
    """Launch a detached PowerShell updater that replaces files after this process exits."""
    if not _is_frozen():
        return {"ok": False, "error": "Обновление доступно только в собранной версии (.exe)"}

    src = _find_source_dir()
    if src is None:
        return {"ok": False, "error": "Сначала скачайте обновление"}

    app_dir = _app_dir()
    pid = os.getpid()
    exe = app_dir / "QuickLabel.exe"
    log_file = app_dir / "update.log"

    ps = f"""$ErrorActionPreference = 'Continue'
$app = '{app_dir}'
$staging = '{_STAGING}'
$src = '{src}'
$exe = '{exe}'
$log = '{log_file}'
function W($m) {{ "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File -FilePath $log -Append -Encoding utf8 }}
W "=== apply: waiting for pid {pid} ==="
try {{ Wait-Process -Id {pid} -Timeout 60 }} catch {{}}
Start-Sleep -Seconds 1
for ($i = 0; $i -lt 30; $i++) {{
    try {{
        if (Test-Path -LiteralPath (Join-Path $app '_internal')) {{
            Remove-Item -LiteralPath (Join-Path $app '_internal') -Recurse -Force -ErrorAction Stop
        }}
        break
    }} catch {{ Start-Sleep -Milliseconds 500 }}
}}
Remove-Item -LiteralPath $exe -Force -ErrorAction SilentlyContinue
try {{
    Copy-Item -Path (Join-Path $src '*') -Destination $app -Recurse -Force -ErrorAction Stop
    W "files replaced"
}} catch {{ W "COPY ERROR: $_" }}
Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue
if (Test-Path -LiteralPath $exe) {{
    W "restarting $exe"
    Start-Process -FilePath $exe -WorkingDirectory $app
}} else {{
    W "ERROR: QuickLabel.exe not found after update"
}}
W "=== apply: done ==="
"""
    helper = Path(tempfile.gettempdir()) / f"quicklabel_apply_{pid}.ps1"
    # UTF-8 with BOM — PowerShell 5.1 requires BOM to handle non-ASCII paths
    helper.write_bytes(b"\xef\xbb\xbf" + ps.encode("utf-8"))

    CREATE_NEW_CONSOLE        = 0x00000010
    CREATE_NEW_PROCESS_GROUP  = 0x00000200
    CREATE_BREAKAWAY_FROM_JOB = 0x01000000

    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
           "-WindowStyle", "Hidden", "-File", str(helper)]
    base = CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP
    try:
        subprocess.Popen(cmd, creationflags=base | CREATE_BREAKAWAY_FROM_JOB, close_fds=True)
    except OSError:
        subprocess.Popen(cmd, creationflags=base, close_fds=True)

    return {"ok": True}

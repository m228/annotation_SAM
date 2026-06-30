"""QuickLabel self-update from GitHub Releases.

Three steps:
  1. check    — compare current version with latest GitHub release
  2. download — fetch the release zip into DATA_DIR/.update/staged
  3. apply    — launch detached PowerShell that waits for app exit,
               replaces files, and restarts the exe

apply works in two deployment shapes:
  * frozen one-folder bundle — the running exe cannot overwrite itself;
  * tiny launcher (default build) — a QuickLabel.exe next to DATA_DIR starts
    ``python -m backend.server`` from source in the .venv, so the *server* is
    not frozen but the files on disk are still ours to replace.
In both, the actual file swap is done by an external PowerShell script after
this process exits, then it restarts QuickLabel.exe. A pure source checkout
with no QuickLabel.exe alongside (dev mode) supports only ``check``.
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
    """Folder that holds QuickLabel.exe + the files we replace.

    This is always DATA_DIR (next to the exe), NOT ``sys.executable``'s parent:
    in launcher mode ``sys.executable`` is the .venv python, whose folder is the
    wrong place to update or restart from.
    """
    return DATA_DIR


def _deployed_exe() -> Path | None:
    """Return QuickLabel.exe next to DATA_DIR if it exists (launcher deploy)."""
    exe = DATA_DIR / "QuickLabel.exe"
    return exe if exe.is_file() else None


def _can_self_update() -> bool:
    """True when we know how to replace files and restart: frozen bundle, or a
    launcher deploy with QuickLabel.exe alongside. False for a dev checkout."""
    return _is_frozen() or _deployed_exe() is not None


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
    """Download and unpack the latest release zip (deployed app only)."""
    if not _can_self_update():
        return {"ok": False, "error": "Обновление доступно только в установленной версии (рядом нет QuickLabel.exe)"}

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
    if not _can_self_update():
        return {"ok": False, "error": "Обновление доступно только в установленной версии (рядом нет QuickLabel.exe)"}

    src = _find_source_dir()
    if src is None:
        return {"ok": False, "error": "Сначала скачайте обновление"}

    app_dir = _app_dir()
    pid = os.getpid()
    exe = app_dir / "QuickLabel.exe"
    log_file = app_dir / "update.log"

    # The staged zip carries no user data (.venv / models / projects / wheels are
    # gitignored and absent from the release), so copying $src/* over $app cannot
    # clobber them. We retry freeing the locked targets: in launcher mode the exe
    # stays locked for a moment after this server exits, until the parent
    # QuickLabel.exe (which spawned us) also exits.
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
for ($i = 0; $i -lt 60; $i++) {{
    $ok = $true
    try {{
        if (Test-Path -LiteralPath (Join-Path $app '_internal')) {{
            Remove-Item -LiteralPath (Join-Path $app '_internal') -Recurse -Force -ErrorAction Stop
        }}
    }} catch {{ $ok = $false }}
    try {{
        if (Test-Path -LiteralPath $exe) {{
            Remove-Item -LiteralPath $exe -Force -ErrorAction Stop
        }}
    }} catch {{ $ok = $false }}
    if ($ok) {{ break }}
    Start-Sleep -Milliseconds 500
}}
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

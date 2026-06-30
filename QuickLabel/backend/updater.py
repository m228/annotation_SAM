"""QuickLabel update check.

Compares the installed VERSION with the latest GitHub release tag so the UI
can show a "new version available" dot and a link to the releases page.

In-app install was removed: it required writing to the running app's folder,
which is fragile on Windows (exe file locks, antivirus, requireAdministrator
mismatches). The UI now sends the user to the GitHub releases page; users
update via update.bat or by downloading the zip manually.
"""

import json
import urllib.request

from .config import GITHUB_REPO, read_version

_API = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
_ASSET_PREFIX = "QuickLabel_"


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


def check_latest() -> dict:
    """Compare current version with the latest GitHub release. Safe from source."""
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

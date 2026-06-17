# Build QuickLabel.exe (lightweight launcher).
#
# The .exe is just a small wrapper around launcher.py — it locates the project's
# Python venv and runs `python -m backend.server`. The venv (~5 GB with torch /
# sam2 / sam3) and the model checkpoints stay on disk; they are NOT embedded.
# This keeps the .exe under ~15 MB and rebuilds fast.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# Same Python lookup as run.ps1 so dev and build share one source of truth.
$VenvCandidates = @(
    $env:QUICKLABEL_PYTHON,
    (Join-Path $Root ".venv\Scripts\python.exe"),
    (Join-Path (Split-Path -Parent $Root) ".venv\Scripts\python.exe")
)
$Venv = $VenvCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $Venv) {
    Write-Error "Python venv не найден. Сначала запустите setup.ps1."
    exit 1
}

Write-Host "Building QuickLabel.exe with $Venv" -ForegroundColor Cyan

# Make sure PyInstaller is available in the venv (idempotent).
& $Venv -m pip install --quiet pyinstaller

# Build to a temporary work area, then move the single binary alongside the app.
$Work = Join-Path $Root "build"
$Dist = Join-Path $Root "dist"
& $Venv -m PyInstaller `
    --onefile `
    --console `
    --name QuickLabel `
    --workpath $Work `
    --distpath $Dist `
    --specpath $Work `
    --noconfirm `
    (Join-Path $Root "launcher.py")

$Built = Join-Path $Dist "QuickLabel.exe"
if (-not (Test-Path $Built)) {
    Write-Error "Сборка не удалась: $Built не найден."
    exit 2
}

Copy-Item $Built (Join-Path $Root "QuickLabel.exe") -Force
Write-Host "`nГотово: $(Join-Path $Root 'QuickLabel.exe')" -ForegroundColor Green
Write-Host "Двойной клик запускает сервер и открывает браузер на http://127.0.0.1:8765"

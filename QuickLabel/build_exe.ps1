# Build QuickLabel.exe and create a distribution zip for GitHub Releases.
#
# The .exe is a tiny wrapper (~7 MB) that finds the local .venv and starts
# python -m backend.server. The heavy deps (torch / sam2 / sam3 / opencv) stay
# on disk in the user's .venv and are NOT bundled into the exe.
#
# Output:
#   QuickLabel/QuickLabel.exe          - the launcher
#   QuickLabel/dist/QuickLabel_vX.Y.Z.zip  - release archive (no models/wheels)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# Read version
$Version = (Get-Content (Join-Path $Root "VERSION") -Raw).Trim()
Write-Host "Building QuickLabel v$Version" -ForegroundColor Cyan

# Locate venv python
$VenvCandidates = @(
    $env:QUICKLABEL_PYTHON,
    (Join-Path $Root ".venv\Scripts\python.exe"),
    (Join-Path (Split-Path -Parent $Root) ".venv\Scripts\python.exe")
)
$Venv = $VenvCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $Venv) {
    Write-Error "Python venv not found. Run .\setup.ps1 first."
    exit 1
}

Write-Host "Python: $Venv" -ForegroundColor Gray

# Install / upgrade PyInstaller in venv
& $Venv -m pip install --quiet --upgrade pyinstaller

# Build the tiny launcher exe
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
    Write-Error "Build failed: $Built not found."
    exit 2
}

# Copy exe to project root (convenience - double-click to run)
Copy-Item $Built (Join-Path $Root "QuickLabel.exe") -Force

# Create distribution zip (code + scripts, no models/wheels/.venv)
$ZipName = "QuickLabel_v$Version.zip"
$ZipPath = Join-Path $Dist $ZipName

$Include = @(
    "QuickLabel.exe",
    "launcher.py",
    "requirements.txt",
    "setup.ps1",
    "run.ps1",
    "run.bat",
    "update.ps1",
    "update.bat",
    "VERSION",
    "README.md",
    "backend",
    "ml_backend",
    "web",
    "projects"
)

Write-Host "`nPacking $ZipName..." -ForegroundColor Cyan

# Use a temp staging folder so we control what goes into the zip
$Stage = Join-Path $Dist "stage_$Version"
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory $Stage | Out-Null

$AppDir = Join-Path $Stage "QuickLabel"
New-Item -ItemType Directory $AppDir | Out-Null

foreach ($Item in $Include) {
    $Src = Join-Path $Root $Item
    if (-not (Test-Path $Src)) { continue }
    $Dst = Join-Path $AppDir $Item
    if ((Get-Item $Src).PSIsContainer) {
        # Copy directory, skip __pycache__ and _train
        Copy-Item $Src $Dst -Recurse -Force
        Get-ChildItem $Dst -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force
        Get-ChildItem $Dst -Recurse -Filter "_train" -Directory | Remove-Item -Recurse -Force
    } else {
        Copy-Item $Src $Dst -Force
    }
}

if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $ZipPath

Remove-Item $Stage -Recurse -Force

Write-Host "`nDone!" -ForegroundColor Green
Write-Host "  Exe:  $(Join-Path $Root 'QuickLabel.exe')"
Write-Host "  Zip:  $ZipPath"
Write-Host "`nTo publish a release:"
Write-Host "  gh release create v$Version '$ZipPath' -t v$Version --generate-notes"

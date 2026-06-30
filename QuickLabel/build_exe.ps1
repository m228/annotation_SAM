# Build QuickLabel.exe and create a distribution zip for GitHub Releases.
#
# The .exe is a tiny wrapper (~7 MB) that finds the local .venv and starts
# python -m backend.server. The heavy deps (torch / sam2 / sam3 / opencv) stay
# on disk in the user's .venv and are NOT bundled into the exe.
#
# Output:
#   QuickLabel/QuickLabel.exe              - the launcher (copy here for double-click)
#   QuickLabel/dist/QuickLabel_vX.Y.Z.zip  - release archive

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $Root
Set-Location $Root

# Read version
$Version = (Get-Content (Join-Path $Root "VERSION") -Raw).Trim()
Write-Host "Building QuickLabel v$Version" -ForegroundColor Cyan

# ── 1. Locate venv python ────────────────────────────────────────────────────
$VenvCandidates = @(
    $env:QUICKLABEL_PYTHON,
    (Join-Path $Root ".venv\Scripts\python.exe"),
    (Join-Path $RepoRoot ".venv\Scripts\python.exe")
)
$Venv = $VenvCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $Venv) {
    Write-Error "Python venv not found. Run .\setup.ps1 first."
    exit 1
}
Write-Host "Python: $Venv" -ForegroundColor Gray

# ── 2. Build the tiny launcher exe ─────────────────────────────────────────
& $Venv -m pip install --quiet --upgrade pyinstaller

$Work = Join-Path $Root "build"
$Dist = Join-Path $Root "dist"
New-Item -ItemType Directory -Force $Dist | Out-Null

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

# Copy exe to project root (double-click to run)
Copy-Item $Built (Join-Path $Root "QuickLabel.exe") -Force

# ── 3. Package release zip via git archive (respects .gitignore) ─────────
$ZipName = "QuickLabel_v$Version.zip"
$ZipPath = Join-Path $Dist $ZipName

$Stage = Join-Path $Dist "stage_$Version"
if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
New-Item -ItemType Directory (Join-Path $Stage "QuickLabel") | Out-Null

Write-Host "`nPacking $ZipName via git archive..." -ForegroundColor Cyan

# Export only git-tracked QuickLabel files (honours .gitignore)
# --output must be inside the repo (git security restriction)
$TarPath = Join-Path $Dist "ql_archive.tar"
git -C $RepoRoot archive --output $TarPath HEAD -- QuickLabel/
if (-not (Test-Path $TarPath)) {
    Write-Error "git archive failed."
    exit 3
}

# Extract the tar (PowerShell 5 has no native tar; use tar.exe bundled with Windows 10+)
tar -xf $TarPath -C $Stage
Remove-Item $TarPath -Force

# Add the freshly built exe (it is gitignored, so not in the archive)
Copy-Item (Join-Path $Root "QuickLabel.exe") (Join-Path $Stage "QuickLabel\QuickLabel.exe") -Force

# Zip the staged folder
if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $ZipPath

Remove-Item $Stage -Recurse -Force

Write-Host "`nDone!" -ForegroundColor Green
Write-Host "  Exe:  $(Join-Path $Root 'QuickLabel.exe')"
Write-Host "  Zip:  $ZipPath"
$SizeMB = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host "  Size: $SizeMB MB"
Write-Host "`nTo publish a release:"
Write-Host "  gh release create v$Version '$ZipPath' -t v$Version --generate-notes"

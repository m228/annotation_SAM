# Build QuickLabel for distribution.
#
# Two modes:
#   .\build_exe.ps1          # (default) tiny launcher exe + release zip
#   .\build_exe.ps1 -Full    # full one-folder server bundle + release zip
#
# Tiny launcher (default):
#   QuickLabel.exe — 7 MB wrapper that finds the local .venv and starts
#   python -m backend.server. Users must run setup.ps1 first.
#   Release zip: ~7 MB, ideal for CI / GitHub Actions.
#
# Full server bundle (-Full):
#   dist/QuickLabel/ — one-folder frozen bundle (Python + fastapi + uvicorn +
#   numpy + pillow + opencv + scipy bundled; torch / SAM stay in .venv).
#   Users still need setup.ps1 for ML deps, but NOT for the server to start.
#   Release zip: ~300-500 MB.

param([switch]$Full)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $Root
Set-Location $Root

$Version = (Get-Content (Join-Path $Root "VERSION") -Raw).Trim()
$Mode = if ($Full) { "full" } else { "launcher" }
Write-Host "Building QuickLabel v$Version [$Mode]" -ForegroundColor Cyan

# ── Locate venv python ────────────────────────────────────────────────────────
$VenvCandidates = @(
    $env:QUICKLABEL_PYTHON,
    (Join-Path $Root ".venv\Scripts\python.exe"),
    (Join-Path $RepoRoot ".venv\Scripts\python.exe")
)
$Venv = $VenvCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $Venv) { Write-Error "Python venv not found. Run .\setup.ps1 first."; exit 1 }
Write-Host "Python: $Venv" -ForegroundColor Gray

& $Venv -m pip install --quiet --upgrade pyinstaller

$Work = Join-Path $Root "build"
$Dist = Join-Path $Root "dist"
New-Item -ItemType Directory -Force $Dist | Out-Null

if ($Full) {
    # ── Full one-folder bundle via app.spec ───────────────────────────────────
    Write-Host "`nBuilding full server bundle (this takes 2-5 min)..." -ForegroundColor Cyan
    & $Venv -m PyInstaller `
        --workpath $Work `
        --distpath $Dist `
        --noconfirm `
        (Join-Path $Root "app.spec")

    $BundleDir = Join-Path $Dist "QuickLabel"
    if (-not (Test-Path (Join-Path $BundleDir "QuickLabel.exe"))) {
        Write-Error "Build failed: $BundleDir\QuickLabel.exe not found."
        exit 2
    }

    # Package the full bundle folder into a zip for distribution
    $ZipName = "QuickLabel_v${Version}_full.zip"
    $ZipPath = Join-Path $Dist $ZipName

    Write-Host "`nPacking $ZipName..." -ForegroundColor Cyan
    if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }

    # Stage: wrap bundle in a QuickLabel\ prefix folder
    $Stage = Join-Path $Dist "stage_full"
    if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
    New-Item -ItemType Directory $Stage | Out-Null
    Copy-Item $BundleDir (Join-Path $Stage "QuickLabel") -Recurse

    # Add user-facing scripts next to the exe (not inside _internal)
    foreach ($f in @("setup.ps1", "update.ps1", "update.bat", "VERSION", "INSTALL.md", "README.md")) {
        $src = Join-Path $Root $f
        if (Test-Path $src) { Copy-Item $src (Join-Path $Stage "QuickLabel\$f") -Force }
    }
    "run.bat", "run.ps1" | ForEach-Object {
        $src = Join-Path $Root $_
        if (Test-Path $src) { Copy-Item $src (Join-Path $Stage "QuickLabel\$_") -Force }
    }

    # Add empty placeholder dirs for user data
    New-Item -ItemType Directory -Force (Join-Path $Stage "QuickLabel\models") | Out-Null
    New-Item -ItemType Directory -Force (Join-Path $Stage "QuickLabel\projects") | Out-Null

    Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $ZipPath
    Remove-Item $Stage -Recurse -Force

    $ExePath = Join-Path $BundleDir "QuickLabel.exe"
} else {
    # ── Tiny launcher (default) ───────────────────────────────────────────────
    & $Venv -m PyInstaller `
        --onefile --console --name QuickLabel `
        --workpath $Work --distpath $Dist --specpath $Work `
        --noconfirm `
        (Join-Path $Root "launcher.py")

    $Built = Join-Path $Dist "QuickLabel.exe"
    if (-not (Test-Path $Built)) { Write-Error "Build failed: $Built not found."; exit 2 }

    # Copy to project root for double-click convenience
    Copy-Item $Built (Join-Path $Root "QuickLabel.exe") -Force

    # Package release zip via git archive (respects .gitignore)
    $ZipName = "QuickLabel_v$Version.zip"
    $ZipPath = Join-Path $Dist $ZipName

    Write-Host "`nPacking $ZipName via git archive..." -ForegroundColor Cyan

    $TarPath = Join-Path $Dist "ql_archive.tar"
    git -C $RepoRoot archive --output $TarPath HEAD -- QuickLabel/
    if (-not (Test-Path $TarPath)) { Write-Error "git archive failed."; exit 3 }

    $Stage = Join-Path $Dist "stage_$Version"
    if (Test-Path $Stage) { Remove-Item $Stage -Recurse -Force }
    New-Item -ItemType Directory (Join-Path $Stage "QuickLabel") | Out-Null
    tar -xf $TarPath -C $Stage
    Remove-Item $TarPath -Force

    # Inject freshly-built exe (gitignored, so not in archive)
    Copy-Item (Join-Path $Root "QuickLabel.exe") (Join-Path $Stage "QuickLabel\QuickLabel.exe") -Force

    if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
    Compress-Archive -Path (Join-Path $Stage "*") -DestinationPath $ZipPath
    Remove-Item $Stage -Recurse -Force

    $ExePath = Join-Path $Root "QuickLabel.exe"
}

Write-Host "`nDone!" -ForegroundColor Green
Write-Host "  Exe:  $ExePath"
Write-Host "  Zip:  $ZipPath"
Write-Host "  Size: $([math]::Round((Get-Item $ZipPath).Length / 1MB, 1)) MB"
Write-Host "`nTo publish a release:"
if ($Full) {
    Write-Host "  gh release create v$Version '$ZipPath' -t v$Version --title 'v$Version (full bundle)' --generate-notes"
} else {
    Write-Host "  gh release create v$Version '$ZipPath' -t v$Version --generate-notes"
}

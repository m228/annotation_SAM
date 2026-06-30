# QuickLabel updater (ASCII only - safe for PowerShell 5.1 on Russian Windows)
# Downloads a release from GitHub and extracts it alongside this script,
# replacing QuickLabel.exe and source code but keeping .venv, models, and projects.
#
# Usage:
#   .\update.ps1                 interactive: ask which version (Enter = latest)
#   .\update.ps1 -Version 1.0.3  download exactly v1.0.3 (with or without leading "v")
#   .\update.ps1 -Force          install latest even if it equals the installed version

param(
    [string]$Version,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

$Repo  = "m228/annotation_SAM"
$ApiBase = "https://api.github.com/repos/$Repo"

function Get-LocalVersion {
    $f = Join-Path $Root "VERSION"
    if (Test-Path $f) { return (Get-Content $f -Raw).Trim() }
    return "0.0.0"
}

Write-Host "QuickLabel Updater" -ForegroundColor Cyan

# ── Pick a target version ─────────────────────────────────────────────────────
# No -Version given and not -Force => ask. Empty answer = latest release.
if (-not $Version -and -not $Force) {
    Write-Host ""
    Write-Host "Enter version to install (e.g. 1.0.3), or press Enter for the latest:" -ForegroundColor Yellow
    $Version = (Read-Host "Version").Trim()
}
$Version = $Version.TrimStart("v", "V")   # accept both "v1.0.3" and "1.0.3"

if ($Version) {
    $Tag = "v$Version"
    Write-Host "Fetching release $Tag from github.com/$Repo ..."
    $Url = "$ApiBase/releases/tags/$Tag"
} else {
    Write-Host "Fetching the latest release from github.com/$Repo ..."
    $Url = "$ApiBase/releases/latest"
}

try {
    $Release = Invoke-RestMethod $Url -ErrorAction Stop
} catch {
    if ($Version) {
        Write-Host "ERROR: Release $Tag not found. Check the version number and try again." -ForegroundColor Red
    } else {
        Write-Host "ERROR: Cannot reach GitHub. Check your internet connection." -ForegroundColor Red
    }
    Write-Host $_.Exception.Message
    pause
    exit 1
}

$LatestTag = $Release.tag_name          # e.g. "v1.2.0"
$LatestVer = $LatestTag.TrimStart("v")
$LocalVer  = Get-LocalVersion

Write-Host "  Installed : v$LocalVer"
Write-Host "  Selected  : $LatestTag"

if (-not $Force -and $LatestVer -eq $LocalVer) {
    Write-Host "`nAlready on v$LocalVer." -ForegroundColor Green
    $ans = (Read-Host "Reinstall anyway? (y/N)").Trim()
    if ($ans -ne "y" -and $ans -ne "Y") { exit 0 }
}

# Find the zip asset (name starts with "QuickLabel_")
$Asset = $Release.assets | Where-Object { $_.name -like "QuickLabel_*.zip" } | Select-Object -First 1
if (-not $Asset) {
    Write-Host "ERROR: No QuickLabel_*.zip asset found in release $LatestTag." -ForegroundColor Red
    pause
    exit 1
}

$ZipUrl  = $Asset.browser_download_url
$ZipFile = Join-Path $env:TEMP "QuickLabel_update_$LatestVer.zip"
$ExtDir  = Join-Path $env:TEMP "QuickLabel_update_$LatestVer"

Write-Host "`nDownloading $($Asset.name) ($([math]::Round($Asset.size/1MB,1)) MB)..."
Invoke-WebRequest $ZipUrl -OutFile $ZipFile -UseBasicParsing

if (Test-Path $ExtDir) { Remove-Item $ExtDir -Recurse -Force }
Write-Host "Extracting..."
Expand-Archive $ZipFile $ExtDir -Force

# The zip contains a QuickLabel\ folder at the root
$Src = Join-Path $ExtDir "QuickLabel"
if (-not (Test-Path $Src)) {
    # Fallback: zip root is the app folder itself
    $Src = $ExtDir
}

Write-Host "Installing to $Root ..."

# Preserve user data
$Preserve = @(".venv", "venv", "models", "wheels", "projects")

Get-ChildItem $Src | ForEach-Object {
    if ($Preserve -contains $_.Name) { return }
    $Dst = Join-Path $Root $_.Name
    if ($_.PSIsContainer) {
        Copy-Item $_.FullName $Dst -Recurse -Force
    } else {
        Copy-Item $_.FullName $Dst -Force
    }
}

# Cleanup temp files
Remove-Item $ZipFile -Force -ErrorAction SilentlyContinue
Remove-Item $ExtDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "`nUpdated to v$LatestVer successfully!" -ForegroundColor Green
Write-Host "Restart QuickLabel.exe to use the new version."
pause

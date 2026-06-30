# QuickLabel updater (UTF-8 with BOM required for PowerShell 5.1 on Russian Windows)
# Downloads the latest release from GitHub and extracts it alongside this script,
# replacing QuickLabel.exe and source code but keeping .venv, models, and projects.
#
# Usage: .\update.ps1
#   or:  .\update.ps1 -Force   (skip version check)

param([switch]$Force)

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
Write-Host "Checking latest release on github.com/$Repo ..."

try {
    $Release = Invoke-RestMethod "$ApiBase/releases/latest" -ErrorAction Stop
} catch {
    Write-Host "ERROR: Cannot reach GitHub. Check your internet connection." -ForegroundColor Red
    Write-Host $_.Exception.Message
    pause
    exit 1
}

$LatestTag = $Release.tag_name          # e.g. "v1.2.0"
$LatestVer = $LatestTag.TrimStart("v")
$LocalVer  = Get-LocalVersion

Write-Host "  Installed : v$LocalVer"
Write-Host "  Available : $LatestTag"

if (-not $Force -and $LatestVer -eq $LocalVer) {
    Write-Host "`nAlready up to date." -ForegroundColor Green
    pause
    exit 0
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

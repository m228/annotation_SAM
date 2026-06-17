# QuickLabel one-time setup (Windows / PowerShell).
# Creates QuickLabel\.venv and installs everything needed to run, including the
# bundled SAM 2 / SAM 3 wheels (sam3 is not on PyPI). Requires Python 3.13 and,
# for GPU acceleration, an NVIDIA card with recent drivers.
#
#   .\setup.ps1            # GPU build (CUDA 12.4)
#   .\setup.ps1 -CpuOnly   # CPU-only torch (no GPU; SAM 3 will be slow)

param([switch]$CpuOnly)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

# 1) Find a Python 3.13 launcher.
$Py = $null
foreach ($cmd in @("py -3.13", "python", "python3")) {
    try {
        $parts = $cmd.Split(" ")
        $ver = & $parts[0] $parts[1..($parts.Length-1)] --version 2>&1
        if ($ver -match "3\.13") { $Py = $cmd; break }
    } catch {}
}
if (-not $Py) { Write-Error "Python 3.13 not found. Install it from python.org first."; exit 1 }
Write-Host "Using Python: $Py ($((& $Py.Split(' ')[0] $Py.Split(' ')[1..9] --version 2>&1)))" -ForegroundColor Cyan

# 2) Create the venv.
$VenvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    & $Py.Split(" ")[0] $Py.Split(" ")[1..9] -m venv (Join-Path $Root ".venv")
}
& $VenvPy -m pip install --upgrade pip

# 3) torch / torchvision (must come first, from the right index).
if ($CpuOnly) {
    & $VenvPy -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
} else {
    & $VenvPy -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
}

# 4) Server + SAM dependencies from PyPI.
& $VenvPy -m pip install -r (Join-Path $Root "requirements.txt")

# 5) Bundled SAM wheels (sam2 + sam3; sam3 is --no-deps, numpy<2 pin is ignored).
& $VenvPy -m pip install (Join-Path $Root "wheels\sam2-1.1.0-py3-none-any.whl")
& $VenvPy -m pip install --no-deps (Join-Path $Root "wheels\sam3-0.1.0-py3-none-any.whl")

Write-Host "`nDone. Make sure models\sam2.1_hiera_large.pt and models\sam3.pt are present, then run .\run.ps1" -ForegroundColor Green

# QuickLabel launcher (Windows / PowerShell)
# Self-contained: bundles its own ml_backend SAM source, model checkpoints and
# SAM wheels. Auto-detects the Python venv so the folder is portable.

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

# Resolve the interpreter: explicit override → local .venv → parent .venv.
$Candidates = @(
    $env:QUICKLABEL_PYTHON,
    (Join-Path $Root ".venv\Scripts\python.exe"),
    (Join-Path (Split-Path -Parent $Root) ".venv\Scripts\python.exe")
)
$Venv = $Candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1

if (-not $Venv) {
    Write-Error "Python venv not found. Run .\setup.ps1 first (creates .venv and installs deps)."
    exit 1
}

Set-Location $Root
Write-Host "Starting QuickLabel with $Venv" -ForegroundColor Cyan
& $Venv -m backend.server

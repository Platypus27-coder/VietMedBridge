param([string]$Python = "", [switch]$Locked)
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot
try {
    if (-not $Python) {
        $ManagedPython = Join-Path $ProjectRoot '.tools/python'
        $Candidate = Get-ChildItem -LiteralPath $ManagedPython -Filter python.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($Candidate) { $Python = $Candidate.FullName }
        else { $Python = 'python' }
    }
    & $Python -c "import sys; assert sys.version_info[:2] == (3,11), 'Select Python 3.11 with -Python <executable>'"
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required.' }
    $VenvPython = Join-Path $ProjectRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'venv creation failed.' }
    }
    & $VenvPython -c "import sys; assert sys.version_info[:2] == (3,11), 'Existing venv requires Python 3.11'"
    if ($LASTEXITCODE -ne 0) { throw 'Use a Python 3.11 venv.' }
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed.' }
    $Requirements = if ($Locked) { 'requirements-lock.txt' } else { 'requirements.txt' }
    & $VenvPython -m pip install -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $VenvPython --version
    & $VenvPython -m pip --version
    Write-Host 'Ready. Activate with .\.venv\Scripts\Activate.ps1'
} finally { Pop-Location }

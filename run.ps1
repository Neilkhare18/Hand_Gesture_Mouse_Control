# PowerShell Portable Launcher for Hand Gesture Mouse Control
Set-Location -Path $PSScriptRoot

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Touchless Hand Gesture Mouse Control System" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan

# 1. Check if standalone executable exists
$ExePath = Join-Path $PSScriptRoot "dist\HandGestureMouse\HandGestureMouse.exe"
if (Test-Path $ExePath) {
    Write-Host "[OK] Found standalone application bundle." -ForegroundColor Green
    Write-Host "Launching HandGestureMouse.exe..." -ForegroundColor Cyan
    & $ExePath @args
    exit $LASTEXITCODE
}

# 2. Dynamically search for Python on this computer
$PythonCandidates = @(
    "python",
    "py",
    "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
    "$env:ProgramFiles\Python313\python.exe",
    "$env:ProgramFiles\Python312\python.exe",
    "$env:ProgramFiles\Python311\python.exe",
    "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe"
)

$ValidPython = $null
foreach ($cand in $PythonCandidates) {
    try {
        $out = & $cand --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $out -match "Python 3\.") {
            $ValidPython = $cand
            break
        }
    } catch {
        # Continue searching
    }
}

if (-not $ValidPython) {
    Write-Host "`n[!] Python was not found on this computer." -ForegroundColor Red
    Write-Host "`nTo run this application:" -ForegroundColor Yellow
    Write-Host " 1. Run this command in terminal to install Python automatically:" -ForegroundColor Yellow
    Write-Host "    winget install Python.Python.3.11 -e" -ForegroundColor White
    Write-Host " 2. Or download Python from: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "    (Check 'Add Python to PATH' during installation)`n" -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
    exit 1
}

Write-Host "[OK] Using Python: $ValidPython" -ForegroundColor Green

# Check dependencies
$depsCheck = & $ValidPython -c "import cv2, mediapipe, numpy, pyautogui, pynput" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[*] Installing required packages from requirements.txt..." -ForegroundColor Yellow
    Write-Host "    (This only runs once on first startup)`n" -ForegroundColor Gray
    $reqPath = Join-Path $PSScriptRoot "requirements.txt"
    & $ValidPython -m pip install -r $reqPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Failed to install dependencies." -ForegroundColor Red
        Read-Host "Press Enter to exit..."
        exit 1
    }
}

Write-Host "`nLaunching application...`n" -ForegroundColor Cyan
$mainPy = Join-Path $PSScriptRoot "main.py"
& $ValidPython $mainPy @args

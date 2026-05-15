# Build and package ARIAKE OCTA Stack Registration for Windows distribution.
# Output: dist\ARIAKE_OCTA_Stack_Registration_v<version>_win64.zip
#
# Prerequisites:
#   - Python 3.14 venv with: pip install -r requirements.txt
#   - Windows Developer Mode ON (Settings -> System -> For developers)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

chcp 65001 | Out-Null
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:FLET_CLI_NO_RICH_OUTPUT = "1"

if (-not (Test-Path "venv\Scripts\flet.exe")) {
    Write-Error "venv not found. Run: py -3.14 -m venv venv; .\venv\Scripts\pip install -r requirements.txt"
}

# Read version from pyproject.toml
$version = "0.1.0"
if (Test-Path "pyproject.toml") {
    if ((Get-Content "pyproject.toml" -Raw) -match 'version\s*=\s*"([^"]+)"') {
        $version = $Matches[1]
    }
}

$productName = "ARIAKE_OCTA_Stack_Registration"
$distName = "${productName}_v${version}_win64"
$releaseSrc = Join-Path $Root "build\flutter\build\windows\x64\runner\Release"
$vcRedistDir = Join-Path $Root "build\vcredist"
$cmakeInstall = Join-Path $Root "build\flutter\build\windows\x64\cmake_install.cmake"
$cmakeExe = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"

function Stage-VcRuntimes {
    Write-Host "Staging VC runtime DLLs (32-bit CMake workaround)..."
    New-Item -ItemType Directory -Force -Path $vcRedistDir | Out-Null
    $dlls = @("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
    foreach ($dll in $dlls) {
        $src = Join-Path $env:SystemRoot "System32\$dll"
        if (-not (Test-Path $src)) {
            Write-Error "Missing required runtime: $src`nInstall: https://aka.ms/vcredist"
        }
        Copy-Item $src (Join-Path $vcRedistDir $dll) -Force
    }
}

function Patch-CmakeInstall {
    if (-not (Test-Path $cmakeInstall)) {
        return $false
    }
    $vc = ($vcRedistDir -replace '\\', '/')
    $content = Get-Content $cmakeInstall -Raw
    $content = $content -replace 'C:/WINDOWS/System32/msvcp140\.dll', "$vc/msvcp140.dll"
    $content = $content -replace 'C:/WINDOWS/System32/vcruntime140\.dll', "$vc/vcruntime140.dll"
    $content = $content -replace 'C:/WINDOWS/System32/vcruntime140_1\.dll', "$vc/vcruntime140_1.dll"
    Set-Content $cmakeInstall $content -NoNewline -Encoding utf8
    return $true
}

function Complete-CmakeInstall {
    if (-not (Test-Path $cmakeExe)) {
        Write-Error "CMake not found. Install Visual Studio 2022 Build Tools with C++ desktop development."
    }
    Write-Host "Running CMake install step..."
    & $cmakeExe -DBUILD_TYPE=Release -P $cmakeInstall
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CMake install failed (exit $LASTEXITCODE)"
    }
}

function Package-Distribution {
    $distRoot = Join-Path $Root "dist"
    $distDir = Join-Path $distRoot $distName
    $zipPath = Join-Path $distRoot "$distName.zip"

    if (-not (Test-Path (Join-Path $releaseSrc "stack-reg.exe"))) {
        Write-Error "Release build not found: $releaseSrc\stack-reg.exe"
    }

    Write-Host "Packaging distribution folder..."
    if (Test-Path $distDir) { Remove-Item $distDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $distDir | Out-Null
    Copy-Item -Path "$releaseSrc\*" -Destination $distDir -Recurse -Force

    @"
ARIAKE OCTA Stack Registration v$version (Windows x64)

Run: $productName.exe  (or stack-reg.exe in this folder)

Requirements:
  - Windows 10/11 x64
  - No Python installation required

Distribute this entire folder (or the zip file) to end users.
"@ | Set-Content (Join-Path $distDir "README.txt") -Encoding utf8

    # Launcher with friendly name (copy of stack-reg.exe)
    Copy-Item (Join-Path $distDir "stack-reg.exe") (Join-Path $distDir "$productName.exe") -Force

    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Write-Host "Creating zip archive..."
    Compress-Archive -Path $distDir -DestinationPath $zipPath -CompressionLevel Optimal

    $zipMb = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
    Write-Host ""
    Write-Host "=== Distribution package ready ==="
    Write-Host "Folder: $distDir"
    Write-Host "Zip:    $zipPath  ($zipMb MB)"
    Write-Host "Run:    $(Join-Path $distDir "$productName.exe")"
}

# --- Main ---
Write-Host "=== Step 1/3: flet build windows (package Python + compile) ==="
$buildExit = 0
try {
    & .\venv\Scripts\flet.exe build windows --yes --no-rich-output
    if ($LASTEXITCODE -ne 0) { $buildExit = $LASTEXITCODE }
} catch {
    $buildExit = 1
}

Write-Host "=== Step 2/3: Complete install (VC runtime workaround) ==="
Stage-VcRuntimes
if (-not (Patch-CmakeInstall)) {
    Write-Error "cmake_install.cmake not found. Run flet build first."
}
Complete-CmakeInstall

Write-Host "=== Step 3/3: Package for distribution ==="
Package-Distribution

if ($buildExit -ne 0) {
    Write-Host ""
    Write-Host "Note: flet reported a non-zero exit code during flutter build,"
    Write-Host "but the install step was completed manually and packaging succeeded."
}

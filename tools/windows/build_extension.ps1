[CmdletBinding()]
param(
    [string]$Python = 'C:\Users\23642\miniconda3\envs\tree-detr\python.exe',
    [string]$VsDevCmd = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$EnvRoot = Split-Path -Parent $Python
$CudaHome = Join-Path $EnvRoot 'Library'

foreach ($RequiredPath in @($Python, $VsDevCmd, (Join-Path $CudaHome 'bin\nvcc.exe'))) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required build dependency was not found: $RequiredPath"
    }
}

$OpsDir = Join-Path $ProjectRoot 'models\ops'
$Build = 'call "{0}" -arch=x64 -host_arch=x64 && set "DISTUTILS_USE_SDK=1" && set "MSSdk=1" && set "CUDA_HOME={1}" && set "TORCH_CUDA_ARCH_LIST=12.0" && set "MAX_JOBS=1" && "{2}" setup.py build install' -f $VsDevCmd, $CudaHome, $Python
Push-Location $OpsDir
try {
    cmd.exe /d /c $Build
    if ($LASTEXITCODE -ne 0) { throw "CUDA extension build failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Host 'CUDA extension built for sm_120 (RTX 5060) successfully.'

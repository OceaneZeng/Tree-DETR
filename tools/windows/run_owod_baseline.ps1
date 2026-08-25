param(
    [ValidateSet('vanilla_d_detr', 'ore_star', 'ow_detr', 'prob', 'oracle')]
    [string]$Method = 'prob',
    [string]$CocoPath = 'data/coco',
    [Parameter(Mandatory = $true)] [string]$TrainAnn,
    [Parameter(Mandatory = $true)] [string]$ValAnn,
    [Parameter(Mandatory = $true)] [string]$OutputDir,
    [string]$Manifest = '',
    [int]$Stage = -1,
    [string]$PythonBin = 'python'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$Runner = Join-Path $ProjectRoot 'tools/owod/run_baseline.py'
$Args = @('--method', $Method, '--coco-path', $CocoPath,
         '--train-ann', $TrainAnn, '--val-ann', $ValAnn,
         '--output-dir', $OutputDir)
if ($Manifest) { $Args += @('--manifest', $Manifest) }
if ($Stage -ge 0) { $Args += @('--stage', $Stage) }
& $PythonBin $Runner @Args
exit $LASTEXITCODE

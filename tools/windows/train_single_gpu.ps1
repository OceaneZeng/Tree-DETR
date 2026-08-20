[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CocoPath,
    [string]$OutputDir = '',
    [string]$Resume = '',
    [string]$Pretrained = '',
    [int]$BatchSize = 1,
    [int]$Epochs = 50,
    [int]$NumWorkers = 2,
    [int]$EvalInterval = 1,
    [double]$LearningRate = 1.25e-5,
    [double]$BackboneLearningRate = 1.25e-6,
    [double]$ClassEmbedLearningRateMultiplier = 1.0,
    [string]$Backbone = 'resnet50',
    [int]$NumClasses = 0,
    [int]$NumQueries = 300,
    [int]$EncoderLayers = 6,
    [int]$DecoderLayers = 6,
    [int]$FeedForwardDim = 1024,
    [int]$Seed = 42,
    [switch]$Lightweight,
    [switch]$NoAugmentation,
    [switch]$WithTree,
    [switch]$Eval,
    [string]$Python = 'C:\Users\23642\miniconda3\envs\tree-detr\python.exe'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$CocoPath = (Resolve-Path $CocoPath).Path
$Required = @(
    (Join-Path $CocoPath 'train2017'),
    (Join-Path $CocoPath 'val2017'),
    (Join-Path $CocoPath 'annotations\instances_train2017.json'),
    (Join-Path $CocoPath 'annotations\instances_val2017.json')
)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "COCO 2017 is incomplete; required path not found: $Path"
    }
}
if (-not (Test-Path -LiteralPath $Python)) { throw "Python executable not found: $Python" }
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'models\ops\MultiScaleDeformableAttention.cp310-win_amd64.pyd'))) {
    throw 'The CUDA extension is missing. Run tools\\windows\\build_extension.ps1 first.'
}

# Keep the local build importable even before setup.py install has refreshed
# site-packages, for example after a source-only rebuild.
$OpsDir = Join-Path $ProjectRoot 'models\ops'
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$OpsDir;$env:PYTHONPATH" } else { $OpsDir }

if (-not $OutputDir) {
    $RunName = if ($WithTree) { 'tree_ee0_r50_bs1' } else { 'baseline_r50_bs1' }
    $OutputDir = Join-Path $ProjectRoot (Join-Path 'exps' $RunName)
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Arguments = @(
    '-u', 'main.py',
    '--coco_path', $CocoPath,
    '--output_dir', $OutputDir,
    '--batch_size', $BatchSize,
    '--epochs', $Epochs,
    '--num_workers', $NumWorkers,
    '--eval_interval', $EvalInterval,
    '--lr', $LearningRate,
    '--lr_backbone', $BackboneLearningRate,
    '--class_embed_lr_mult', $ClassEmbedLearningRateMultiplier,
    '--backbone', $Backbone,
    '--num_queries', $NumQueries,
    '--enc_layers', $EncoderLayers,
    '--dec_layers', $DecoderLayers,
    '--dim_feedforward', $FeedForwardDim,
    '--seed', $Seed,
    '--device', 'cuda'
)
if ($NumClasses -gt 0) { $Arguments += @('--num_classes', $NumClasses) }
if ($Lightweight) { $Arguments += '--lightweight' }
if ($NoAugmentation) { $Arguments += '--no-augmentation' }
if ($WithTree) { $Arguments += '--with_tree' }
if ($Resume) { $Arguments += @('--resume', $Resume) }
if ($Pretrained) { $Arguments += @('--pretrained', $Pretrained) }
if ($Eval) { $Arguments += '--eval' }

Write-Host "Starting $(if ($WithTree) { 'Tree-DETR EE-0' } else { 'Deformable-DETR baseline' }) at $OutputDir"
Write-Host "batch_size=$BatchSize lr=$LearningRate epochs=$Epochs"
Push-Location $ProjectRoot
try {
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell wraps native stderr lines as ErrorRecords. Merge
        # them explicitly so warnings remain visible/loggable without becoming
        # terminating errors under the wrapper's Stop policy.
        $ErrorActionPreference = 'Continue'
        & $Python @Arguments 2>&1 | ForEach-Object { Write-Output $_ }
        $PythonExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($PythonExitCode -ne 0) { throw "Training exited with code $PythonExitCode" }
}
finally {
    Pop-Location
}

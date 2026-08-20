[CmdletBinding()]
param(
    [string]$DataRoot = 'C:\programlearning\datasets\oxford-pet-tree-detr',
    [int]$BaselineEpochs = 20,
    [int]$IncrementEpochs = 10,
    [int]$ReplayBudget = 64,
    [ValidateRange(1, 1000)]
    [int]$MinMatchedPerClass = 20,
    [int]$DataSeed = 42,
    [int]$Seed = 42,
    [ValidateRange(1, 64)]
    [int]$BatchSize = 4,
    [string]$BaselineCheckpoint = '',
    [switch]$ForceBaseline,
    [switch]$RunDespiteQualityGate,
    [switch]$ModuleAblations,
    [string]$OutputTag = '',
    [string]$Python = 'C:\Users\23642\miniconda3\envs\tree-detr\python.exe'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PrepareScript = Join-Path $ProjectRoot 'tools\data\prepare_oxford_pet.py'
$TrainScript = Join-Path $ProjectRoot 'tools\windows\train_single_gpu.ps1'
$IncrementScript = Join-Path $ProjectRoot 'tools\graph_local\run_increment.py'

if (-not (Test-Path -LiteralPath $Python)) { throw "Python executable not found: $Python" }

& $Python $PrepareScript --root $DataRoot --seed $DataSeed --unknown-per-species 4 `
    --train-per-class 30 --val-per-class 20 --unknown-val-per-class 20 `
    --increment-per-class 20 --increment-unknown-index 0
if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed with code $LASTEXITCODE" }

$CocoPath = Join-Path $DataRoot 'coco_pet_mini'
$MetadataPath = Join-Path $CocoPath 'split_metadata.json'
$Metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
$ExperimentRoot = Join-Path $ProjectRoot 'exps\pet_graph_local'
$BaselineDir = Join-Path $ExperimentRoot "baseline_seed$Seed"
$DefaultBaselineCheckpoint = Join-Path $BaselineDir 'checkpoint.pth'
$BaselineComplete = Join-Path $BaselineDir 'training_complete.json'

$Common = @{
    CocoPath = $CocoPath
    BatchSize = $BatchSize
    NumWorkers = 0
    LearningRate = 1e-4
    BackboneLearningRate = 1e-5
    Backbone = 'resnet18'
    NumClasses = [int]$Metadata.num_known
    NumQueries = 100
    EncoderLayers = 4
    DecoderLayers = 4
    FeedForwardDim = 512
    EvalInterval = 5
    Seed = $Seed
    Lightweight = $true
    Python = $Python
}

if ($BaselineCheckpoint) {
    $BaselineCheckpoint = (Resolve-Path -LiteralPath $BaselineCheckpoint).Path
    if ($ForceBaseline) { throw '-ForceBaseline cannot be used with -BaselineCheckpoint' }
    Write-Host "Using explicitly supplied baseline checkpoint: $BaselineCheckpoint"
}
else {
    $BaselineCheckpoint = $DefaultBaselineCheckpoint
    if ($ForceBaseline) {
        & $TrainScript @Common -OutputDir $BaselineDir -Epochs $BaselineEpochs
    }
    elseif (Test-Path -LiteralPath $BaselineComplete) {
        $Completed = Get-Content -LiteralPath $BaselineComplete -Raw | ConvertFrom-Json
        if ([int]$Completed.epochs -lt $BaselineEpochs) {
            Write-Host "Extending baseline from $($Completed.epochs) to $BaselineEpochs epochs"
            & $TrainScript @Common -OutputDir $BaselineDir -Epochs $BaselineEpochs -Resume $BaselineCheckpoint
        }
        else {
            Write-Host "Using existing baseline checkpoint: $BaselineCheckpoint"
        }
    }
    elseif (Test-Path -LiteralPath $BaselineCheckpoint) {
        & $TrainScript @Common -OutputDir $BaselineDir -Epochs $BaselineEpochs -Resume $BaselineCheckpoint
    }
    else {
        & $TrainScript @Common -OutputDir $BaselineDir -Epochs $BaselineEpochs
    }
}
if ($LASTEXITCODE -ne 0) { throw "Baseline training failed with code $LASTEXITCODE" }

$RunName = "increment_seed$Seed"
if ($OutputTag) { $RunName = "${RunName}_$OutputTag" }
$RunDir = Join-Path $ExperimentRoot $RunName
$Arguments = @(
    $IncrementScript,
    '--coco_path', $CocoPath,
    '--baseline', $BaselineCheckpoint,
    '--metadata', $MetadataPath,
    '--output_dir', $RunDir,
    '--device', 'cuda',
    '--batch_size', $BatchSize,
    '--num_workers', '0',
    '--backbone', 'resnet18',
    '--num_queries', '100',
    '--enc_layers', '4',
    '--dec_layers', '4',
    '--dim_feedforward', '512',
    '--lightweight',
    '--increment-epochs', $IncrementEpochs,
    '--replay-budget', $ReplayBudget,
    '--min-matched-per-class', $MinMatchedPerClass,
    '--seed', $Seed
)
if ($RunDespiteQualityGate) { $Arguments += '--run-despite-quality-gate' }
if ($ModuleAblations) { $Arguments += '--module-ablations' }

Push-Location $ProjectRoot
$PythonStderrLog = Join-Path ([System.IO.Path]::GetTempPath()) "tree-detr-graph-local-$([guid]::NewGuid().ToString('N')).stderr.log"
try {
    # Keep native stderr out of Windows PowerShell's error stream.  It can
    # otherwise turn harmless Python warnings into terminating errors when
    # $ErrorActionPreference is Stop.
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'SilentlyContinue'
        & $Python @Arguments 2> $PythonStderrLog
        $PythonExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if (Test-Path -LiteralPath $PythonStderrLog) {
        Get-Content -LiteralPath $PythonStderrLog
    }
    if ($PythonExitCode -ne 0) { throw "Graph-local experiment failed with code $PythonExitCode" }
}
finally {
    Remove-Item -LiteralPath $PythonStderrLog -Force -ErrorAction SilentlyContinue
    Pop-Location
}

Write-Host "Graph-local experiment completed. Review $(Join-Path $RunDir 'summary.json')"

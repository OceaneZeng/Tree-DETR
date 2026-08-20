[CmdletBinding()]
param(
    [string]$DataRoot = 'C:\programlearning\datasets\oxford-pet-tree-detr',
    [int]$KnownPerSpecies = 3,
    [int]$TrainPerClass = 60,
    [int]$ValPerClass = 30,
    [int]$OverfitPerClass = 4,
    [int]$OverfitEpochs = 100,
    [int]$BaselineEpochs = 100,
    [int]$Seed = 42,
    [double]$OverfitAp50Gate = 0.90,
    [double]$BaselineAp50Gate = 0.40,
    [string]$Python = 'C:\Users\23642\miniconda3\envs\tree-detr\python.exe'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PrepareScript = Join-Path $ProjectRoot 'tools\data\prepare_oxford_pet.py'
$TrainScript = Join-Path $ProjectRoot 'tools\windows\train_single_gpu.ps1'
$DatasetName = "coco_pet_small$($KnownPerSpecies * 2)"
$ExperimentRoot = Join-Path $ProjectRoot 'exps\pet_baseline_diagnostics'

if (-not (Test-Path -LiteralPath $Python)) { throw "Python executable not found: $Python" }

& $Python $PrepareScript --root $DataRoot --seed $Seed --unknown-per-species 4 `
    --known-per-species $KnownPerSpecies --train-per-class $TrainPerClass `
    --val-per-class $ValPerClass --unknown-val-per-class $ValPerClass `
    --increment-per-class $TrainPerClass --increment-unknown-index 0 `
    --overfit-per-class $OverfitPerClass --dataset-name $DatasetName
if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed with code $LASTEXITCODE" }

$Metadata = Get-Content -LiteralPath (Join-Path $DataRoot "$DatasetName\split_metadata.json") -Raw | ConvertFrom-Json
$Common = @{
    BatchSize = 4
    NumWorkers = 0
    LearningRate = 1e-4
    BackboneLearningRate = 1e-5
    Backbone = 'resnet18'
    NumClasses = [int]$Metadata.num_known
    NumQueries = 20
    EncoderLayers = 2
    DecoderLayers = 2
    FeedForwardDim = 256
    EvalInterval = 5
    Seed = $Seed
    Lightweight = $true
    Python = $Python
}

function Get-FinalAp50([string]$OutputDir) {
    $LastLog = Get-Content -LiteralPath (Join-Path $OutputDir 'log.txt') | Select-Object -Last 1 | ConvertFrom-Json
    if ($null -eq $LastLog.test_coco_eval_bbox) { throw "No COCO evaluation recorded in $OutputDir" }
    return [double]$LastLog.test_coco_eval_bbox[1]
}

$OverfitOutput = Join-Path $ExperimentRoot "overfit_small$($KnownPerSpecies * 2)_seed$Seed"
& $TrainScript @Common -CocoPath (Join-Path $DataRoot "${DatasetName}_overfit") `
    -OutputDir $OverfitOutput -Epochs $OverfitEpochs -NoAugmentation
if ($LASTEXITCODE -ne 0) { throw "Overfit diagnostic failed with code $LASTEXITCODE" }
$OverfitAp50 = Get-FinalAp50 $OverfitOutput
Write-Host "Overfit AP50=$OverfitAp50; gate=$OverfitAp50Gate"
if ($OverfitAp50 -lt $OverfitAp50Gate) {
    throw "Overfit gate failed. Check labels, transforms, or evaluator before running a generalization baseline."
}

$BaselineOutput = Join-Path $ExperimentRoot "baseline_small$($KnownPerSpecies * 2)_seed$Seed"
& $TrainScript @Common -CocoPath (Join-Path $DataRoot $DatasetName) `
    -OutputDir $BaselineOutput -Epochs $BaselineEpochs
if ($LASTEXITCODE -ne 0) { throw "Small-task baseline failed with code $LASTEXITCODE" }
$BaselineAp50 = Get-FinalAp50 $BaselineOutput
Write-Host "Small-task baseline AP50=$BaselineAp50; gate=$BaselineAp50Gate"
if ($BaselineAp50 -lt $BaselineAp50Gate) {
    throw "Small-task generalization gate failed. Do not run incremental adaptation."
}

Write-Host "Baseline diagnostics passed. The small-task dataset and checkpoint are ready for a later incremental experiment."

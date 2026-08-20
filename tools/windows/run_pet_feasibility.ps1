[CmdletBinding()]
param(
    [string]$DataRoot = 'C:\programlearning\datasets\oxford-pet-tree-detr',
    [int]$BaselineEpochs = 20,
    [int]$FineTuneEpochs = 5,
    [switch]$PrepareOnly,
    [switch]$RunTreeAfterPass,
    [switch]$ForceTrain,
    [string]$Python = 'C:\Users\23642\miniconda3\envs\tree-detr\python.exe'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PrepareScript = Join-Path $ProjectRoot 'tools\data\prepare_oxford_pet.py'
$TrainScript = Join-Path $ProjectRoot 'tools\windows\train_single_gpu.ps1'

$PreviousErrorActionPreference = $ErrorActionPreference
try {
    $ErrorActionPreference = 'Continue'
    & $Python $PrepareScript --root $DataRoot --seed 42 --unknown-per-species 4 `
        --train-per-class 30 --val-per-class 12 --unknown-val-per-class 20 `
        2>&1 | ForEach-Object { Write-Output $_ }
    $PrepareExitCode = $LASTEXITCODE
}
finally { $ErrorActionPreference = $PreviousErrorActionPreference }
if ($PrepareExitCode -ne 0) { throw "Dataset preparation failed with code $PrepareExitCode" }

$CocoPath = Join-Path $DataRoot 'coco_pet_mini'
$MetadataPath = Join-Path $CocoPath 'split_metadata.json'
$Metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
if ($PrepareOnly) {
    Write-Host "Prepared $($Metadata.train_images) training images at $CocoPath"
    return
}

$ExperimentRoot = Join-Path $ProjectRoot 'exps\pet_feasibility'
$BaselineDir = Join-Path $ExperimentRoot 'baseline_pretrain'
$BaselineCheckpoint = Join-Path $BaselineDir 'checkpoint.pth'
$BaselineComplete = Join-Path $BaselineDir 'training_complete.json'
$Common = @{
    CocoPath = $CocoPath
    BatchSize = 8
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
    Lightweight = $true
    Python = $Python
}

if ($ForceTrain) {
    & $TrainScript @Common -OutputDir $BaselineDir -Epochs $BaselineEpochs
}
elseif (Test-Path -LiteralPath $BaselineComplete) {
    Write-Host "Using existing baseline checkpoint: $BaselineCheckpoint"
}
elseif (Test-Path -LiteralPath $BaselineCheckpoint) {
    Write-Host "Resuming incomplete baseline checkpoint: $BaselineCheckpoint"
    & $TrainScript @Common -OutputDir $BaselineDir -Epochs $BaselineEpochs `
        -Resume $BaselineCheckpoint
}
else {
    & $TrainScript @Common -OutputDir $BaselineDir -Epochs $BaselineEpochs
}

$Features = Join-Path $ExperimentRoot 'stage0_features.npz'
$Stage0Ann = Join-Path $CocoPath 'annotations\instances_stage0.json'
try {
    $ErrorActionPreference = 'Continue'
    & $Python (Join-Path $ProjectRoot 'tools\stage0\extract_features.py') `
        --coco_path $CocoPath --stage0_ann $Stage0Ann --split_metadata $MetadataPath `
        --features_output $Features --resume $BaselineCheckpoint --device cuda `
        --backbone resnet18 --num_classes ([int]$Metadata.num_known) --num_queries 100 `
        --enc_layers 4 --dec_layers 4 --dim_feedforward 512 --lightweight --num_workers 0 `
        2>&1 | ForEach-Object { Write-Output $_ }
    $ExtractExitCode = $LASTEXITCODE
}
finally { $ErrorActionPreference = $PreviousErrorActionPreference }
if ($ExtractExitCode -ne 0) { throw "Stage-0 feature extraction failed with code $ExtractExitCode" }

try {
    $ErrorActionPreference = 'Continue'
    & $Python (Join-Path $ProjectRoot 'tools\stage0\run_stage0.py') `
        --features $Features --seed 42 2>&1 | ForEach-Object { Write-Output $_ }
    $Stage0Exit = $LASTEXITCODE
}
finally { $ErrorActionPreference = $PreviousErrorActionPreference }
if ($Stage0Exit -ne 0) {
    Write-Warning 'One or more Stage-0 feasibility gates failed; tree fine-tuning was not started.'
    return
}

if (-not $RunTreeAfterPass) {
    Write-Host 'Stage-0 passed. Re-run with -RunTreeAfterPass for the matched control/tree fine-tuning pair.'
    return
}

$ControlDir = Join-Path $ExperimentRoot 'control_finetune'
$TreeDir = Join-Path $ExperimentRoot 'tree_ee0_finetune'
& $TrainScript @Common -OutputDir $ControlDir -Epochs $FineTuneEpochs -Pretrained $BaselineCheckpoint
if ($LASTEXITCODE -ne 0) { throw "Control fine-tuning failed with code $LASTEXITCODE" }
& $TrainScript @Common -OutputDir $TreeDir -Epochs $FineTuneEpochs `
    -Pretrained $BaselineCheckpoint -WithTree
if ($LASTEXITCODE -ne 0) { throw "Tree fine-tuning failed with code $LASTEXITCODE" }

Write-Host "Matched fine-tuning runs completed under $ExperimentRoot"

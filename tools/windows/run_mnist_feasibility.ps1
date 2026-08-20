[CmdletBinding()]
param(
    [string]$DataRoot = 'C:\programlearning\datasets\mnist-tree-detr',
    [int]$BaselineEpochs = 10,
    [int]$FineTuneEpochs = 3,
    [switch]$PrepareOnly,
    [switch]$RunTreeAfterPass,
    [switch]$ForceTrain,
    [string]$Python = 'C:\Users\23642\miniconda3\envs\tree-detr\python.exe'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PrepareScript = Join-Path $ProjectRoot 'tools\data\prepare_mnist_detection.py'
$TrainScript = Join-Path $ProjectRoot 'tools\windows\train_single_gpu.ps1'

& $Python $PrepareScript --root $DataRoot --seed 42 `
    --train-per-class 100 --val-per-class 20 --unknown-val-per-class 40
if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed with code $LASTEXITCODE" }

$CocoPath = Join-Path $DataRoot 'coco_mnist_mini'
$MetadataPath = Join-Path $CocoPath 'split_metadata.json'
$Metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
if ($PrepareOnly) {
    Write-Host "Prepared $($Metadata.train_images) training images at $CocoPath"
    return
}

$ExperimentRoot = Join-Path $ProjectRoot 'exps\mnist_feasibility'
$BaselineDir = Join-Path $ExperimentRoot 'baseline_pretrain'
$BaselineCheckpoint = Join-Path $BaselineDir 'checkpoint.pth'
$Common = @{
    CocoPath = $CocoPath
    BatchSize = 2
    NumWorkers = 0
    LearningRate = 2.5e-5
    BackboneLearningRate = 2.5e-6
    Backbone = 'resnet50'
    NumClasses = [int]$Metadata.num_known
    NumQueries = 50
    EncoderLayers = 3
    DecoderLayers = 3
    FeedForwardDim = 256
    EvalInterval = 2
    Lightweight = $true
    Python = $Python
}

if ($ForceTrain -or -not (Test-Path -LiteralPath $BaselineCheckpoint)) {
    & $TrainScript @Common -OutputDir $BaselineDir -Epochs $BaselineEpochs
    if ($LASTEXITCODE -ne 0) { throw "Baseline training failed with code $LASTEXITCODE" }
}
else {
    Write-Host "Using existing baseline checkpoint: $BaselineCheckpoint"
}

$Features = Join-Path $ExperimentRoot 'stage0_features.npz'
& $Python (Join-Path $ProjectRoot 'tools\stage0\extract_features.py') `
    --coco_path $CocoPath `
    --stage0_ann (Join-Path $CocoPath 'annotations\instances_stage0.json') `
    --split_metadata $MetadataPath --features_output $Features `
    --resume $BaselineCheckpoint --device cuda --backbone resnet50 `
    --num_classes ([int]$Metadata.num_known) --num_queries 50 `
    --enc_layers 3 --dec_layers 3 --dim_feedforward 256 `
    --lightweight --num_workers 0
if ($LASTEXITCODE -ne 0) { throw "Stage-0 feature extraction failed with code $LASTEXITCODE" }

& $Python (Join-Path $ProjectRoot 'tools\stage0\run_stage0.py') --features $Features --seed 42
if ($LASTEXITCODE -ne 0) {
    Write-Warning 'One or more Stage-0 gates failed; matched tree fine-tuning was not started.'
    return
}
if (-not $RunTreeAfterPass) {
    Write-Host 'Stage-0 passed. Re-run with -RunTreeAfterPass for matched control/tree fine-tuning.'
    return
}

& $TrainScript @Common -OutputDir (Join-Path $ExperimentRoot 'control_finetune') `
    -Epochs $FineTuneEpochs -Pretrained $BaselineCheckpoint
if ($LASTEXITCODE -ne 0) { throw "Control fine-tuning failed with code $LASTEXITCODE" }
& $TrainScript @Common -OutputDir (Join-Path $ExperimentRoot 'tree_ee0_finetune') `
    -Epochs $FineTuneEpochs -Pretrained $BaselineCheckpoint -WithTree
if ($LASTEXITCODE -ne 0) { throw "Tree fine-tuning failed with code $LASTEXITCODE" }

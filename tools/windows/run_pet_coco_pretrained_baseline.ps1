[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$CocoCheckpoint,
    [string]$DataRoot = 'C:\programlearning\datasets\oxford-pet-tree-detr',
    [int]$KnownPerSpecies = 3,
    [int]$TrainPerClass = 100,
    [int]$ValPerClass = 30,
    [int]$Epochs = 50,
    [int]$Seed = 42,
    [double]$Ap50Gate = 0.40,
    [string]$Python = 'C:\Users\23642\miniconda3\envs\tree-detr\python.exe'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PrepareScript = Join-Path $ProjectRoot 'tools\data\prepare_oxford_pet.py'
$TrainScript = Join-Path $ProjectRoot 'tools\windows\train_single_gpu.ps1'
$DatasetName = "coco_pet_pretrained_small$($KnownPerSpecies * 2)"
$OutputDir = Join-Path $ProjectRoot "exps\pet_baseline_diagnostics\coco_pretrained_small$($KnownPerSpecies * 2)_seed$Seed"

if (-not (Test-Path -LiteralPath $Python)) { throw "Python executable not found: $Python" }
if (-not (Test-Path -LiteralPath $CocoCheckpoint)) { throw "COCO detector checkpoint not found: $CocoCheckpoint" }
$CocoCheckpoint = (Resolve-Path -LiteralPath $CocoCheckpoint).Path

& $Python $PrepareScript --root $DataRoot --seed $Seed --unknown-per-species 4 `
    --known-per-species $KnownPerSpecies --train-per-class $TrainPerClass `
    --val-per-class $ValPerClass --unknown-val-per-class $ValPerClass `
    --increment-per-class $TrainPerClass --increment-unknown-index 0 `
    --dataset-name $DatasetName
if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed with code $LASTEXITCODE" }

$Metadata = Get-Content -LiteralPath (Join-Path $DataRoot "$DatasetName\split_metadata.json") -Raw | ConvertFrom-Json
& $TrainScript -CocoPath (Join-Path $DataRoot $DatasetName) -OutputDir $OutputDir `
    -Pretrained $CocoCheckpoint -BatchSize 2 -Epochs $Epochs -NumWorkers 0 -EvalInterval 5 `
    -LearningRate 2e-5 -BackboneLearningRate 2e-6 -ClassEmbedLearningRateMultiplier 10 `
    -Backbone resnet50 -NumClasses ([int]$Metadata.num_known) -NumQueries 300 `
    -EncoderLayers 6 -DecoderLayers 6 -FeedForwardDim 1024 -Seed $Seed -Python $Python
if ($LASTEXITCODE -ne 0) { throw "COCO-pretrained Pet baseline failed with code $LASTEXITCODE" }

$Final = Get-Content -LiteralPath (Join-Path $OutputDir 'log.txt') | Select-Object -Last 1 | ConvertFrom-Json
$Ap50 = [double]$Final.test_coco_eval_bbox[1]
Write-Host "COCO-pretrained small-task AP50=$Ap50; gate=$Ap50Gate"
if ($Ap50 -lt $Ap50Gate) {
    throw "Pretrained baseline gate failed. Do not run incremental adaptation."
}
Write-Host "Pretrained baseline passed. Record this checkpoint before considering later continual-learning work."

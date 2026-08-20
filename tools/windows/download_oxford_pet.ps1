[CmdletBinding()]
param(
    [string]$DataRoot = 'C:\programlearning\datasets\oxford-pet-tree-detr',
    [int]$Connections = 8,
    [switch]$DownloadOnly,
    [string]$Python = 'C:\Users\23642\miniconda3\envs\tree-detr\python.exe'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$DownloadDir = Join-Path $DataRoot 'downloads'
New-Item -ItemType Directory -Force -Path $DownloadDir | Out-Null

$Archives = @(
    @{
        Name = 'images.tar.gz'
        Url = 'https://thor.robots.ox.ac.uk/pets/images.tar.gz'
        MinimumBytes = 700MB
    },
    @{
        Name = 'annotations.tar.gz'
        Url = 'https://thor.robots.ox.ac.uk/pets/annotations.tar.gz'
        MinimumBytes = 10MB
    }
)

function Get-Archive {
    param([hashtable]$Archive)

    $Target = Join-Path $DownloadDir $Archive.Name
    if ((Test-Path -LiteralPath $Target) -and
        ((Get-Item -LiteralPath $Target).Length -ge $Archive.MinimumBytes)) {
        Write-Host "Using existing archive: $Target"
        return
    }

    # The Oxford mirror does not support HTTP Range, so segmented or resumed
    # transfers fail with curl error 33. Keep the final name atomic instead.
    $Partial = "$Target.part"
    if (Test-Path -LiteralPath $Partial) {
        Remove-Item -LiteralPath $Partial -Force
    }
    Write-Host "Downloading $($Archive.Name) with curl"
    & curl.exe -L --fail --retry 5 --retry-all-errors --retry-delay 3 `
        --output $Partial $Archive.Url
    if ($LASTEXITCODE -ne 0) { throw "curl download failed: $($Archive.Name)" }
    Move-Item -LiteralPath $Partial -Destination $Target -Force

    $Length = (Get-Item -LiteralPath $Target).Length
    if ($Length -lt $Archive.MinimumBytes) {
        throw "Downloaded archive is too small: $Target ($Length bytes)"
    }
    Write-Host "Verified $Target ($Length bytes)"
}

foreach ($Archive in $Archives) {
    Get-Archive $Archive
}

if ($DownloadOnly) {
    Write-Host "Oxford Pet archives downloaded to $DownloadDir"
    return
}

$PrepareScript = Join-Path $ProjectRoot 'tools\data\prepare_oxford_pet.py'
& $Python $PrepareScript --root $DataRoot --seed 42 --unknown-per-species 4 `
    --train-per-class 30 --val-per-class 12 --unknown-val-per-class 20
if ($LASTEXITCODE -ne 0) { throw "Oxford Pet conversion failed with code $LASTEXITCODE" }

Write-Host "Dataset ready: $(Join-Path $DataRoot 'coco_pet_mini')"
Write-Host 'No training process was started.'

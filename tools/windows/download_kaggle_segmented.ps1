[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Output,
    [string]$Url = 'https://www.kaggle.com/api/v1/datasets/download/devdgohil/the-oxfordiiit-pet-dataset',
    [long]$TotalBytes = 818292036,
    [ValidateRange(1, 16)]
    [int]$Connections = 8
)

$ErrorActionPreference = 'Stop'
$Output = [IO.Path]::GetFullPath($Output)
$SegmentDir = "$Output.segments"
New-Item -ItemType Directory -Force -Path $SegmentDir | Out-Null
$ChunkBytes = [long][Math]::Ceiling($TotalBytes / [double]$Connections)

function Append-File {
    param([string]$Destination, [string]$Source)
    $OutStream = [IO.File]::Open($Destination, [IO.FileMode]::Append,
                                 [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        $InStream = [IO.File]::OpenRead($Source)
        try { $InStream.CopyTo($OutStream) }
        finally { $InStream.Dispose() }
    }
    finally { $OutStream.Dispose() }
}

# Preserve bytes from the earlier verified single-connection attempt as the
# start of segment zero.
$LegacyPartial = "$Output.part"
$SegmentZero = Join-Path $SegmentDir 'segment-00.bin'
if ((Test-Path -LiteralPath $LegacyPartial) -and
    -not (Test-Path -LiteralPath $SegmentZero)) {
    Move-Item -LiteralPath $LegacyPartial -Destination $SegmentZero
}

$Transfers = @()
for ($Index = 0; $Index -lt $Connections; $Index++) {
    $Start = [long]$Index * $ChunkBytes
    $End = [Math]::Min($TotalBytes - 1, $Start + $ChunkBytes - 1)
    $Expected = $End - $Start + 1
    $Segment = Join-Path $SegmentDir ('segment-{0:D2}.bin' -f $Index)
    $Delta = "$Segment.delta"

    if (-not (Test-Path -LiteralPath $Segment)) {
        [IO.File]::WriteAllBytes($Segment, [byte[]]::new(0))
    }
    if (Test-Path -LiteralPath $Delta) {
        Append-File -Destination $Segment -Source $Delta
        Remove-Item -LiteralPath $Delta -Force
    }
    $Existing = (Get-Item -LiteralPath $Segment).Length
    if ($Existing -gt $Expected) {
        throw "Segment $Index is larger than expected ($Existing > $Expected)"
    }
    if ($Existing -eq $Expected) {
        Write-Host "Segment $Index already complete ($Expected bytes)"
        continue
    }

    $RangeStart = $Start + $Existing
    $Range = "$RangeStart-$End"
    Write-Host "Starting segment ${Index}: bytes $Range"
    $Arguments = @(
        '-L', '--fail', '--retry', '8', '--retry-all-errors',
        '--retry-delay', '3', '--range', $Range, '--output', $Delta, $Url
    )
    $Process = Start-Process -FilePath 'curl.exe' -ArgumentList $Arguments `
        -NoNewWindow -PassThru
    $Transfers += [pscustomobject]@{
        Index = $Index; Process = $Process; Segment = $Segment
        Delta = $Delta; Expected = $Expected
    }
}

foreach ($Transfer in $Transfers) {
    $Transfer.Process.WaitForExit()
    if ($Transfer.Process.ExitCode -ne 0) {
        throw "Segment $($Transfer.Index) failed with curl exit code $($Transfer.Process.ExitCode)"
    }
    Append-File -Destination $Transfer.Segment -Source $Transfer.Delta
    Remove-Item -LiteralPath $Transfer.Delta -Force
    $Actual = (Get-Item -LiteralPath $Transfer.Segment).Length
    if ($Actual -ne $Transfer.Expected) {
        throw "Segment $($Transfer.Index) length mismatch: $Actual != $($Transfer.Expected)"
    }
    Write-Host "Verified segment $($Transfer.Index): $Actual bytes"
}

$Combined = "$Output.part"
if (Test-Path -LiteralPath $Combined) { Remove-Item -LiteralPath $Combined -Force }
$CombinedStream = [IO.File]::Create($Combined)
try {
    for ($Index = 0; $Index -lt $Connections; $Index++) {
        $Segment = Join-Path $SegmentDir ('segment-{0:D2}.bin' -f $Index)
        $InputStream = [IO.File]::OpenRead($Segment)
        try { $InputStream.CopyTo($CombinedStream) }
        finally { $InputStream.Dispose() }
    }
}
finally { $CombinedStream.Dispose() }

$CombinedLength = (Get-Item -LiteralPath $Combined).Length
if ($CombinedLength -ne $TotalBytes) {
    throw "Combined archive length mismatch: $CombinedLength != $TotalBytes"
}
Move-Item -LiteralPath $Combined -Destination $Output -Force
Write-Host "Segmented download complete: $Output ($CombinedLength bytes)"

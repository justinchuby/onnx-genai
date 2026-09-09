param(
    [int]$Rounds = 10,
    [string]$TargetDirectory = "target\debug\deps",
    [string]$ArtifactDirectory = "windows-arm64-diagnostics"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$artifactRoot = New-Item -ItemType Directory -Force -Path $ArtifactDirectory
$logRoot = New-Item -ItemType Directory -Force -Path (Join-Path $artifactRoot "isolated-logs")
$binaryRoot = New-Item -ItemType Directory -Force -Path (Join-Path $artifactRoot "binaries")

function Resolve-TestBinary {
    param([Parameter(Mandatory)][string]$Pattern)

    $binary = Get-ChildItem -Path $TargetDirectory -Filter $Pattern |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $binary) {
        throw "No test executable matched '$Pattern' under '$TargetDirectory'."
    }
    return $binary
}

function Copy-DebugFiles {
    param([Parameter(Mandatory)][System.IO.FileInfo]$Binary)

    Copy-Item -Force $Binary.FullName $binaryRoot
    $pdb = [System.IO.Path]::ChangeExtension($Binary.FullName, ".pdb")
    if (Test-Path $pdb) {
        Copy-Item -Force $pdb $binaryRoot
    } else {
        Write-Warning "No adjacent PDB found for $($Binary.Name)."
    }
}

function Invoke-IsolatedFilter {
    param(
        [Parameter(Mandatory)][System.IO.FileInfo]$Binary,
        [Parameter(Mandatory)][string]$Filter,
        [Parameter(Mandatory)][int]$Round,
        [switch]$Exact
    )

    $safeFilter = $Filter -replace '[^A-Za-z0-9_.-]', '_'
    $log = Join-Path $logRoot "$($Binary.BaseName)-$safeFilter-round-$Round.log"
    $arguments = @($Filter, "--nocapture", "--test-threads=1")
    if ($Exact) {
        $arguments += "--exact"
    }

    "binary=$($Binary.FullName)`nfilter=$Filter`nround=$Round`narguments=$($arguments -join ' ')" |
        Set-Content -Path $log
    & $Binary.FullName @arguments 2>&1 | Tee-Object -FilePath $log -Append
    $exitCode = $LASTEXITCODE
    "exit_code=$exitCode" | Add-Content -Path $log
    if ($exitCode -ne 0) {
        return "$($Binary.Name) filter '$Filter' round $Round exited $exitCode"
    }
    return $null
}

if ($Rounds -lt 1 -or $Rounds -gt 25) {
    throw "Rounds must be in the bounded range 1..25; received $Rounds."
}

$libBinary = Resolve-TestBinary "onnx_runtime_ep_cpu-*.exe"
$differentialBinary = Resolve-TestBinary "native_vs_mlas_differential-*.exe"
Copy-DebugFiles $libBinary
Copy-DebugFiles $differentialBinary

$failures = [System.Collections.Generic.List[string]]::new()
for ($round = 1; $round -le $Rounds; $round++) {
    $failure = Invoke-IsolatedFilter `
        -Binary $libBinary `
        -Filter "backend_ab::tests::every_ab_covered_family_has_both_halves" `
        -Round $round `
        -Exact
    if ($null -ne $failure) {
        $failures.Add($failure)
    }

    # The retry crashed after the seven non-SDPA tests had reported success.
    # This substring selects the three remaining SDPA tests in a fresh process.
    $failure = Invoke-IsolatedFilter `
        -Binary $differentialBinary `
        -Filter "sdpa" `
        -Round $round
    if ($null -ne $failure) {
        $failures.Add($failure)
    }

    $failure = Invoke-IsolatedFilter `
        -Binary $differentialBinary `
        -Filter "concurrent_sdpa_sessions_lose_no_work" `
        -Round $round `
        -Exact
    if ($null -ne $failure) {
        $failures.Add($failure)
    }
}

$summary = Join-Path $artifactRoot "isolated-summary.txt"
if ($failures.Count -eq 0) {
    "All $($Rounds * 3) process-isolated invocations passed." | Set-Content $summary
} else {
    @(
        "$($failures.Count) process-isolated invocation(s) failed:"
        $failures
    ) | Set-Content $summary
    $failures | Set-Content (Join-Path $artifactRoot "isolated-failure.txt")
}

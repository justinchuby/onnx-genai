param(
    [int]$Rounds = 10,
    [int]$FullProcessRounds = 5,
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
        [Parameter(Mandatory)][AllowEmptyString()][string]$Filter,
        [Parameter(Mandatory)][int]$Round,
        [switch]$Exact
    )

    $filterLabel = if ([string]::IsNullOrEmpty($Filter)) { "<all-tests>" } else { $Filter }
    $safeFilter = if ([string]::IsNullOrEmpty($Filter)) {
        "all-tests"
    } else {
        $Filter -replace '[^A-Za-z0-9_.-]', '_'
    }
    $log = Join-Path $logRoot "$($Binary.BaseName)-$safeFilter-round-$Round.log"
    $arguments = @()
    if (![string]::IsNullOrEmpty($Filter)) {
        $arguments += $Filter
        $arguments += "--nocapture"
        $arguments += "--test-threads=1"
        if ($Exact) {
            $arguments += "--exact"
        }
    }

    "binary=$($Binary.FullName)`nfilter=$filterLabel`nround=$Round`narguments=$($arguments -join ' ')" |
        Set-Content -Path $log
    & $Binary.FullName @arguments 2>&1 |
        Tee-Object -FilePath $log -Append |
        Out-Host
    $exitCode = $LASTEXITCODE
    "exit_code=$exitCode" | Add-Content -Path $log
    return [pscustomobject]@{
        binary = $Binary.Name
        filter = $filterLabel
        exact = [bool]$Exact
        round = $Round
        exit_code = $exitCode
        log = $log
    }
}

if ($Rounds -lt 1 -or $Rounds -gt 25) {
    throw "Rounds must be in the bounded range 1..25; received $Rounds."
}
if ($FullProcessRounds -lt 1 -or $FullProcessRounds -gt 10) {
    throw "FullProcessRounds must be in the bounded range 1..10; received $FullProcessRounds."
}

$libBinary = Resolve-TestBinary "onnx_runtime_ep_cpu-*.exe"
$differentialBinary = Resolve-TestBinary "native_vs_mlas_differential-*.exe"
Copy-DebugFiles $libBinary
Copy-DebugFiles $differentialBinary

$failures = [System.Collections.Generic.List[string]]::new()
$results = [System.Collections.Generic.List[object]]::new()
for ($round = 1; $round -le $Rounds; $round++) {
    $result = Invoke-IsolatedFilter `
        -Binary $libBinary `
        -Filter "backend_ab::tests::every_ab_covered_family_has_both_halves" `
        -Round $round `
        -Exact
    $results.Add($result)
    if ($result.exit_code -ne 0) {
        $failures.Add("$($result.binary) filter '$($result.filter)' round $round exited $($result.exit_code)")
    }

    # The retry crashed after the seven non-SDPA tests had reported success.
    # This substring selects the three remaining SDPA tests in a fresh process.
    $result = Invoke-IsolatedFilter `
        -Binary $differentialBinary `
        -Filter "sdpa" `
        -Round $round
    $results.Add($result)
    if ($result.exit_code -ne 0) {
        $failures.Add("$($result.binary) filter '$($result.filter)' round $round exited $($result.exit_code)")
    }

    $result = Invoke-IsolatedFilter `
        -Binary $differentialBinary `
        -Filter "concurrent_sdpa_sessions_lose_no_work" `
        -Round $round `
        -Exact
    $results.Add($result)
    if ($result.exit_code -ne 0) {
        $failures.Add("$($result.binary) filter '$($result.filter)' round $round exited $($result.exit_code)")
    }
}

for ($round = 1; $round -le $FullProcessRounds; $round++) {
    foreach ($binary in @($libBinary, $differentialBinary)) {
        $result = Invoke-IsolatedFilter -Binary $binary -Filter "" -Round $round
        $results.Add($result)
        if ($result.exit_code -ne 0) {
            $failures.Add("$($result.binary) full process round $round exited $($result.exit_code)")
        }
    }
}

$binaryManifest = @($libBinary, $differentialBinary) | ForEach-Object {
    $pdb = [System.IO.Path]::ChangeExtension($_.FullName, ".pdb")
    [pscustomobject]@{
        executable = $_.FullName
        executable_sha256 = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLowerInvariant()
        pdb = if (Test-Path $pdb) { $pdb } else { $null }
        pdb_sha256 = if (Test-Path $pdb) {
            (Get-FileHash -Algorithm SHA256 $pdb).Hash.ToLowerInvariant()
        } else {
            $null
        }
    }
}
$manifest = [pscustomobject]@{
    checkout_sha = (& git rev-parse HEAD).Trim()
    diagnostic_head_sha = $env:NXRT_DIAGNOSTIC_HEAD_SHA
    diagnostic_base_sha = $env:NXRT_DIAGNOSTIC_BASE_SHA
    github_run_id = $env:GITHUB_RUN_ID
    github_run_attempt = $env:GITHUB_RUN_ATTEMPT
    filtered_rounds = $Rounds
    full_process_rounds = $FullProcessRounds
    binaries = $binaryManifest
    invocations = $results
}
$manifestPath = Join-Path $artifactRoot "attribution-manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content $manifestPath
Write-Host "=== exact attribution manifest ==="
Get-Content $manifestPath | Write-Host
Write-Host "=== end exact attribution manifest ==="

$summary = Join-Path $artifactRoot "isolated-summary.txt"
if ($failures.Count -eq 0) {
    "All $(($Rounds * 3) + ($FullProcessRounds * 2)) process-isolated invocations passed." |
        Set-Content $summary
} else {
    @(
        "$($failures.Count) process-isolated invocation(s) failed:"
        $failures
    ) | Set-Content $summary
    $failures | Set-Content (Join-Path $artifactRoot "isolated-failure.txt")
}

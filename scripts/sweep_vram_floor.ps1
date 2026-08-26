param(
    [Parameter(Mandatory = $true)]
    [string]$Exe,

    [Parameter(Mandatory = $true)]
    [string]$ModelDir,

    [double[]]$LimitsGiB = @(2.34, 2.35, 2.36, 2.37, 2.38, 2.39, 2.40, 2.41, 2.42, 2.43, 2.44, 2.45, 2.46, 2.47, 2.48, 2.49, 2.50, 2.55, 2.56, 2.58, 2.59, 2.60),

    [string]$Prompt = "hi",

    [int]$MaxNewTokens = 32,

    [int]$Seed = 123
)

$ErrorActionPreference = "Stop"

$ledgerRefusalPattern = "memory ledger refused|ledger understates device use"
$results = @()

foreach ($limit in ($LimitsGiB | Sort-Object)) {
    $limitText = "{0:0.00}GiB" -f $limit
    $output = & $Exe --profile generate $ModelDir --prompt $Prompt --max-new-tokens $MaxNewTokens --greedy --seed $Seed --vram-limit $limitText 2>&1 | Out-String
    $exitCode = $LASTEXITCODE
    $ledgerRefused = $output -match $ledgerRefusalPattern
    $passed = ($exitCode -eq 0) -and -not $ledgerRefused
    $results += [pscustomobject]@{
        LimitGiB      = $limit
        Status        = if ($passed) { "pass" } elseif ($ledgerRefused) { "fail-ledger-refusal" } else { "fail" }
        ExitCode      = $exitCode
        LedgerRefused = $ledgerRefused
    }
}

$results | Format-Table -AutoSize

$seenPass = $false
$nonMonotonic = $false
foreach ($result in $results) {
    if ($result.Status -eq "pass") {
        $seenPass = $true
    } elseif ($seenPass) {
        $nonMonotonic = $true
    }
}

if ($nonMonotonic) {
    Write-Error "VRAM sweep is non-monotonic after scoring ledger-refusal warnings as failures; no floor may be quoted."
    exit 2
}

$floor = $results | Where-Object { $_.Status -eq "pass" } | Select-Object -First 1
if ($null -eq $floor) {
    Write-Error "No passing limit found."
    exit 1
}

Write-Host ("Monotonic floor: {0:0.00} GiB" -f $floor.LimitGiB)

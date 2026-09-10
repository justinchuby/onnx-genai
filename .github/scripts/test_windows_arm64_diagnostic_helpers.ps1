$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "windows_arm64_diagnostic_helpers.ps1")

$missingKey = Get-OptionalPropertyValue -InputObject $null -Name "Disabled"
if ($null -ne $missingKey) {
    throw "A missing registry key must return null."
}

$missingValue = Get-OptionalPropertyValue -InputObject ([pscustomobject]@{ Other = 1 }) -Name "Disabled"
if ($null -ne $missingValue) {
    throw "An existing registry key without Disabled must return null."
}

$presentValue = Get-OptionalPropertyValue -InputObject ([pscustomobject]@{ Disabled = 1 }) -Name "Disabled"
if ($presentValue -ne 1) {
    throw "An existing Disabled value must be returned."
}

$scratch = Join-Path $PSScriptRoot ".windows-arm64-diagnostic-helper-test-$PID"
try {
    New-Item -ItemType Directory -Force -Path $scratch | Out-Null
    $commandFile = Join-Path $scratch "commands with spaces.txt"
    $commandLines = Write-CdbCommandFile `
        -Path $commandFile `
        -SymbolDirectory "C:\symbols with spaces" `
        -SourceDirectory "C:\source with spaces" `
        -SymbolCache "C:\cache with spaces"
    $expectedCommands = @(
        ".sympath C:\symbols with spaces;srv*C:\cache with spaces*https://msdl.microsoft.com/download/symbols"
        ".srcpath C:\source with spaces"
        ".lines -e"
        ".reload /f"
        "!analyze -v"
        ".ecxr"
        "r"
        "ln @pc"
        "ub @pc L8"
        "u @pc L8"
        "kv"
        "lm"
        "q"
    )
    if (@(Compare-Object $expectedCommands $commandLines -SyncWindow 0).Count -ne 0) {
        throw "CDB command-file construction changed unexpectedly."
    }
    if ((Get-Content $commandFile).Count -ne $expectedCommands.Count) {
        throw "CDB commands must be written as separate command-file lines."
    }
    $arguments = Get-CdbArgumentList `
        -OutputPath "C:\output with spaces.txt" `
        -DumpPath "C:\dump with spaces.dmp" `
        -CommandFile $commandFile
    $expectedArguments = @(
        "-logo"
        "C:\output with spaces.txt"
        "-z"
        "C:\dump with spaces.dmp"
        "-cf"
        $commandFile
    )
    if (@(Compare-Object $expectedArguments $arguments -SyncWindow 0).Count -ne 0) {
        throw "CDB native argv construction changed unexpectedly."
    }
    if ($arguments -contains "-c" -or ($arguments -join "`n") -match "\.reload /f;") {
        throw "CDB commands must not be passed through a concatenated -c argument."
    }

    $scanRoot = Join-Path $scratch "scan"
    New-Item -ItemType Directory -Force -Path (Join-Path $scanRoot "symbol-cache") | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $scanRoot "binaries") | Out-Null
    Set-Content (Join-Path $scanRoot "analysis.txt") "text evidence"
    Set-Content (Join-Path $scanRoot "execution.json") "{}"
    Set-Content (Join-Path $scanRoot "runner.log") "log evidence"
    Set-Content (Join-Path $scanRoot "binaries\probe.exe") "binary"
    Set-Content (Join-Path $scanRoot "binaries\probe.pdb") "symbols"
    Set-Content (Join-Path $scanRoot "probe.dmp") "dump"
    Set-Content (Join-Path $scanRoot "probe.ilk") "linker state"
    Set-Content (Join-Path $scanRoot "symbol-cache\download.txt") "cached symbols"
    $scanFiles = @(Get-SecretScanFiles -Path $scanRoot)
    $scanNames = @($scanFiles | ForEach-Object {
        [System.IO.Path]::GetRelativePath($scanRoot, $_.FullName)
    })
    $expectedScanNames = @("analysis.txt", "execution.json", "runner.log")
    if (@(Compare-Object $expectedScanNames $scanNames -SyncWindow 0).Count -ne 0) {
        throw "Secret-scan allowlist selected unexpected files: $($scanNames -join ', ')."
    }
} finally {
    if (Test-Path $scratch) {
        Remove-Item -Recurse -Force $scratch
    }
}

Write-Host "Windows ARM64 diagnostic helper regressions passed."

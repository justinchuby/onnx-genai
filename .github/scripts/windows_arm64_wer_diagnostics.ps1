param(
    [Parameter(Mandatory)]
    [ValidateSet("SetupAndSelfTest", "ConfigureTarget", "WaitForTargetDump", "CheckForRealDump", "CollectFailure", "SecretScan", "Cleanup")]
    [string]$Action,
    [Parameter(Mandatory)]
    [string]$DiagnosticRoot,
    [string]$TestedRoot = "",
    [string]$TargetExecutable = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "windows_arm64_diagnostic_helpers.ps1")

$DiagnosticRoot = [System.IO.Path]::GetFullPath($DiagnosticRoot)
$werRegistryPath = "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps"
$werRegistryKey = "HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps"
$stateRoot = Join-Path $DiagnosticRoot "registry-state"
$artifactRoot = Join-Path $DiagnosticRoot "artifacts"
$dumpRoot = Join-Path $DiagnosticRoot "dumps"
$selfTestRoot = Join-Path $DiagnosticRoot "self-test"
$analysisRoot = Join-Path $artifactRoot "analysis"
$manifestRoot = Join-Path $artifactRoot "manifests"

function New-DiagnosticDirectories {
    foreach ($path in @($DiagnosticRoot, $stateRoot, $artifactRoot, $dumpRoot, $selfTestRoot, $analysisRoot, $manifestRoot)) {
        New-Item -ItemType Directory -Force -Path $path | Out-Null
    }
}

function Find-Cdb {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Windows Kits\10\Debuggers\arm64\cdb.exe",
        "$env:ProgramFiles\Windows Kits\10\Debuggers\arm64\cdb.exe"
    )
    $cdb = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($null -eq $cdb) {
        throw "ARM64 cdb.exe is required but was not found in the Windows SDK debugger directories."
    }
    return (Resolve-Path $cdb).Path
}

function Wait-ForDump {
    param(
        [Parameter(Mandatory)][string]$ExecutablePrefix,
        [int]$TimeoutSeconds = 90
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $lifecycleLog = Join-Path $manifestRoot "wer-lifecycle.log"
    do {
        $werFault = @(Get-Process -Name WerFault -ErrorAction SilentlyContinue)
        $werFaultPids = @($werFault | ForEach-Object Id)
        "$(Get-Date -Format o) phase=waiting werfault_pids=$($werFaultPids -join ',')" | Add-Content $lifecycleLog
        $dump = Get-ChildItem -Path $dumpRoot -Filter "$ExecutablePrefix*.dmp" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTimeUtc -Descending |
            Select-Object -First 1
        if ($null -ne $dump) {
            $lastLength = -1L
            $stableSamples = 0
            while ((Get-Date) -lt $deadline) {
                $werFault = @(Get-Process -Name WerFault -ErrorAction SilentlyContinue)
                $current = Get-Item -LiteralPath $dump.FullName -ErrorAction SilentlyContinue
                $werFaultPids = @($werFault | ForEach-Object Id)
                "$(Get-Date -Format o) phase=stabilizing werfault_pids=$($werFaultPids -join ',') dump='$($dump.Name)' size=$(if ($null -eq $current) { -1 } else { $current.Length })" |
                    Add-Content $lifecycleLog
                if ($null -ne $current -and $current.Length -gt 0 -and $current.Length -eq $lastLength) {
                    $stableSamples++
                } else {
                    $stableSamples = 0
                }

                function Configure-Target {
                    New-DiagnosticDirectories
                    if ([string]::IsNullOrWhiteSpace($TargetExecutable) -or !(Test-Path $TargetExecutable)) {
                        throw "ConfigureTarget requires the dynamically resolved test executable; received '$TargetExecutable'."
                    }
                    $executable = Get-Item -LiteralPath $TargetExecutable
                    $pdbPath = [System.IO.Path]::ChangeExtension($executable.FullName, ".pdb")
                    if (!(Test-Path $pdbPath)) {
                        throw "Dynamically resolved target '$($executable.FullName)' has no matching PDB '$pdbPath'."
                    }

                    $processKey = Join-Path $werRegistryPath $executable.Name
                    New-Item -Force $processKey | Out-Null
                    New-ItemProperty -Force $processKey DumpFolder -PropertyType ExpandString -Value $dumpRoot | Out-Null
                    New-ItemProperty -Force $processKey DumpType -PropertyType DWord -Value 2 | Out-Null
                    New-ItemProperty -Force $processKey DumpCount -PropertyType DWord -Value 1 | Out-Null

                    $binaryDirectory = New-Item -ItemType Directory -Force -Path (Join-Path $artifactRoot "binaries")
                    Copy-Item -Force $executable.FullName $binaryDirectory
                    Copy-Item -Force $pdbPath $binaryDirectory
                    [pscustomobject]@{
                        process_name = $executable.Name
                        process_registry_key = $processKey
                        executable = $executable.FullName
                        executable_sha256 = (Get-FileHash -Algorithm SHA256 $executable.FullName).Hash.ToLowerInvariant()
                        pdb = $pdbPath
                        pdb_sha256 = (Get-FileHash -Algorithm SHA256 $pdbPath).Hash.ToLowerInvariant()
                    } | ConvertTo-Json | Set-Content (Join-Path $manifestRoot "resolved-crash-target.json")

                    & reg.exe query ($werRegistryKey + "\" + $executable.Name) /s /reg:64 2>&1 |
                        Set-Content (Join-Path $manifestRoot "target-wer-registry.txt")
                }

                function Wait-ForTargetDump {
                    New-DiagnosticDirectories
                    if ([string]::IsNullOrWhiteSpace($TargetExecutable)) {
                        throw "WaitForTargetDump requires TargetExecutable."
                    }
                    $prefix = [System.IO.Path]::GetFileNameWithoutExtension($TargetExecutable)
                    $dump = Wait-ForDump -ExecutablePrefix $prefix -TimeoutSeconds 180
                    if ($null -eq $dump) {
                        throw "No stable WER dump appeared for '$([System.IO.Path]::GetFileName($TargetExecutable))' within 180 seconds."
                    }
                    "Stable target dump: $($dump.FullName) ($($dump.Length) bytes)" | Write-Host
                }
                if ($null -ne $current) {
                    $lastLength = $current.Length
                }
                if ($werFault.Count -eq 0 -and $stableSamples -ge 2) {
                    return $current
                }
                Start-Sleep -Seconds 2
            }
            throw "WER created '$($dump.FullName)', but WerFault did not finish and stabilize the dump within $TimeoutSeconds seconds."
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    return $null
}

function Invoke-CdbAnalysis {
    param(
        [Parameter(Mandatory)][string]$Cdb,
        [Parameter(Mandatory)][System.IO.FileInfo]$Dump,
        [Parameter(Mandatory)][string]$SymbolDirectory,
        [Parameter(Mandatory)][string]$SourceDirectory,
        [Parameter(Mandatory)][string]$OutputPath
    )

    $helpInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $helpInfo.FileName = $Cdb
    $helpInfo.UseShellExecute = $false
    $helpInfo.RedirectStandardOutput = $true
    $helpInfo.RedirectStandardError = $true
    $helpInfo.ArgumentList.Add("-?")
    $helpProcess = [System.Diagnostics.Process]::Start($helpInfo)
    $help = $helpProcess.StandardOutput.ReadToEnd() + $helpProcess.StandardError.ReadToEnd()
    $helpProcess.WaitForExit()
    if ($help -notmatch '(?im)(?:^|\s)(?:-|/)cf\b') {
        throw "Installed CDB '$Cdb' does not advertise the required -cf command-file switch."
    }

    $commandFile = [System.IO.Path]::ChangeExtension($OutputPath, ".commands.txt")
    Write-CdbCommandFile `
        -Path $commandFile `
        -SymbolDirectory $SymbolDirectory `
        -SourceDirectory $SourceDirectory `
        -SymbolCache (Join-Path $DiagnosticRoot "symbol-cache") | Out-Null
    $arguments = Get-CdbArgumentList `
        -OutputPath $OutputPath `
        -DumpPath $Dump.FullName `
        -CommandFile $commandFile
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $Cdb
    $startInfo.UseShellExecute = $false
    foreach ($argument in $arguments) {
        $startInfo.ArgumentList.Add($argument)
    }
    $process = [System.Diagnostics.Process]::Start($startInfo)
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        throw "cdb failed for '$($Dump.FullName)' with exit code $($process.ExitCode); command file='$commandFile'."
    }
}

function Save-HashManifest {
    param(
        [Parameter(Mandatory)][string]$Executable,
        [Parameter(Mandatory)][string]$Pdb,
        [Parameter(Mandatory)][string]$Path
    )

    if (!(Test-Path $Executable) -or !(Test-Path $Pdb)) {
        throw "Cannot hash the matching executable/PDB pair: executable='$Executable', pdb='$Pdb'."
    }
    [pscustomobject]@{
        executable = (Resolve-Path $Executable).Path
        executable_sha256 = (Get-FileHash -Algorithm SHA256 $Executable).Hash.ToLowerInvariant()
        pdb = (Resolve-Path $Pdb).Path
        pdb_sha256 = (Get-FileHash -Algorithm SHA256 $Pdb).Hash.ToLowerInvariant()
    } | ConvertTo-Json | Set-Content -Path $Path
}

function Assert-NoSecrets {
    param([Parameter(Mandatory)][string]$Path)

    $patterns = @(
        "github_pat_[A-Za-z0-9_]{20,}",
        "gh[pousr]_[A-Za-z0-9]{20,}",
        "AKIA[0-9A-Z]{16}",
        "-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----",
        ("Bea" + "rer [A-Za-z0-9._~+/=-]{20,}")
    )
    $findings = [System.Collections.Generic.List[string]]::new()
    $scanFiles = @(Get-SecretScanFiles -Path $Path)
    foreach ($file in $scanFiles) {
        $stream = [System.IO.File]::OpenRead($file.FullName)
        try {
            $buffer = New-Object byte[] (1024 * 1024)
            $overlap = New-Object byte[] 512
            $overlapLength = 0
            while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                $combined = New-Object byte[] ($overlapLength + $read)
                if ($overlapLength -gt 0) {
                    [Array]::Copy($overlap, 0, $combined, 0, $overlapLength)
                }
                [Array]::Copy($buffer, 0, $combined, $overlapLength, $read)
                $ascii = [System.Text.Encoding]::ASCII.GetString($combined)
                $unicode = [System.Text.Encoding]::Unicode.GetString($combined)
                foreach ($pattern in $patterns) {
                    if ($ascii -match $pattern -or $unicode -match $pattern) {
                        $findings.Add("$($file.FullName): matched secret pattern '$pattern'")
                    }
                }
                $overlapLength = [Math]::Min($overlap.Length, $combined.Length)
                [Array]::Copy($combined, $combined.Length - $overlapLength, $overlap, 0, $overlapLength)
            }
        } finally {
            $stream.Dispose()
        }
    }
    if ($findings.Count -ne 0) {
        $findings | Set-Content (Join-Path $DiagnosticRoot "SECRET_SCAN_FAILED.txt")
        throw "Secret scan rejected $($findings.Count) potential secret(s); artifacts must not be uploaded."
    }
    "Secret scan passed for $($scanFiles.Count) allowlisted text/script/manifest/log artifact files; binaries and symbol cache were excluded by construction." |
        Set-Content (Join-Path $manifestRoot "secret-scan.txt")
}

function Setup-AndSelfTest {
    New-DiagnosticDirectories

    $keyExisted = Test-Path $werRegistryPath
    [pscustomobject]@{ local_dumps_key_existed = $keyExisted } |
        ConvertTo-Json | Set-Content (Join-Path $stateRoot "registry-state.json")
    if ($keyExisted) {
        & reg.exe export $werRegistryKey (Join-Path $stateRoot "LocalDumps.reg") /y | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to back up the existing HKLM WER LocalDumps registry key."
        }
    }

    New-Item -Force $werRegistryPath | Out-Null
    New-ItemProperty -Force $werRegistryPath DumpFolder -PropertyType ExpandString -Value $dumpRoot | Out-Null
    New-ItemProperty -Force $werRegistryPath DumpType -PropertyType DWord -Value 2 | Out-Null
    New-ItemProperty -Force $werRegistryPath DumpCount -PropertyType DWord -Value 1 | Out-Null
    $werState = if (Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting") {
        Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows\Windows Error Reporting"
    } else {
        $null
    }
    $policyDisabled = Get-OptionalPropertyValue -InputObject $werState -Name "Disabled"
    $environmentReport = Join-Path $manifestRoot "wer-environment.txt"
    @(
        "dump_root=$dumpRoot"
        "dump_root_exists=$(Test-Path $dumpRoot)"
        "dump_root_acl=$((Get-Acl $dumpRoot).Sddl)"
        "wer_service=$((Get-Service WerSvc | Select-Object Name, Status, StartType | ConvertTo-Json -Compress))"
        "policy_disabled=$(if ($null -eq $policyDisabled) { '<missing>' } else { $policyDisabled })"
    ) | Set-Content $environmentReport
    "=== native registry view ===" | Add-Content $environmentReport
    & reg.exe query $werRegistryKey /s /reg:64 2>&1 | Add-Content $environmentReport
    "=== alternate registry view ===" | Add-Content $environmentReport
    & reg.exe query $werRegistryKey /s /reg:32 2>&1 | Add-Content $environmentReport

    $cdb = Find-Cdb
    Set-Content -Path (Join-Path $stateRoot "cdb-path.txt") -Value $cdb

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (!(Test-Path $vswhere)) {
        throw "vswhere.exe was not found; cannot compile the ARM64 access-violation self-test."
    }
    $installation = (& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath).Trim()
    if ([string]::IsNullOrWhiteSpace($installation)) {
        throw "Visual Studio ARM64 C++ tools were not found."
    }
    $vsDevCmd = Join-Path $installation "Common7\Tools\VsDevCmd.bat"
    $source = Join-Path $PSScriptRoot "windows_arm64_crash_self_test.c"
    $executable = Join-Path $selfTestRoot "nxrt_crash_self_test.exe"
    $pdb = Join-Path $selfTestRoot "nxrt_crash_self_test.pdb"
    $compileLog = Join-Path $selfTestRoot "compile.log"
    $compileCommand = "`"$vsDevCmd`" -no_logo -arch=arm64 -host_arch=arm64 && cl.exe /nologo /Zi /Od /W4 /WX `"$source`" /Fe:`"$executable`" /Fd:`"$pdb`" /link /debug:full /pdb:`"$pdb`""
    & cmd.exe /d /s /c $compileCommand 2>&1 | Tee-Object -FilePath $compileLog
    if ($LASTEXITCODE -ne 0) {
        throw "ARM64 access-violation self-test compilation failed with exit code $LASTEXITCODE."
    }

    $hashManifest = Join-Path $selfTestRoot "hashes-before.json"
    Save-HashManifest -Executable $executable -Pdb $pdb -Path $hashManifest
    $process = Start-Process -FilePath $executable -Wait -PassThru
    "exit_code=$($process.ExitCode)" | Set-Content (Join-Path $selfTestRoot "exit-code.txt")
    $unsignedExit = [System.BitConverter]::ToUInt32([System.BitConverter]::GetBytes([int32]$process.ExitCode), 0)
    if ($unsignedExit -ne 3221225477) {
        throw "Self-test child exited $($process.ExitCode), expected Windows access violation 0xC0000005."
    }

    $dump = Wait-ForDump -ExecutablePrefix "nxrt_crash_self_test"
    if ($null -eq $dump) {
        throw "HKLM WER LocalDumps did not capture the intentional access violation within 90 seconds."
    }
    Save-HashManifest -Executable $executable -Pdb $pdb -Path (Join-Path $selfTestRoot "hashes-after.json")
    if ((Get-FileHash $hashManifest).Hash -eq (Get-FileHash (Join-Path $selfTestRoot "hashes-after.json")).Hash) {
        # The manifests contain identical paths and hashes; byte identity is the desired proof.
    } else {
        throw "The self-test executable/PDB hash manifest changed between execution and dump analysis."
    }

    $analysis = Join-Path $selfTestRoot "cdb.txt"
    Invoke-CdbAnalysis `
        -Cdb $cdb `
        -Dump $dump `
        -SymbolDirectory $selfTestRoot `
        -SourceDirectory $PSScriptRoot `
        -OutputPath $analysis
    $analysisText = Get-Content -Raw $analysis
    $assertions = [ordered]@{
        access_violation = '(?i)c0000005'
        matching_symbolized_frame = '(?i)nxrt_crash_self_test!intentional_access_violation_probe'
        matching_source = '(?i)windows_arm64_crash_self_test\.c'
    }
    foreach ($assertion in $assertions.GetEnumerator()) {
        if ($analysisText -notmatch $assertion.Value) {
            throw "CDB self-test analysis failed '$($assertion.Key)' proof; expected pattern '$($assertion.Value)'."
        }
    }
    [pscustomobject]@{
        dump = $dump.Name
        dump_sha256 = (Get-FileHash -Algorithm SHA256 $dump.FullName).Hash.ToLowerInvariant()
        executable = [System.IO.Path]::GetFileName($executable)
        executable_sha256 = (Get-FileHash -Algorithm SHA256 $executable).Hash.ToLowerInvariant()
        pdb = [System.IO.Path]::GetFileName($pdb)
        pdb_sha256 = (Get-FileHash -Algorithm SHA256 $pdb).Hash.ToLowerInvariant()
        expected_exception = "0xc0000005"
        required_faulting_frame = "intentional_access_violation_probe"
        required_symbolized_frame = "nxrt_crash_self_test!intentional_access_violation_probe"
        required_source = "windows_arm64_crash_self_test.c"
        assertions = [pscustomobject]@{
            access_violation = $true
            matching_symbolized_frame = $true
            matching_source = $true
        }
        cdb = $cdb
        verified = $true
    } | ConvertTo-Json | Set-Content (Join-Path $selfTestRoot "proof.json")

    Remove-Item -Force $dump.FullName
    Copy-Item -Recurse -Force $selfTestRoot (Join-Path $artifactRoot "self-test-proof")
    Assert-NoSecrets -Path $artifactRoot
    "Self-test passed: HKLM WER captured a full ARM64 dump and CDB named intentional_access_violation_probe." |
        Write-Host
}

function Check-ForRealDump {
    New-DiagnosticDirectories
    $dumps = @(Get-ChildItem -Path $dumpRoot -Filter *.dmp -ErrorAction SilentlyContinue)
    if ($dumps.Count -gt 1) {
        throw "Dump bound violated: expected at most one real dump, found $($dumps.Count)."
    }
    if ($dumps.Count -eq 1) {
        "real_dump=true" >> $env:GITHUB_OUTPUT
        "Real crash dump captured: $($dumps[0].FullName)" | Write-Host
    } else {
        "real_dump=false" >> $env:GITHUB_OUTPUT
    }
}

function Collect-Failure {
    New-DiagnosticDirectories
    if ([string]::IsNullOrWhiteSpace($TestedRoot) -or !(Test-Path $TestedRoot)) {
        throw "CollectFailure requires an existing TestedRoot; received '$TestedRoot'."
    }
    if (Test-Path $selfTestRoot) {
        $selfTestArtifact = Join-Path $artifactRoot "self-test-proof"
        if (Test-Path $selfTestArtifact) {
            Remove-Item -Recurse -Force $selfTestArtifact
        }
        Copy-Item -Recurse -Force $selfTestRoot $selfTestArtifact
    }

    $broadLog = Join-Path $artifactRoot "logs\broad-tests.log"
    $targetMatch = $null
    if (Test-Path $broadLog) {
        $targetMatch = [regex]::Matches(
            (Get-Content -Raw $broadLog),
            '(?im)process didn''t exit successfully:\s*`([^`]*native_vs_mlas_differential-[^`\\/:]+\.exe)(?:\s[^`]*)?`'
        ) | Select-Object -Last 1
    }
    if ($null -ne $targetMatch) {
        $reportedExecutable = $targetMatch.Groups[1].Value
        $executableName = [System.IO.Path]::GetFileName($reportedExecutable)
        $executable = Get-ChildItem -Path (Join-Path $TestedRoot "target\debug\deps") -Filter $executableName -File |
            Select-Object -First 1
        if ($null -eq $executable) {
            throw "Cargo reported crashing target '$reportedExecutable', but '$executableName' was not found under target/debug/deps."
        }
        $pdbPath = [System.IO.Path]::ChangeExtension($executable.FullName, ".pdb")
        if (!(Test-Path $pdbPath)) {
            throw "Cargo reported crashing target '$reportedExecutable', but dynamically matching PDB '$pdbPath' was not found."
        }
        [pscustomobject]@{
            reported_executable = $reportedExecutable
            resolved_executable = $executable.FullName
            resolved_pdb = $pdbPath
            executable_sha256 = (Get-FileHash -Algorithm SHA256 $executable.FullName).Hash.ToLowerInvariant()
            pdb_sha256 = (Get-FileHash -Algorithm SHA256 $pdbPath).Hash.ToLowerInvariant()
        } | ConvertTo-Json | Set-Content (Join-Path $manifestRoot "resolved-crash-target.json")

        $dumpPrefix = [System.IO.Path]::GetFileNameWithoutExtension($executableName)
        $dump = Wait-ForDump -ExecutablePrefix $dumpPrefix -TimeoutSeconds 120
        if ($null -eq $dump) {
            throw "Cargo reported access violation in '$executableName', but no matching WER dump stabilized within 120 seconds."
        }
    }

    $dumps = @(Get-ChildItem -Path $dumpRoot -Filter *.dmp -ErrorAction SilentlyContinue)
    if ($dumps.Count -gt 1) {
        throw "Dump bound violated during collection: found $($dumps.Count) real dumps."
    }
    $files = [System.Collections.Generic.List[object]]::new()
    foreach ($dump in $dumps) {
        if ($dump.Name -like "nxrt_crash_self_test*") {
            "The intentional self-test dump was not retained after a failed self-test; see self-test logs." |
                Set-Content (Join-Path $analysisRoot "self-test-dump-removed.txt")
            Remove-Item -Force $dump.FullName
            continue
        }
        $cdb = Find-Cdb
        $analysis = Join-Path $analysisRoot "$($dump.BaseName)-cdb.txt"
        Invoke-CdbAnalysis `
            -Cdb $cdb `
            -Dump $dump `
            -SymbolDirectory (Join-Path $TestedRoot "target\debug\deps") `
            -SourceDirectory $TestedRoot `
            -OutputPath $analysis

        $executableName = $dump.BaseName -replace '\.\d+$', ''
        if (!$executableName.EndsWith(".exe", [System.StringComparison]::OrdinalIgnoreCase)) {
            $executableName = "$executableName.exe"
        }
        $executable = Get-ChildItem -Path (Join-Path $TestedRoot "target\debug\deps") -Filter $executableName -File |
            Select-Object -First 1
        if ($null -eq $executable) {
            throw "A dump was captured for '$executableName', but its exact executable was not found under target/debug/deps."
        }
        $pdbPath = [System.IO.Path]::ChangeExtension($executable.FullName, ".pdb")
        if (!(Test-Path $pdbPath)) {
            throw "A dump was captured for '$executableName', but matching PDB '$pdbPath' was not found."
        }
        $binaryDirectory = New-Item -ItemType Directory -Force -Path (Join-Path $artifactRoot "binaries")
        Copy-Item -Force $executable.FullName $binaryDirectory
        Copy-Item -Force $pdbPath $binaryDirectory
        $files.Add([pscustomobject]@{
            dump = $dump.Name
            dump_sha256 = (Get-FileHash -Algorithm SHA256 $dump.FullName).Hash.ToLowerInvariant()
            executable = $executable.Name
            executable_sha256 = (Get-FileHash -Algorithm SHA256 $executable.FullName).Hash.ToLowerInvariant()
            pdb = [System.IO.Path]::GetFileName($pdbPath)
            pdb_sha256 = (Get-FileHash -Algorithm SHA256 $pdbPath).Hash.ToLowerInvariant()
            analysis = [System.IO.Path]::GetFileName($analysis)
        })
        Copy-Item -Force $dump.FullName $artifactRoot
    }
    $files | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $manifestRoot "captured-files.json")
    Assert-NoSecrets -Path $artifactRoot
}

function Restore-Registry {
    if (!(Test-Path $stateRoot)) {
        Write-Warning "No registry backup directory exists; cleanup has no recorded state to restore."
        return
    }
    if (Test-Path $werRegistryPath) {
        Remove-Item -Recurse -Force $werRegistryPath
    }
    $backup = Join-Path $stateRoot "LocalDumps.reg"
    if (Test-Path $backup) {
        & reg.exe import $backup | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to restore the original HKLM WER LocalDumps registry key."
        }
    }
}

switch ($Action) {
    "SetupAndSelfTest" { Setup-AndSelfTest }
    "ConfigureTarget" { Configure-Target }
    "WaitForTargetDump" { Wait-ForTargetDump }
    "CheckForRealDump" { Check-ForRealDump }
    "CollectFailure" { Collect-Failure }
    "SecretScan" { New-DiagnosticDirectories; Assert-NoSecrets -Path $artifactRoot }
    "Cleanup" { Restore-Registry }
}

function Get-OptionalPropertyValue {
    param(
        [AllowNull()][object]$InputObject,
        [Parameter(Mandatory)][string]$Name
    )

    if ($null -eq $InputObject) {
        return $null
    }
    $property = $InputObject.PSObject.Properties[$Name]
    if ($null -eq $property) {
        return $null
    }
    return $property.Value
}

function Get-LocalDumpProcessSubkey {
    param(
        [Parameter(Mandatory)][string]$LocalDumpsPath,
        [Parameter(Mandatory)][string]$ExecutablePath
    )

    $executableName = [System.IO.Path]::GetFileName($ExecutablePath.Replace('\', '/'))
    if ([string]::IsNullOrWhiteSpace($executableName) -or
        !$executableName.EndsWith(".exe", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "LocalDumps process configuration requires an executable path ending in .exe; received '$ExecutablePath'."
    }
    return "$($LocalDumpsPath.TrimEnd('\'))\$executableName"
}

function Update-DumpStabilityState {
    param(
        [long]$PreviousLength,
        [int]$StableSamples,
        [AllowNull()][object]$CurrentLength
    )

    if ($null -eq $CurrentLength -or [long]$CurrentLength -le 0) {
        return [pscustomobject]@{ Length = $PreviousLength; StableSamples = 0 }
    }
    $length = [long]$CurrentLength
    $nextSamples = if ($length -eq $PreviousLength) {
        $StableSamples + 1
    } else {
        0
    }
    return [pscustomobject]@{
        Length = $length
        StableSamples = $nextSamples
    }
}

function Get-WindowsProcessExitClassification {
    param([Parameter(Mandatory)][int]$ExitCode)

    $unsigned = [System.BitConverter]::ToUInt32([System.BitConverter]::GetBytes([int32]$ExitCode), 0)
    if ($ExitCode -eq 0) {
        return [pscustomobject]@{ Kind = "success"; IsAccessViolation = $false; UnsignedExitCode = $unsigned }
    }
    if ($ExitCode -eq 101) {
        return [pscustomobject]@{ Kind = "rust-test-failure"; IsAccessViolation = $false; UnsignedExitCode = $unsigned }
    }
    if ($ExitCode -eq 139) {
        return [pscustomobject]@{ Kind = "shell-mapped-access-violation"; IsAccessViolation = $true; UnsignedExitCode = $unsigned }
    }
    if ($unsigned -eq 3221225477) {
        return [pscustomobject]@{ Kind = "windows-access-violation"; IsAccessViolation = $true; UnsignedExitCode = $unsigned }
    }
    return [pscustomobject]@{ Kind = "other-failure"; IsAccessViolation = $false; UnsignedExitCode = $unsigned }
}

function New-CdbCommandLines {
    param(
        [Parameter(Mandatory)][string]$SymbolDirectory,
        [Parameter(Mandatory)][string]$SourceDirectory,
        [Parameter(Mandatory)][string]$SymbolCache
    )

    return @(
        ".sympath $SymbolDirectory;srv*$SymbolCache*https://msdl.microsoft.com/download/symbols"
        ".srcpath $SourceDirectory"
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
}

function Write-CdbCommandFile {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$SymbolDirectory,
        [Parameter(Mandatory)][string]$SourceDirectory,
        [Parameter(Mandatory)][string]$SymbolCache
    )

    $lines = New-CdbCommandLines `
        -SymbolDirectory $SymbolDirectory `
        -SourceDirectory $SourceDirectory `
        -SymbolCache $SymbolCache
    Set-Content -Path $Path -Value $lines -Encoding ascii
    return $lines
}

function Get-CdbArgumentList {
    param(
        [Parameter(Mandatory)][string]$OutputPath,
        [Parameter(Mandatory)][string]$DumpPath,
        [Parameter(Mandatory)][string]$CommandFile
    )

    return @("-logo", $OutputPath, "-z", $DumpPath, "-cf", $CommandFile)
}

function Get-SecretScanFiles {
    param([Parameter(Mandatory)][string]$Path)

    $textExtensions = @(".json", ".log", ".md", ".ps1", ".txt", ".yaml", ".yml")
    return @(Get-ChildItem -Path $Path -File -Recurse | Where-Object {
        $relativePath = [System.IO.Path]::GetRelativePath($Path, $_.FullName)
        $segments = $relativePath -split '[\\/]'
        $segments -notcontains "symbol-cache" -and
            $textExtensions -contains $_.Extension.ToLowerInvariant()
    })
}

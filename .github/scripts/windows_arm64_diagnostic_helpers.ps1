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

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

Write-Host "Optional registry property regression passed."

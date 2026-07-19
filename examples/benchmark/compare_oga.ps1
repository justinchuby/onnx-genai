<#
.SYNOPSIS
    Matched decode-throughput comparison: onnx-genai (profile_decode) vs
    onnxruntime-genai (oga_bench.py), greedy, over one or more model dirs.

.DESCRIPTION
    Both harnesses chat-template the same prompt, so the two runtimes decode the
    SAME greedy sequence (verified byte-identical). Runs are alternated in order
    (us-first / oga-first) with cooldowns to reduce thermal bias on laptop GPUs.

.PARAMETER ModelDirs
    One or more directories each containing a genai_config.json (the exported
    model). Example: -ModelDirs (Get-ChildItem -Directory .\models\*).FullName

.PARAMETER Tokens
    New tokens to decode per run (default 200).

.PARAMETER Prompt
    Prompt text (default: a short instruct prompt).

.EXAMPLE
    .\compare_oga.ps1 -ModelDirs "C:\models\phi-4-mini\v5","C:\models\qwen3-0.6b\v2"

.NOTES
    Requires:
      * profile_decode built:  cargo build --release -p onnx-genai-bench --features cuda-ort --bin profile_decode
      * oga installed:         pip install onnxruntime-genai-cuda
      * $env:ORT_ROOT and the CUDA env set (see repo docs).
#>
param(
    [Parameter(Mandatory = $true)] [string[]] $ModelDirs,
    [int] $Tokens = 200,
    [string] $Prompt = "Explain the theory of relativity in simple terms.",
    [int] $Warmups = 1,
    [int] $Runs = 2,
    [int] $CooldownSeconds = 6
)

$env:OGA_WARMUPS = "$Warmups"
$env:OGA_RUNS = "$Runs"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ogaBench = Join-Path $scriptDir "oga_bench.py"
$profileDecode = ".\target\release\profile_decode.exe"

function Cool { if ($CooldownSeconds -gt 0) { Start-Sleep -Seconds $CooldownSeconds } }

function Run-Us($dir) {
    $out = & $profileDecode --model $dir --tokens $Tokens --warmups $Warmups --runs $Runs --temperature 0.0 --seed 17 --prompt $Prompt 2>&1 | Out-String
    if ($out -match "->\s*([0-9.]+)\s*tok/s") { [double]$Matches[1] } else { -1 }
}
function Run-Oga($dir) {
    $out = & python $ogaBench $dir $Prompt $Tokens 2>&1 | Out-String
    if ($out -match "->\s*([0-9.]+)\s*tok/s") { [double]$Matches[1] } else { -1 }
}

$i = 0
$rows = @()
foreach ($dir in $ModelDirs) {
    $label = Split-Path -Leaf $dir
    if (-not (Test-Path (Join-Path $dir "genai_config.json"))) {
        Write-Output ("{0,-24} SKIP (no genai_config.json)" -f $label); $i++; continue
    }
    $usFirst = ($i % 2 -eq 0)
    Cool
    if ($usFirst) { $us = Run-Us $dir } else { $oga = Run-Oga $dir }
    Cool
    if ($usFirst) { $oga = Run-Oga $dir } else { $us = Run-Us $dir }
    $ratio = if ($oga -gt 0) { [math]::Round($us / $oga, 2) } else { "n/a" }
    $rows += [pscustomobject]@{ Model = $label; Us = [math]::Round($us, 1); Oga = [math]::Round($oga, 1); "Us/Oga" = $ratio }
    Write-Output ("{0,-24} us={1,8} oga={2,8} ratio={3}" -f $label, [math]::Round($us, 1), [math]::Round($oga, 1), $ratio)
    $i++
}

Write-Output ""
$rows | Format-Table -AutoSize | Out-String | Write-Output

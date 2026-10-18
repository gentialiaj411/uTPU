<#
Run the two Boolean full-top routes plus the complete vector OOC route.
Vivado user-state is redirected into build/ because a sandboxed invocation
otherwise fails before TCL execution while loading per-user features.
#>
[CmdletBinding()]
param(
    [string]$Vivado = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$State = Join-Path $Root "build\xilinx_env"
New-Item -ItemType Directory -Force -Path $State | Out-Null
$env:APPDATA = $State
$env:LOCALAPPDATA = $State
$env:HOME = $State
$env:XILINX_TCLSTORE_USERAREA = $State
if (-not $Vivado) {
    $Command = Get-Command vivado -ErrorAction SilentlyContinue
    if (-not $Command) { throw "Vivado is not on PATH; rerun with -Vivado <path-to-vivado.bat>." }
    $Vivado = $Command.Source
}

$Common = @("part", "xc7s50csga324-1", "top", "top_board", "ARRAY_SIZE", "8",
            "COMPUTE_DATA_WIDTH", "8", "MAX_BATCH_COUNT", "4", "clock_period", "10",
            "CORE_CLK_HZ", "50000000", "PROG_DEPTH", "65536", "BUFFER_SIZE", "32768",
            "WEIGHT_REGION_WORDS", "27648")
foreach ($Mode in 0,1) {
    $Prefix = "boolean_vec$Mode"
    $Log = Join-Path $Root "build\${Prefix}_vivado.log"
    & $Vivado -mode batch -source (Join-Path $Root "scripts\synth_design_space_point.tcl") `
        -tclargs @($Common + @("VECTOR_UNIT_EN", "$Mode", "report_prefix", $Prefix, "proj_name", $Prefix)) `
        2>&1 | Tee-Object -FilePath $Log
    if ($LASTEXITCODE) { throw "Vivado full-top VEC=$Mode failed; see $Log" }
}
$OocLog = Join-Path $Root "build\vector_ooc_vivado.log"
& $Vivado -mode batch -source (Join-Path $Root "scripts\synth_vector_ooc.tcl") 2>&1 | Tee-Object -FilePath $OocLog
if ($LASTEXITCODE) { throw "Vivado vector OOC failed; see $OocLog" }
Push-Location $Root
try { python tools/parse_boolean_synth.py } finally { Pop-Location }

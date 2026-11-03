# Regenerate every gate-required decode/board artifact from a clean checkout, then run the strict gate.
# Sequential, each step logged to build/regen_logs/<name>.log, stops on first failure. Needs iverilog (C:\iverilog\bin),
# Vivado 2025.2 and models/ (see docs/REPRODUCE_DECODE.md). Roughly 3-4 hours, most of it fault injection and the 15M e2e.
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root
$logDir = Join-Path $root 'build/regen_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Logged([string]$Name, [string[]]$CommandArgs) {
    $log = Join-Path $logDir ($Name + '.log')
    $raw = Join-Path $logDir ($Name + '.output.tmp')
    "[$(Get-Date -Format o)] START python $($CommandArgs -join ' ')" | Set-Content -LiteralPath $log
    Remove-Item -LiteralPath $raw -Force -ErrorAction SilentlyContinue
    & python @CommandArgs *> $raw
    $rc = $LASTEXITCODE
    if (Test-Path -LiteralPath $raw) { Get-Content -LiteralPath $raw | Add-Content -LiteralPath $log; Remove-Item -LiteralPath $raw -Force }
    "[$(Get-Date -Format o)] EXIT $rc" | Add-Content -LiteralPath $log
    Write-Output "$(Get-Date -Format HH:mm) $Name -> $rc"
    if ($rc -ne 0) { throw "$Name failed with exit code $rc. See $log" }
}

$env:UTPU_SIM = 'icarus'

# 1. generated inputs: LUTs, golden captures, flash images, 15M wrappers
Invoke-Logged 'gen_luts' @('tools/decode/gen_luts.py')
foreach ($g in @('gemv','gemv15','act_quant','rms','vec','attn','attn15')) { Invoke-Logged "golden_$g" @("tools/decode/golden_$g.py") }
Invoke-Logged 'pack_260k' @('tools/decode/pack_decode_image.py','--model','260k')
Invoke-Logged 'pack_15m' @('tools/decode/pack_decode_image.py','--model','15m')
Invoke-Logged 'wrap_15m' @('tools/decode/gen_15m_wrap.py')
Invoke-Logged 'wrap_qspi' @('tools/decode/gen_qspi_wrap.py')

# 2. unit differential runners
Invoke-Logged 'unit_gemv' @('tools/decode/run_gemv_rtl.py')
Invoke-Logged 'unit_gemv15' @('tools/decode/run_gemv_rtl.py','--golden','build/decode/golden_gemv15','--output','bench/results/decode_gemv15_rtl.json')
foreach ($u in @('act_quant','rms','vec','attn','attn15','qspi')) { Invoke-Logged "unit_$u" @("tools/decode/run_$($u)_rtl.py") }

# 3. end-to-end decode (260K configs, then the board UART TB, then 15M)
Invoke-Logged 'e2e_260k_kv8_p1' @('tools/decode/run_decode_rtl.py','--model','260k','--kv-bits','8','--period','1','--prompts','1','--gen','8','--output','bench/results/decode_e2e_rtl_kv8_p1.json')
Invoke-Logged 'e2e_260k_kv4_p1' @('tools/decode/run_decode_rtl.py','--model','260k','--kv-bits','4','--period','1','--prompts','1','--gen','8','--output','bench/results/decode_e2e_rtl_kv4_p1.json')
Invoke-Logged 'e2e_260k_kv8_p2' @('tools/decode/run_decode_rtl.py','--model','260k','--kv-bits','8','--period','2','--gen','8','--output','bench/results/decode_e2e_rtl_kv8_p2.json')
Invoke-Logged 'e2e_260k_kv8_p4' @('tools/decode/run_decode_rtl.py','--model','260k','--kv-bits','8','--period','4','--gen','8','--output','bench/results/decode_e2e_rtl_kv8_p4.json')
Invoke-Logged 'e2e_260k_qspi' @('tools/decode/run_decode_rtl.py','--model','260k','--kv-bits','8','--period','1','--prompts','1','--gen','8','--flash','qspi','--output','bench/results/decode_e2e_rtl_kv8_p1_qspi.json')
Invoke-Logged 'uart_rtl' @('tools/board/run_uart_rtl.py')
Invoke-Logged 'e2e_15m_kv4_p1' @('tools/decode/run_decode_rtl.py','--model','15m','--kv-bits','4','--gen','8','--output','bench/results/decode_e2e_rtl_15m_kv4_p1.json')

# 4. fault injection, every unit
foreach ($u in @('gemv','actq','rms','vec','attn','attn15','top','qspi','board')) { Invoke-Logged "fault_$u" @('tools/decode/fault_inject.py',$u) }

# 5. Vivado: out-of-context synthesis, full board implementation, pin check
$D = 'rtl/decode/'
$top = @("${D}decode_pkg.sv","${D}seq_mul.sv","${D}mul_pipe.sv","${D}div_const_pipe.sv","${D}gemv_unit.sv","${D}act_quant.sv","${D}rmsnorm.sv","${D}vec_alu.sv","${D}attention.sv","${D}top_decode.sv")
$synthJobs = @(
    @('gemv_unit','decode_gemv_unit',"${D}gemv_unit.sv"),
    @('act_quant','act_quant',"${D}act_quant.sv"),
    @('rmsnorm','rmsnorm',"${D}decode_pkg.sv","${D}seq_mul.sv","${D}rmsnorm.sv"),
    @('vec_alu','vec_alu',"${D}decode_pkg.sv","${D}vec_alu.sv"),
    @('attention','attention',"${D}decode_pkg.sv","${D}mul_pipe.sv","${D}div_const_pipe.sv","${D}attention.sv"),
    (@('top_decode_260k','top_decode') + $top),
    (@('top_decode_15m','top_decode_15m') + $top + @('build/decode/top_decode_15m_wrap.sv')),
    @('qspi_reader','qspi_reader',"${D}qspi_reader.sv"),
    (@('top_decode_15m_qspi','top_decode_15m_qspi') + $top + @("${D}qspi_reader.sv",'build/decode/top_decode_15m_qspi_wrap.sv'))
)
foreach ($j in $synthJobs) { Invoke-Logged "synth_$($j[0])" (@('tools/decode/synth_ooc.py') + [string[]]$j) }
Invoke-Logged 'boolean_top_impl' @('tools/board/impl_top.py')
Invoke-Logged 'boolean_pin_check' @('tools/board/pin_check.py')

# 6. strict gate
Invoke-Logged 'strict_gate' @('tools/decode_gate.py')
Write-Output "REGEN_ALL complete: strict gate passed. Logs: $logDir"

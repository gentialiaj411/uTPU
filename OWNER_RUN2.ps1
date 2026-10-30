$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root
$logDir = Join-Path $root 'build/owner_logs2'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Invoke-Logged([string]$Name, [string]$Exe, [string[]]$CommandArgs) {
    $log = Join-Path $logDir ($Name + '.log')
    $raw = Join-Path $logDir ($Name + '.output.tmp')
    "[$(Get-Date -Format o)] START $Exe $($CommandArgs -join ' ')" | Set-Content -LiteralPath $log
    Remove-Item -LiteralPath $raw -Force -ErrorAction SilentlyContinue
    & $Exe @CommandArgs *> $raw
    $rc = $LASTEXITCODE
    if (Test-Path -LiteralPath $raw) { Get-Content -LiteralPath $raw | Add-Content -LiteralPath $log; Remove-Item -LiteralPath $raw -Force }
    "[$(Get-Date -Format o)] EXIT $rc" | Add-Content -LiteralPath $log
    if ($rc -ne 0) { throw "$Name failed with exit code $rc. See $log" }
}

$py = 'python'
Invoke-Logged -Name 'pack_15m' -Exe $py -CommandArgs @('tools/decode/pack_decode_image.py','--model','15m')
Invoke-Logged -Name 'wrap_15m' -Exe $py -CommandArgs @('tools/decode/gen_15m_wrap.py')
Invoke-Logged -Name 'wrap_qspi' -Exe $py -CommandArgs @('tools/decode/gen_qspi_wrap.py')
# Finish timeboxed fault-provenance regeneration before physical closure.
$env:UTPU_SIM = 'icarus'
foreach ($unit in @('qspi','rms','vec','attn','attn15')) {
    Invoke-Logged -Name ('fault_' + $unit) -Exe $py -CommandArgs @('tools/decode/fault_inject.py',$unit)
}
$synthJobs = @(
    @('gemv_unit','decode_gemv_unit','rtl/decode/gemv_unit.sv'),
    @('act_quant','act_quant','rtl/decode/act_quant.sv'),
    @('rmsnorm','rmsnorm','rtl/decode/decode_pkg.sv','rtl/decode/seq_mul.sv','rtl/decode/rmsnorm.sv'),
    @('vec_alu','vec_alu','rtl/decode/decode_pkg.sv','rtl/decode/vec_alu.sv'),
    @('attention','attention','rtl/decode/decode_pkg.sv','rtl/decode/mul_pipe.sv','rtl/decode/div_const_pipe.sv','rtl/decode/attention.sv'),
    @('top_decode_260k','top_decode','rtl/decode/decode_pkg.sv','rtl/decode/seq_mul.sv','rtl/decode/mul_pipe.sv','rtl/decode/div_const_pipe.sv','rtl/decode/gemv_unit.sv','rtl/decode/act_quant.sv','rtl/decode/rmsnorm.sv','rtl/decode/vec_alu.sv','rtl/decode/attention.sv','rtl/decode/top_decode.sv'),
    @('top_decode_15m','top_decode_15m','rtl/decode/decode_pkg.sv','rtl/decode/seq_mul.sv','rtl/decode/mul_pipe.sv','rtl/decode/div_const_pipe.sv','rtl/decode/gemv_unit.sv','rtl/decode/act_quant.sv','rtl/decode/rmsnorm.sv','rtl/decode/vec_alu.sv','rtl/decode/attention.sv','rtl/decode/top_decode.sv','build/decode/top_decode_15m_wrap.sv'),
    @('qspi_reader','qspi_reader','rtl/decode/qspi_reader.sv'),
    @('top_decode_15m_qspi','top_decode_15m_qspi','rtl/decode/decode_pkg.sv','rtl/decode/seq_mul.sv','rtl/decode/mul_pipe.sv','rtl/decode/div_const_pipe.sv','rtl/decode/gemv_unit.sv','rtl/decode/act_quant.sv','rtl/decode/rmsnorm.sv','rtl/decode/vec_alu.sv','rtl/decode/attention.sv','rtl/decode/top_decode.sv','rtl/decode/qspi_reader.sv','build/decode/top_decode_15m_qspi_wrap.sv')
)
foreach ($j in $synthJobs) { $cmdline=@('tools/decode/synth_ooc.py') + [string[]]$j; Invoke-Logged -Name "synth_$($j[0])" -Exe $py -CommandArgs $cmdline }
Invoke-Logged -Name 'boolean_top_impl' -Exe $py -CommandArgs @('tools/board/impl_top.py')
Invoke-Logged -Name 'boolean_pin_check' -Exe $py -CommandArgs @('tools/board/pin_check.py')
Invoke-Logged -Name 'strict_gate' -Exe $py -CommandArgs @('tools/decode_gate.py')
Write-Output 'OWNER_RUN2 complete'

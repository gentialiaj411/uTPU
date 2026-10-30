from pathlib import Path
s=Path('tools/decode/tb/qspi_flash_model.sv').read_text();assert s.count("flash_o = 4'bz; flash_oe = 4'h0;")==2
s=s.replace("flash_o = 4'bz; flash_oe = 4'h0;", "flash_o = 4'h0; flash_oe = 4'h0; // inactive data is masked by explicit output enable")
Path('sprint/data/qspi_flash_model_idle_data.sv').write_text(s)

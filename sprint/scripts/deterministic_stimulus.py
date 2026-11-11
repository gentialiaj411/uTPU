from pathlib import Path
p=Path('tools/decode/run_qspi_rtl.py');s=p.read_text()
s=s.replace('for (int k = 0; k < 150; k++) begin sp(ra(), 1 + rnd(12)); cyc(rnd(30)); end', 'for (int k = 0; k < 150; k++) begin logic [31:0] address; int count; address = ra(); count = 1 + rnd(12); sp(address, count); cyc(rnd(30)); end')
s=s.replace('seek(ra()); take(4 + rnd(10), 100 - 20 * rnd(4));', 'int count, pct; seek(ra()); count = 4 + rnd(10); pct = 100 - 20 * rnd(4); take(count, pct);')
s=s.replace('sp(ra(), 3 + rnd(5)); st_ready = 1;', 'logic [31:0] address; int count; address = ra(); count = 3 + rnd(5); sp(address, count); st_ready = 1;')
p.write_text(s)
p=Path('tools/board/run_uart_rtl.py');s=p.read_text();start=s.index('TB = ');end=s.index('def ',start);tb=s[start:end];tb=tb.replace('@(posedge clk);', '@(posedge clk); #0.1;');tb=tb.replace('while (rd >= wr) @(posedge clk); #0.1;', 'while (rd >= wr) begin @(posedge clk); #0.1; end');s=s[:start]+tb+s[end:];p.write_text(s)

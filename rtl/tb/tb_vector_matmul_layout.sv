`timescale 1ns/1ps
module tb_vector_matmul_layout;
    logic clk=0, rst=1, start=0, causal=1;
    logic [3:0] subop=4;
    logic [15:0] dst=20, src0=0, param=100, length=4, rdata;
    logic busy, done, error, we, re, store_en;
    logic [15:0] addr, wdata, mem [0:1023];
    integer cycles=0;
    always #5 clk=~clk;
    always_ff @(posedge clk) begin
        if(re) rdata<=mem[addr];
        if(we) mem[addr]<=wdata;
    end
    vector_sequencer #(.ADDRESS_SIZE(16),.BUFFER_WORDS(1024)) dut(
        .clk(clk),.rst(rst),.start(start),.subop(subop),.causal(causal),.dst(dst),.src0(src0),.param(param),.length(length),
        .buf_rdata(rdata),.busy(busy),.done(done),.error(error),.buf_we(we),.buf_re(re),.buf_store_en(store_en),.buf_addr(addr),.buf_wdata(wdata));
    initial begin
        integer i; for(i=0;i<1024;i=i+1) mem[i]=0;
        // A=[[1,2],[3,4]], B=[[1,2],[3,4]], C=A*B^T with causal mask.
        mem[0]=16'h0201; mem[1]=16'h0403; mem[10]=16'h0201; mem[11]=16'h0403;
        mem[100]=5; mem[101]=10; mem[102]=16'h8421; mem[103]=16'h0202;
        mem[104]=16'h0004; mem[105]=1; mem[106]=0; mem[107]=0;
        repeat(3) @(posedge clk); rst<=0; @(posedge clk); start<=1; @(posedge clk); start<=0;
        while(!done && cycles<2000) begin @(posedge clk); cycles=cycles+1; end
        if(!done || error || mem[20]!==16'h0005 || mem[21]!==16'h190b) begin
            $display("VMATMUL_FAIL done=%0d error=%0d words=%h %h cycles=%0d",done,error,mem[20],mem[21],cycles); $fatal;
        end
        $display("VMATMUL_PASS cycles=%0d",cycles);
        // Row-major 2x2 -> LIN, then its defined inverse. Padding is zero.
        causal<=0; subop<=5; src0<=0; dst<=40; param<=110; length<=4;
        mem[110]=6; mem[111]=16'h0021; mem[112]=0; mem[113]=16'h0202;
        repeat(2) @(posedge clk); start<=1; @(posedge clk); start<=0; cycles=0;
        while(!done && cycles<2000) begin @(posedge clk); cycles=cycles+1; end
        if(!done || error || mem[40]!==16'h0301 || mem[41]!==0 || mem[42]!==16'h0402 || mem[43]!==0) begin
            $display("VLAYOUT_FWD_FAIL done=%0d error=%0d words=%h %h %h %h",done,error,mem[40],mem[41],mem[42],mem[43]); $fatal;
        end
        src0<=40; dst<=50; param<=118; mem[118]=6; mem[119]=16'h0021; mem[120]=3; mem[121]=16'h0202;
        repeat(2) @(posedge clk); start<=1; @(posedge clk); start<=0; cycles=0;
        while(!done && cycles<2000) begin @(posedge clk); cycles=cycles+1; end
        if(!done || error || mem[50]!==16'h0201 || mem[51]!==16'h0403) begin
            $display("VLAYOUT_REV_FAIL done=%0d error=%0d words=%h %h",done,error,mem[50],mem[51]); $fatal;
        end
        $display("VLAYOUT_PASS cycles=%0d",cycles); $finish;
    end
endmodule

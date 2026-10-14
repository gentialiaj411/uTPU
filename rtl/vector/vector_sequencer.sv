// One-buffer-port controller for the v1 VEC operations.  It intentionally
// serializes words: the surrounding top FSM owns the port while busy.
module vector_sequencer #(
    parameter ADDRESS_SIZE = 16
) (
    input logic clk, input logic rst, input logic start,
    input logic [3:0] subop, input logic [ADDRESS_SIZE-1:0] dst, src0, param,
    input logic [15:0] length, input logic [15:0] buf_rdata,
    output logic busy, done, error,
    output logic buf_we, buf_re, buf_store_en,
    output logic [ADDRESS_SIZE-1:0] buf_addr,
    output logic [15:0] buf_wdata
);
    localparam [3:0] ADD=0, NORM=1, POS=2, SOFT=3;
    localparam [4:0] IDLE=0, PR=1, PW0=2, PW1=3, CHECK=4,
                     GR=5, GW0=6, GW1=7, AR=8, AW0=9, AW1=10,
                     BR=11, BW0=12, BW1=13, WRITE=14, WW=15,
                     NAR=16, NAW0=17, NAW1=18, NWRITE=19, NWW=20,
                     SR=21, SW0=22, SW1=23, SWRITE=24, SWW=25,
                     FINISH=26;
    logic [4:0] state;
    logic [15:0] record [0:7];
    logic signed [7:0] a [0:47], b [0:47], gamma [0:47], score [0:31];
    logic [15:0] norm_word, soft_word;
    logic signed [7:0] ea, eb, eo;
    logic [5:0] gamma_shift, valid_count;
    logic [5:0] index;
    logic [5:0] row;
    logic [15:0] word_a, word_b;
    logic [15:0] words_total;

    vector_serial_alu u_alu(.a(a),.gamma(gamma),.score(score),.word_index(index),
        .gamma_shift(gamma_shift),.valid_count(valid_count),.norm_word(norm_word),.soft_word(soft_word));

    function automatic signed [63:0] rne_shift(input signed [63:0] v, input integer sh);
        reg signed [63:0] mag,q,rem,half;
        begin
            if (sh <= 0) rne_shift=v;
            else begin mag=v<0?-v:v; q=mag>>>sh; rem=mag-(q<<<sh); half=64'sd1<<<(sh-1);
                if ((rem>half)||((rem==half)&&q[0])) q=q+1;
                rne_shift=v<0?-q:q; end
        end
    endfunction
    function automatic signed [7:0] sat8(input signed [63:0] v);
        if(v>127) sat8=127; else if(v < -128) sat8=-128; else sat8=v[7:0];
    endfunction

    always_comb begin
        buf_we=0; buf_re=0; buf_store_en=0; buf_addr=0; buf_wdata=0;
        case(state)
            PR: begin buf_re=1; buf_store_en=1; buf_addr=param+index; end
            GR: begin buf_re=1; buf_store_en=1; buf_addr=record[1]+index; end
            AR, NAR: begin buf_re=1; buf_store_en=1; buf_addr=src0 + ((subop==NORM) ? row*24+index : index); end
            BR: begin buf_re=1; buf_store_en=1; buf_addr=(subop==POS ? record[1] : record[1])+index; end
            SR: begin buf_re=1; buf_store_en=1; buf_addr=src0+row*16+index; end
            WRITE: begin buf_we=1; buf_store_en=1; buf_addr=dst+index; buf_wdata={sat8(rne_shift($signed(word_a[15:8]),eo-ea)+rne_shift($signed(word_b[15:8]),eo-eb)),sat8(rne_shift($signed(word_a[7:0]),eo-ea)+rne_shift($signed(word_b[7:0]),eo-eb))}; end
            NWRITE: begin buf_we=1; buf_store_en=1; buf_addr=dst+row*24+index; buf_wdata=norm_word; end
            SWRITE: begin buf_we=1; buf_store_en=1; buf_addr=dst+row*16+index; buf_wdata=soft_word; end
            default: ;
        endcase
    end

    always_ff @(posedge clk) begin
        if(rst) begin state<=IDLE; busy<=0; done<=0; error<=0; index<=0; row<=0; end
        else begin
            done<=0;
            case(state)
                IDLE: if(start) begin busy<=1; error<=0; index<=0; row<=0; words_total<=length>>1; state<=PR; end
                PR: state<=PW0; PW0: state<=PW1;
                PW1: begin record[index]<=buf_rdata; if(index==7) begin index<=0; state<=CHECK; end else begin index<=index+1; state<=PR; end end
                CHECK: begin
                    ea<=record[2][7:0]; eb<=record[2][15:8]; eo<=record[3][7:0]; gamma_shift<=record[2][5:0];
                    if((subop==NORM && record[0]!=2) || (subop==ADD && record[0]!=1) || (subop==POS && record[0]!=3) || (subop==SOFT && record[0]!=4)) begin error<=1; state<=FINISH; end
                    else if(subop==NORM) state<=GR; else if(subop==SOFT) state<=SR; else state<=AR;
                end
                GR: state<=GW0; GW0: state<=GW1;
                GW1: begin gamma[index*2]<=buf_rdata[7:0]; gamma[index*2+1]<=buf_rdata[15:8]; if(index==23) begin index<=0; state<=NAR; end else begin index<=index+1; state<=GR; end end
                AR: state<=AW0; AW0: state<=AW1;
                AW1: begin word_a<=buf_rdata; if(subop==NORM) begin a[index*2]<=buf_rdata[7:0]; a[index*2+1]<=buf_rdata[15:8]; if(index==23) begin index<=0; state<=NWRITE; end else begin index<=index+1; state<=NAR; end end else state<=BR; end
                NAR: state<=NAW0; NAW0: state<=NAW1;
                NAW1: begin a[index*2]<=buf_rdata[7:0]; a[index*2+1]<=buf_rdata[15:8]; if(index==23) begin index<=0; state<=NWRITE; end else begin index<=index+1; state<=NAR; end end
                BR: state<=BW0; BW0: state<=BW1;
                BW1: begin word_b<=buf_rdata; state<=WRITE; end
                WRITE: state<=WW;
                WW: if(index+1>=words_total) begin state<=FINISH; end else begin index<=index+1; state<=AR; end
                NWRITE: state<=NWW;
                NWW: if(index==23) begin if(row+1>=words_total/24) state<=FINISH; else begin row<=row+1; index<=0; state<=NAR; end end else begin index<=index+1; state<=NWRITE; end
                SR: state<=SW0; SW0: state<=SW1;
                SW1: begin score[index*2]<=buf_rdata[7:0]; score[index*2+1]<=buf_rdata[15:8]; if(index==15) begin valid_count<=row+1; index<=0; state<=SWRITE; end else begin index<=index+1; state<=SR; end end
                SWRITE: state<=SWW;
                SWW: if(index==15) begin if(row==31) state<=FINISH; else begin row<=row+1; index<=0; state<=SR; end end else begin index<=index+1; state<=SWRITE; end
                FINISH: begin busy<=0; done<=1; state<=IDLE; end
                default: state<=IDLE;
            endcase
        end
    end
endmodule

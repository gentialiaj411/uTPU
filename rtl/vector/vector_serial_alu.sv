// Area-oriented companion to vector_alu: one norm word and one softmax word
// per cycle.  The sequencer owns row storage and advances word_index.
module vector_serial_alu(
 input logic signed [7:0] a[0:47], input logic signed [7:0] gamma[0:47],
 input logic signed [7:0] score[0:31], input logic [5:0] word_index,
 input logic [5:0] gamma_shift, input logic [5:0] valid_count,
 output logic [15:0] norm_word, output logic [15:0] soft_word);
 logic [15:0] exp_q15[0:128],rsqrt_q14[0:191]; integer i,ss,q,rem,e,m,bin,mx,rel,sum,num;
 reg signed [63:0] r0,t,r1,p0,p1;
 initial begin $readmemh("rtl/vector/exp_q15.mem",exp_q15); $readmemh("rtl/vector/rsqrt_q14.mem",rsqrt_q14); end
 function automatic signed [63:0] rne(input signed [63:0] v,input integer sh);
   reg signed[63:0] z,d,r,h; begin if(sh<=0) rne=v; else begin z=v<0?-v:v;d=z>>>sh;r=z-(d<<<sh);h=64'sd1<<<(sh-1);if(r>h||(r==h&&d[0]))d=d+1;rne=v<0?-d:d;end end endfunction
 function automatic [7:0] sat(input signed [63:0] v); begin if(v>127)sat=8'h7f;else if(v < -128)sat=8'h80;else sat=v[7:0];end endfunction
 always_comb begin
   ss=0;for(i=0;i<48;i=i+1)ss=ss+a[i]*a[i];q=ss/48;rem=ss%48;if(2*rem>48||(2*rem==48&&q[0]))q=q+1;q=q+1;e=0;for(i=0;i<31;i=i+1)if(q>=(1<<i)&&!(i&1))e=i;m=rne($signed(q)<<<14,e);if(m<16384)m=16384;if(m>65535)m=65535;bin=(m-16384)>>8;if(bin>191)bin=191;r0=rsqrt_q14[bin];t=rne($signed(m)*r0*r0,28);r1=rne(r0*(49152-t),15);p0=$signed(a[word_index*2])*$signed(gamma[word_index*2])*r1;p1=$signed(a[word_index*2+1])*$signed(gamma[word_index*2+1])*r1;norm_word={sat(rne(p1,14+gamma_shift+(e>>1))),sat(rne(p0,14+gamma_shift+(e>>1)))};
   mx=-128;for(i=0;i<32;i=i+1)if(i<valid_count&&score[i]>mx)mx=score[i];sum=0;for(i=0;i<32;i=i+1)if(i<valid_count)begin rel=score[i]-mx;if(rel < -128)rel=-128;if(rel>0)rel=0;sum=sum+exp_q15[-rel];end soft_word=0;for(i=0;i<2;i=i+1)if(word_index*2+i<valid_count)begin rel=score[word_index*2+i]-mx;if(rel < -128)rel=-128;if(rel>0)rel=0;num=255*exp_q15[-rel]+sum/2;if(i==0)soft_word[7:0]=num/sum;else soft_word[15:8]=num/sum;end
 end
endmodule

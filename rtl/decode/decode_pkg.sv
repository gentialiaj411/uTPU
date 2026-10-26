// Shared integer helpers for the decode engine. Each mirrors numerics_v2/fixed.py on int64
// values (numpy int64 wrap-around semantics), so units can be compared bit-exactly.
package decode_pkg;

    // fixed.rshift_rne: signed shift right, round half to even on the magnitude; s == 0 passes x.
    function automatic logic signed [63:0] rshift_rne(input logic signed [63:0] x, input logic [6:0] s);
        logic [63:0] mag, q, rem, half;
        logic inc;
        if (s == 0) return x;
        mag  = x[63] ? 64'(-x) : 64'(x);
        q    = mag >> s;
        rem  = mag & ((64'd1 << s) - 64'd1);
        half = 64'd1 << (s - 7'd1);
        inc  = (rem > half) || ((rem == half) && q[0]);
        q    = q + 64'(inc);
        return x[63] ? -$signed(q) : $signed(q);
    endfunction

    // fixed.div_const_rne quotient estimate. small: |n| <= 2^31 path (RNE of mag*R/2^32);
    // otherwise the wide path (round half up).
    function automatic logic [63:0] div_const_q0(input logic [63:0] mag, input logic [32:0] recip);
        return div_const_q0_from_prod(mag, 97'(mag) * 97'(recip));
    endfunction

    // Same, given the precomputed product mag * recip (e.g. from seq_mul).
    function automatic logic [63:0] div_const_q0_from_prod(input logic [63:0] mag, input logic [96:0] prod);
        logic [63:0] q;
        logic [31:0] frac;
        if (mag <= 64'h8000_0000) begin
            q    = prod[95:32];
            frac = prod[31:0];
            if (frac > 32'h8000_0000 || (frac == 32'h8000_0000 && q[0])) q = q + 1;
            return q;
        end
        return 64'((prod + (97'd1 << 31)) >> 32);
    endfunction

    // mag - q*d as a wide signed value
    function automatic logic signed [79:0] div_resid(input logic [63:0] mag, input logic [63:0] q, input logic [9:0] d);
        return $signed({16'd0, mag}) - $signed({16'd0, q}) * $signed({70'd0, d});
    endfunction

    // fixed.div_const_rne correction. The small path applies three sequential numpy updates
    // (each later test sees the updated q, the error term uses the original q0); the wide
    // path applies a single if/elif correction.
    function automatic logic [63:0] div_const_fix(input logic [63:0] mag, input logic [63:0] q0, input logic [9:0] d);
        logic signed [79:0] r0;
        logic [79:0] err2;
        logic gt, tie;
        logic [63:0] q;
        r0   = div_resid(mag, q0, d);
        err2 = r0[79] ? 80'(-r0) << 1 : 80'(r0) << 1;
        gt   = err2 > 80'(d);
        tie  = (err2 == 80'(d)) && q0[0];
        if (mag > 64'h8000_0000) begin
            if (gt) return r0[79] ? q0 - 1 : q0 + 1;
            if (tie && r0[79]) return q0 - 1;
            return q0;
        end
        q = q0;
        if (gt && !r0[79] && r0 != 0) q = q + 1;               // mag > q*d
        if (gt && div_resid(mag, q, d) < 0) q = q - 1;          // mag < q*d (updated q)
        if (tie && div_resid(mag, q, d) < 0) q = q - 1;
        return q;
    endfunction

    // div_const_fix given r0 = mag - q0*d (registered by the caller); q moves by +-1 so the
    // residual is updated by -+d instead of being re-multiplied.
    function automatic logic [63:0] div_const_fix_resid(input logic wide, input logic [63:0] q0,
                                                        input logic signed [79:0] r0, input logic [9:0] d);
        logic [79:0] err2;
        logic gt, tie;
        logic [63:0] q;
        logic signed [79:0] r;
        err2 = r0[79] ? 80'(-r0) << 1 : 80'(r0) << 1;
        gt   = err2 > 80'(d);
        tie  = (err2 == 80'(d)) && q0[0];
        if (wide) begin
            if (gt) return r0[79] ? q0 - 1 : q0 + 1;
            if (tie && r0[79]) return q0 - 1;
            return q0;
        end
        q = q0; r = r0;
        if (gt && !r0[79] && r0 != 0) begin q = q + 1; r = r - $signed({70'd0, d}); end
        if (gt && r[79]) begin q = q - 1; r = r + $signed({70'd0, d}); end
        if (tie && r[79]) q = q - 1;
        return q;
    endfunction

endpackage

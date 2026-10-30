# P5 prefetch review
The premise that the reader idles throughout RMS/attention/SwiGLU/quant is false in the integrated RTL. top_decode.sv has an independent four-segment prefetch state machine. st_ready depends on pf_active, pf_seek, pf_left and FIFO capacity, not the compute state. Thus it already prefetches the next sequential matrix during those phases until the 4096-byte FIFO fills. A second prefetch enable parameter cannot create this overlap again.

Potential alternative: enlarge FIFO from 4096 to 8192 or more, costing at least another BRAM36 per 4096 bytes. At 0.25 B/clk, filling 4096 bytes takes DERIVED 16384 clocks, enough to cover only part of the long phases. Some embedding/classifier segment transitions are already handled independently; do not flush or reposition the FIFO without preserving segment order and consumed-byte accounting.

Disposition: TIMEBOXED design-only. No speculative ON parameter or RTL change retained. An occupancy/wait-reason counter plus controlled FIFO-depth sweep would be required before recommending extra BRAM. Exact NOT RUN commands require adding a runner FIFO-depth override, not silently changing the DUT defaults.

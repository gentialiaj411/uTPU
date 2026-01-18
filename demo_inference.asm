; uTPU Assembly Demo: Single 2x2 Matrix Multiplication Tile
; This demonstrates the atomic operation repeated ~1000 times for the full network.
; Operation: Outputs = ReLU(Quantize(Inputs * Weights))

; -------------------------------------------------------------------------
; 1. LOAD DATA TO BUFFER
; -------------------------------------------------------------------------

; Store Weights [1, 2, 3, 4] into Memory Section B (0x080)
; Packed as 4-bit values: 0x4321
STORE #0x4321, 0x080

; Store Inputs [1, 1, 0, 0] into Memory Section A (0x000)
; Packed as 4-bit values: 0x0011 (Inputs are 1, 1, padded with 0s)
STORE #0x0011, 0x000

; -------------------------------------------------------------------------
; 2. LOAD COMPUTE ARRAY
; -------------------------------------------------------------------------

; Load Weights from Address 0x080 into the PE Array
; Opcode: LOAD, with Weigh-Load-Enable bit set
LOADWEI 0x080

; Load Inputs from Address 0x000 into the PE Array
; Opcode: LOAD (Input mode default)
LOADIN 0x000

; -------------------------------------------------------------------------
; 3. EXECUTE
; -------------------------------------------------------------------------

; Run Compute Cycle
; C = Enable Compute (Matrix Mult)
; Q = Enable Quantizer (Int16 -> Int4)
; R = Enable Relu (Activation)
; Result Destination: Memory Section C (0x100)
RUN 0x100 C Q R

; -------------------------------------------------------------------------
; 4. RETRIEVE RESULT
; -------------------------------------------------------------------------

; Fetch the result from Memory Section C (0x100) to UART
FETCH 0x100

; Stop execution
HALT

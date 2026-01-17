==========
## SPEC
==========

### Overview

Composed of an 8-bit UART communications module
buffered with a 256 B wide FIFO on the UART input
and output. These FIFOs connect to the unified buffer, 
which has a word size of 2 B. The memory connects to 
the compute unit made of a 4-bit 2x2 systolic array, 
quantizer, and leakyrelu modules.

UART RX -> FIFO RX ------------>|
                      MAC ARRAY | BUFFER
                                |
                      QUANTIZER |
                                |
                      LEAKYRELU |
                                |
UART TX <- FIFO TX <------------|


### ISA
    Multi-word instructions where each word is 16 bits

    Address is 9 bits wide
    
    A-TYPE STORE (for now address_length is not used as always 4 bits)
     3 bits  1 bit     1 bit                         4 bits                         4 bits
    [OPCODE] [NOT USED] [IMMEDIATE/ADDRESS INDICATOR] [address_lengthb or NOT USED] [address_lengtha]
              (Was top/bottom but would only be used if sending 2 int4 instead of 4)
     7 bits     9 bits
    [NOT USED] [ADDRESS]
    or 
    [int4] [int4] [int4] [int4]

    - FIRST SEND THE 4xint4 (or address to where it is) then the address of where to store it

    C-TYPE RUN (if COMPUTE_EN then QUANTIZER_EN implied)
    3 bits    1 bit        1 bits        1 bit     9 bits
    [OPCODE] [COMPUTE_EN] [QUANTIZER_EN] [RELU_EN] [RESULT_ADDRESS]

    B-TYPE LOAD ( WILL HAVE TO BE CHANGED IF DIMENSIONS OF MAC ARRAY CHANGES) for 2x2 loads top values
    may need to be changed if array size is not divisible by 4
    3 bits    1 bit                  3 bits     9 bits
    [OPCODE] [IN/WEIGHTS INDICATOR] [NOT USED] [ADDRESS]

    B-TYPE FETCH
    3 bits   1 bit     3 bits      9 bis
    [OPCODE] [TOP/BOT] [NOT USED] [ADDRESS]

    D-TYPE 
    [OPCODE] [NOT USED]



### Modules
#### UART Module
    Is used retrieve information from the user. 
    

    TODO: Figure what else this is going to do


#### 2x2 Systolic MAC Array:
    Has a accumulate resgister inside storing weights
    Every clock it takes its inputs, multiplies them and
    adds to the accumulated value passed to it
    This output is then sent down the array 
    Can be put into modes to load weights

    Control signals:
    - compute (this causes mac flow through array)
    - load_en (allows the internal weights to be loaded)

#### MAC:
    Performs a multiply and then adds result to the accumulator
    Each are loaded with a weight and that multiplies to input
    After every multiply the accumulator sum is outputted
    
    Control signals:
    - compute (this causes mac flow through array)
    - load_en (allows the internal weights to be loaded)


#### Quantizer:
    The int16 output from the MAC array will be quantized 
    into a int4. 

    [-32768, 32767] -> [-8, 7]

    All this does is >> 12


#### LeakyReLU Unit:
    Takes the int4 output of the Quantizer and performs ReLU
    
    leakyReLU = x if x>0 else ax

    a is stored in a register in the unit


#### Unified Buffer:
    All of the tensor are stored here
    1 KB with 2 Byte words

    0x000-0x1FF TOTAL

    Partitioned into four sections

    0x000-0x07F A
    0x080-0x0FF B
    0x100-0x17F C
    0x180-0x1FF D
   
   Control signals:
    - we 
    - re
    - array_en
    - fifo_en

#### FIFO:
    This is a queue between the uart module and the core
    It's values are one byte wide and is 256 B in total.
    
    Design based on: Simulation and Synthesis Techniques for Asynchronous
    FIFO Design - Clifford E. Cummings 
    - The method uses both a w_ptr and r_ptr instead of moving the data
    around
   
    The program will live here

    0x000 - 0x0FF
    
    Has a pointer which points to first open slot
    Control Signals:
    - clk
    - rst
    - we
    - re
    
    Flags:
    - Empty
    - Full

### Possible Problems in Design
    1. What happens if UART_RX begins receiving data before the fifo has stored the previous



### What to do

    Fix the laning issue between the buffer and compute 
        - mem to 4-input
    Work out how to control the parameters with the toml file
    
    

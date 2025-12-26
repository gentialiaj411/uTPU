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

UART RX -> FIFO RX -------------|
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
    
    A-TYPE STORE
     3 bits  1 bit     1 bit                         4 bits                         4 bits
    [OPCODE] [TOP/BOT] [IMMEDIATE/ADDRESS INDICATOR] [address_lengthb or NOT USED] [address_lengtha]
     
     7 bits     9 bits
    [NOT USED] [ADDRESS]
    or 
    [int4] [int4] [int4] [int4]

    B-TYPE RUN
    3 bits    1 bit     3 bits      9 bits
    [OPCODE] [RELU_EN] [NOT USED] [RESULT_ADDRESS]

    B-TYPE LOAD ( WILL HAVE TO BE CHANGED IF DIMENSIONS OF MAC ARRAY CHANGES)
    3 bits    1 bit                  3 bits     9 bits
    [OPCODE] [IN/WEIGHTS INDICATOR] [NOT USED] [ADDRESS]

    B-TYPE FETCH
    3 bits   1 bit     3 bits      9 bis
    [OPCODE] [TOP/BOT] [NOT USED] [ADDRESS]

    D-TYPE 
    [OPCODE] [NOT USED]

#### Opcodes (3 bits)
    
    STORE (A-TYPE)

    FETCH (B-TYPE)

    RUN (B-TYPE)

    LOAD (B-TYPE)

    HALT (D-TYPE)

    **** (NOT SET)

    **** (NOT SET)

    NOP (D-TYPE)


#### Instructions
    
    STORETOP/STOREBOT - which loads the top/bottom of the 16 bit mem location
    
    ENRELU/DISRELU - which change the activation state of RELU module
    
    LOADIN - loads the 4 vals at the address into the compute unit input
     
    LOADWEI - loads the 4 weights at the address into mac array
    
    RUN -  allows the inputs to flow through the compute unit
    
    FETCHTOP/FETCHBOT - which returns the top/bottom 8 bits at mem location
        
    HALT
      
    NOP


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



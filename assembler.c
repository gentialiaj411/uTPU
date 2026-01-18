/*
 * uTPU Assembler - Final Version
 * 
 * INSTRUCTION FORMAT (16 bits):
 * ============================
 * Bits [2:0]   = OPCODE (3 bits)
 * Bit  [3]     = Mode0 (TOP/BOT for STORE/FETCH, load_en for LOAD, COMPUTE_EN for RUN)
 * Bit  [4]     = Mode1 (ADDR_INDICATOR for STORE, QUANTIZER_EN for RUN)
 * Bit  [5]     = Mode2 (RELU_EN for RUN only)
 * Bit  [6]     = unused
 * Bits [15:7]  = ADDRESS (9 bits)
 *
 * BYTE ORDER: Low byte first, then High byte
 * Hardware expects the lowest 8 bits of a word first via UART.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

#define MAX_LINE_LENGTH 256
#define MAX_INSTRUCTIONS 1024
#define ADDRESS_BITS 9
#define OPCODE_BITS 3

typedef enum {
    OP_STORE = 0,   /* 000 */
    OP_FETCH = 1,   /* 001 */
    OP_RUN   = 2,   /* 010 */
    OP_LOAD  = 3,   /* 011 */
    OP_HALT  = 4,   /* 100 */
    OP_NOP   = 5,   /* 101 */
    OP_INVALID = -1
} Opcode;

//Trim whitespace from string 
char* trim(char* str) {
    while (isspace((unsigned char)*str)) str++;
    if (*str == 0) return str;
    char* end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return str;
}

void to_upper(char* str) {
    for (int i = 0; str[i]; i++) {
        str[i] = toupper((unsigned char)str[i]);
    }
}

uint16_t parse_number(const char* token) {
    if (token == NULL) return 0;
    if (token[0] == '0' && (token[1] == 'x' || token[1] == 'X')) {
        return (uint16_t)strtol(token, NULL, 16);
    }
    return (uint16_t)strtol(token, NULL, 10);
}

Opcode get_opcode(const char* mnemonic) {
    if (strcmp(mnemonic, "STORE") == 0 ||
        strcmp(mnemonic, "STORETOP") == 0 ||
        strcmp(mnemonic, "STOREBOT") == 0) {
        return OP_STORE;
    }
    if (strcmp(mnemonic, "FETCH") == 0 ||
        strcmp(mnemonic, "FETCHTOP") == 0 ||
        strcmp(mnemonic, "FETCHBOT") == 0) {
        return OP_FETCH;
    }
    if (strcmp(mnemonic, "RUN") == 0) {
        return OP_RUN;
    }
    if (strcmp(mnemonic, "LOAD") == 0 ||
        strcmp(mnemonic, "LOADIN") == 0 ||
        strcmp(mnemonic, "LOADWEI") == 0) {
        return OP_LOAD;
    }
    if (strcmp(mnemonic, "HALT") == 0) {
        return OP_HALT;
    }
    if (strcmp(mnemonic, "NOP") == 0) {
        return OP_NOP;
    }
    return OP_INVALID;
}

   // Returns: number of words generated (1 or 2), 0 for empty/comment, -1 for error
 
int encode_instruction(const char* line, int line_num, uint16_t* output) {
    char buffer[MAX_LINE_LENGTH];
    strncpy(buffer, line, MAX_LINE_LENGTH - 1);
    buffer[MAX_LINE_LENGTH - 1] = '\0';
    
    // Only use ';' for comments (# is used for immediate values)
    char* comment = strchr(buffer, ';');
    if (comment) *comment = '\0';
    
    char* trimmed = trim(buffer);
    if (strlen(trimmed) == 0) {
        return 0;
    }
    
    char* tokens[8] = {NULL};
    int token_count = 0;
    char* token = strtok(trimmed, " \t,");
    while (token && token_count < 8) {
        tokens[token_count++] = token;
        token = strtok(NULL, " \t,");
    }
    
    if (token_count == 0) {
        return 0;
    }
    char mnemonic[32];
    strncpy(mnemonic, tokens[0], sizeof(mnemonic) - 1);
    mnemonic[sizeof(mnemonic) - 1] = '\0';
    to_upper(mnemonic);
    
    Opcode opcode = get_opcode(mnemonic);
    if (opcode == OP_INVALID) {
        fprintf(stderr, "Error (line %d): Unknown instruction '%s'\n", line_num, tokens[0]);
        return -1;
    }
    
    uint16_t instr = 0;
    int words = 1;
    
    switch (opcode) {
        case OP_NOP:
        case OP_HALT:
            instr = (uint16_t)opcode;
            break;
        
        case OP_FETCH: {
            int is_bot = (strstr(mnemonic, "BOT") != NULL) ? 1 : 0;
            uint16_t addr = 0;
            if (token_count > 1) {
                addr = parse_number(tokens[1]) & 0x1FF;
            }
            instr = (uint16_t)opcode;
            instr |= (is_bot << 3);
            instr |= (addr << 7);
            break;
        }
        
        case OP_LOAD: {
            int load_en = (strstr(mnemonic, "WEI") != NULL) ? 1 : 0;
            uint16_t addr = 0;
            if (token_count > 1) {
                addr = parse_number(tokens[1]) & 0x1FF;
            }
            instr = (uint16_t)opcode;
            instr |= (load_en << 3);
            instr |= (addr << 7);
            break;
        }
        
        case OP_RUN: {
            int compute_en = 1, quant_en = 1, relu_en = 1;
            uint16_t addr = 0;
            for (int i = 1; i < token_count; i++) {
                char upper_tok[32];
                strncpy(upper_tok, tokens[i], sizeof(upper_tok) - 1);
                upper_tok[sizeof(upper_tok) - 1] = '\0';
                to_upper(upper_tok);
                int has_flags = (strchr(upper_tok, 'C') || strchr(upper_tok, 'Q') || strchr(upper_tok, 'R'));
                if (has_flags && !isdigit(upper_tok[0])) {
                    compute_en = (strchr(upper_tok, 'C') != NULL) ? 1 : 0;
                    quant_en = (strchr(upper_tok, 'Q') != NULL) ? 1 : 0;
                    relu_en = (strchr(upper_tok, 'R') != NULL) ? 1 : 0;
                } else {
                    addr = parse_number(tokens[i]) & 0x1FF;
                }
            }
            instr = (uint16_t)opcode;
            instr |= (compute_en << 3);
            instr |= (quant_en << 4);
            instr |= (relu_en << 5);
            instr |= (addr << 7);
            break;
        }
        
        case OP_STORE: {
            int is_bot = (strstr(mnemonic, "BOT") != NULL) ? 1 : 0;
            uint16_t source_val = 0;
            uint16_t dest_addr = 0;
            int source_is_addr = 1; // Default to address

            // Expecting 2 arguments: STORE <source>, <dest>
            if (token_count < 3) {
                 fprintf(stderr, "Error (line %d): STORE requires 2 arguments: source, dest\n", line_num);
                 return -1;
            }

            // Parse Source (Token 1)
            char* src_token = tokens[1];
            if (src_token[0] == '#') {
                source_is_addr = 0; // Immediate
                source_val = parse_number(src_token + 1); // Skip '#'
            } else {
                source_is_addr = 1; // Address
                source_val = parse_number(src_token) & 0x1FF;
            }

            // Parse Destination (Token 2)
            dest_addr = parse_number(tokens[2]) & 0x1FF;

            // Word 1: Opcode + is_bot (bit 3) + source_type (bit 4)
            instr = (uint16_t)opcode;
            instr |= (is_bot << 3);
            instr |= (source_is_addr << 4); 
            output[0] = instr;

            // Word 2: Source Value
            output[1] = source_val;

            // Word 3: Destination Address
            output[2] = dest_addr;

            words = 3;
            return words;
        }
        default: break;
    }
    
    output[0] = instr;
    return words;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("uTPU Assembler\nUsage: %s <input.asm> [-o output_base]\n", argv[0]);
        return 1;
    }
    const char* input_file = argv[1];
    char output_base[256];
    strncpy(output_base, input_file, sizeof(output_base) - 1);
    output_base[sizeof(output_base) - 1] = '\0';
    char* dot = strrchr(output_base, '.');
    if (dot) *dot = '\0';
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            strncpy(output_base, argv[++i], sizeof(output_base) - 1);
            output_base[sizeof(output_base) - 1] = '\0';
        }
    }
    FILE* fp = fopen(input_file, "r");
    if (!fp) {
        fprintf(stderr, "Error: Could not open '%s'\n", input_file);
        return 1;
    }
    uint16_t instructions[MAX_INSTRUCTIONS];
    int count = 0, line_num = 0, errors = 0;
    char line[MAX_LINE_LENGTH];
    while (fgets(line, sizeof(line), fp)) {
        line_num++;
        uint16_t out[3];
        int words = encode_instruction(line, line_num, out);
        if (words < 0) errors++;
        else if (words > 0) {
            for (int i = 0; i < words && count < MAX_INSTRUCTIONS; i++) {
                instructions[count++] = out[i];
            }
        }
    }
    fclose(fp);
    if (errors > 0) return 1;

    // Write .mem file (Hex) 
    char mem_file[512];
    snprintf(mem_file, sizeof(mem_file), "%s.mem", output_base);
    FILE* mem_fp = fopen(mem_file, "w");
    for (int i = 0; i < count; i++) fprintf(mem_fp, "%04X\n", instructions[i]);
    fclose(mem_fp);

    // Write .bin file
    char bin_file[512];
    snprintf(bin_file, sizeof(bin_file), "%s.bin", output_base);
    FILE* bin_fp = fopen(bin_file, "wb");
    for (int i = 0; i < count; i++) {
        uint8_t low = instructions[i] & 0xFF;
        uint8_t high = (instructions[i] >> 8) & 0xFF;
        fwrite(&low, 1, 1, bin_fp);
        fwrite(&high, 1, 1, bin_fp);
    }
    fclose(bin_fp);
    printf("Assembled %d words to %s.mem and %s.bin\n", count, output_base, output_base);
    return 0;
}

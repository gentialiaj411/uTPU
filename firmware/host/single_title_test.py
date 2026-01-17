import sys
import numpy as np
sys.path.append('..')
from host.uart_driver import UARTDriver
from host.program_loader import ProgramLoader


TEST_CASES = [
    # Test 1: Simple positive values
    {
        'name': 'Simple Positive',
        'weights': [1, 1, 1, 1],  
        'inputs': [2, 3],         
        'expected': [5, 5],       
        'description': 'All weights = 1, should sum inputs'
    },
    
    {
        'name': 'Weighted Sum',
        'weights': [1, 2, 3, 4],
        'inputs': [1, 1],
        'expected': [3, 7],       
        'description': 'Weights [1,2,3,4], inputs [1,1]'
    },
    {
        'name': 'Negative Output',
        'weights': [-2, -2, -2, -2], 
        'inputs': [1, 1],
        'expected': [-1, -1],        
        'description': 'Should test LeakyReLU with negative'
    },
    
    # Test 4: Zero inputs
    {
        'name': 'Zero Inputs',
        'weights': [5, 6, 7, 7],
        'inputs': [0, 0],
        'expected': [0, 0],
        'description': 'Zero inputs should give zero outputs'
    },
    
    {
        'name': 'Testbench Case',
        'weights': [5, 6, 7, 1], 
        'inputs': [1, 2],
        'expected': [0, 0],
        'description': 'Case from pe_array_tb2.sv testbench'
    },
]

def runSingleTest(loader, test_case):
    name = test_case['name']
    weights = test_case['weights']
    inputs = test_case['inputs']
    expected = test_case['expected']
    
    print(f"\n{'='*50}")
    print(f"Test: {name}")
    print(f"Description: {test_case['description']}")
    print(f"Weights: {weights}")
    print(f"Inputs: {inputs}")
    print(f"Expected: {expected}")
    
    try:
        results = loader.execute_2x2_matmul(weights, inputs)
        
        print(f"Got: {results}")
        
        passed = (results == expected)
        
        if passed:
            print("✓ PASSED")
        else:
            print("✗ FAILED")
            print(f"  Expected {expected}, got {results}")
        
        return passed
        
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def run_all_tests(port: str) -> None:
    
    print("="*60)
    print("uTPU Single Tile Test Suite")
    print("="*60)

    uart = UARTDriver(port, baud=115200)
    loader = ProgramLoader(uart, verbose=False)
    
    print("\nResetting chip...")
    loader.resetChip()
    
    passed = 0
    failed = 0
    
    for test_case in TEST_CASES:
        if runSingleTest(loader, test_case):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{len(TEST_CASES)}")
    print(f"Failed: {failed}/{len(TEST_CASES)}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {failed} test(s) failed")
    
    uart.close()

def interactive_test(port: str) -> None:
    
    print("="*60)
    print("uTPU Interactive Test Mode")
    print("="*60)
    print("Enter 'q' to quit\n")
    
    uart = UARTDriver(port, baud=115200)
    loader = ProgramLoader(uart, verbose=True)
    loader.resetChip()
    
    while True:
        try:
            print("\nEnter 4 weights (comma-separated, e.g., 1,2,3,4):")
            weights_input = input("> ").strip()
            
            if weights_input.lower() == 'q':
                break
            
            weights = [int(x.strip()) for x in weights_input.split(',')]
            if len(weights) != 4:
                print("Error: Need exactly 4 weights")
                continue
            
            print("Enter 2 inputs (comma-separated, e.g., 1,2):")
            inputs_input = input("> ").strip()
            
            if inputs_input.lower() == 'q':
                break
            
            inputs = [int(x.strip()) for x in inputs_input.split(',')]
            if len(inputs) != 2:
                print("Error: Need exactly 2 inputs")
                continue
            
            print("\nRunning computation...")
            results = loader.execute2x2MatMul(
                weights, 
                inputs,
                ProgramLoader.BUFFER_SECTION_B,
                ProgramLoader.BUFFER_SECTION_A,
                ProgramLoader.BUFFER_SECTION_C
                )
            
            print(f"\nWeights: [[{weights[0]}, {weights[1]}], [{weights[2]}, {weights[3]}]]")
            print(f"Inputs: [{inputs[0]}, {inputs[1]}]")
            print(f"Results: {results}")
            
            manual_0 = inputs[0] * weights[0] + inputs[1] * weights[1]
            manual_1 = inputs[0] * weights[2] + inputs[1] * weights[3]
            print(f"\nManual (before quant): [{manual_0}, {manual_1}]")
            
        except ValueError as e:
            print(f"Invalid input: {e}")
        except Exception as e:
            print(f"Error: {e}")
    
    uart.close()
    print("\nGoodbye!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='uTPU Single Tile Test')
    parser.add_argument('--port', '-p', default='COM3', 
                        help='Serial port (default: COM3)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_test(args.port)
    else:
        run_all_tests(args.port)
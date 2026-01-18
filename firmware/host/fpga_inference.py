
import numpy as np
import sys
import os

from tiled_inference import TiledInferenceEngine, get_default_paths


#runs inference on physical fpga hardware
#uses simulation when hardware not connected
class FPGAInference:

    def __init__(self, port=None, verbose=False):
        self.verbose = verbose
        self.simulation_mode = port is None

        weights_dir, model_path, _ = get_default_paths()

        #load weights and parameters
        self.engine = TiledInferenceEngine(weights_dir, model_path, verbose=verbose)

        if not self.simulation_mode:
            try:
                from uart_driver import UARTDriver
                from program_loader import ProgramLoader

                self.uart = UARTDriver(port, baud=115200)
                self.loader = ProgramLoader(self.uart, verbose=verbose)
                self._log(f"Connected to FPGA on {port}")
                self.loader.resetChip()
            except Exception as e:
                print(f"Warning: Could not connect to FPGA: {e}")
                print("Falling back to simulation mode")
                self.simulation_mode = True

        if self.simulation_mode:
            self._log("Running in SIMULATION mode")
        else:
            self.engine.tile_runner = self.run_tile_on_hardware
            self._log("Running in HARDWARE tile mode (2x2 tiles via UART)")
            print("NOTE: Hardware tile mode uses per-tile quantized outputs; accuracy may differ from software.")

    def _log(self, msg):
        if self.verbose:
            print(f"[FPGA] {msg}")

    #run single 2x2 tile on hardware
    def run_tile_on_hardware(self, weights, inputs):
        if self.simulation_mode:
            #simulate hardware behavior
            w = weights.astype(np.int32)
            x = inputs.astype(np.int32)
            return np.array([
                w[0, 0] * x[0] + w[0, 1] * x[1],
                w[1, 0] * x[0] + w[1, 1] * x[1]
            ], dtype=np.int32)
        else:
            weight_list = weights.astype(np.int8).flatten().tolist()
            input_list = inputs.astype(np.int8).flatten().tolist()
            results = self.loader.execute2x2MatMul(
                weight_list,
                input_list,
                self.loader.BUFFER_SECTION_B,
                self.loader.BUFFER_SECTION_A,
                self.loader.BUFFER_SECTION_C,
                quantize=True,
                relu=False,
            )
            if len(results) < 2:
                raise RuntimeError("FPGA tile read returned no data (UART timeout?)")
            return np.array(results, dtype=np.int32)

    #simulated hardware tile
    def run_tile_simulated(self, weights, inputs):
        w = weights.astype(np.int32)
        x = inputs.astype(np.int32)
        return np.array([
            w[0, 0] * x[0] + w[0, 1] * x[1],
            w[1, 0] * x[0] + w[1, 1] * x[1]
        ], dtype=np.int32)

    #predict digit for image
    def predict(self, image):
        return self.engine.predict(image)

    #evaluate accuracy
    def evaluate(self, images, labels, max_samples=None):
        return self.engine.evaluate(images, labels, max_samples)

    #close uart connection
    def close(self):
        if not self.simulation_mode and hasattr(self, 'uart'):
            self.uart.close()


def main():
    import argparse

    _, _, data_dir = get_default_paths()

    parser = argparse.ArgumentParser(description='FPGA MNIST Inference')
    parser.add_argument('--port', '-p', type=str, default=None,
                        help='Serial port (e.g. COM3). Omit for simulation.')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--interactive', '-i', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    print("=" * 60)
    print("uTPU FPGA MNIST Inference")
    print("=" * 60)

    #initialize
    fpga = FPGAInference(port=args.port, verbose=args.verbose)

    #load test data
    test_images = np.load(os.path.join(data_dir, 'mnist_14x14_test.npy'))
    test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))

    print(f"Loaded {len(test_labels)} test samples")

    if args.sample is not None:
        #single sample
        pred, logits = fpga.predict(test_images[args.sample])
        actual = test_labels[args.sample]
        print(f"\nSample {args.sample}:")
        print(f"  Predicted: {pred}")
        print(f"  Actual:    {actual}")
        print(f"  {'CORRECT' if pred == actual else 'WRONG'}")

    elif args.eval:
        #full evaluation
        print(f"\nEvaluating...")
        acc, correct, total = fpga.evaluate(test_images, test_labels, args.num_samples)
        print(f"\nAccuracy: {100*acc:.2f}% ({correct}/{total})")

    elif args.interactive:
        #interactive mode
        print("\nInteractive mode. Enter sample index (0-9999) or 'q' to quit.")
        while True:
            try:
                inp = input("\nSample> ").strip()
                if inp.lower() == 'q':
                    break
                idx = int(inp)
                if 0 <= idx < len(test_labels):
                    pred, _ = fpga.predict(test_images[idx])
                    actual = test_labels[idx]
                    print(f"Predicted: {pred}, Actual: {actual} {'✓' if pred == actual else '✗'}")
                else:
                    print(f"Out of range (0-{len(test_labels)-1})")
            except ValueError:
                print("Enter a number or 'q'")
            except KeyboardInterrupt:
                break

    else:
        #quick demo
        print("\nQuick demo on 10 samples:")
        correct = 0
        for i in range(10):
            pred, _ = fpga.predict(test_images[i])
            actual = test_labels[i]
            if pred == actual:
                correct += 1
            print(f"  [{i}] Pred: {pred}, Actual: {actual} {'✓' if pred == actual else '✗'}")
        print(f"\n{correct}/10 correct")

    fpga.close()


if __name__ == "__main__":
    main()

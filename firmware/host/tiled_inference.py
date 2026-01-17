import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../software/model'))


#runs inference using 2x2 tiled matmul
#models hardware int32 accumulator behavior exactly
class TiledInferenceEngine:

    def __init__(self, weights_dir, model_path, verbose=False):
        self.verbose = verbose

        weights_dir = os.path.abspath(weights_dir)
        model_path = os.path.abspath(model_path)

        self._log(f"Loading weights from: {weights_dir}")
        self._log(f"Loading model from: {model_path}")

        #check files exist
        if not os.path.exists(weights_dir):
            raise FileNotFoundError(f"Weights directory not found: {weights_dir}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        #load int4 weights from npy files
        fc1_weight_path = os.path.join(weights_dir, 'fc1_weight.npy')
        fc2_weight_path = os.path.join(weights_dir, 'fc2_weight.npy')
        scales_path = os.path.join(weights_dir, 'scales.npy')

        if not os.path.exists(fc1_weight_path):
            raise FileNotFoundError(f"FC1 weights not found: {fc1_weight_path}")
        if not os.path.exists(fc2_weight_path):
            raise FileNotFoundError(f"FC2 weights not found: {fc2_weight_path}")
        if not os.path.exists(scales_path):
            raise FileNotFoundError(f"Scales not found: {scales_path}")

        self.fc1_weight = np.load(fc1_weight_path).astype(np.int8)  # (16, 196)
        self.fc2_weight = np.load(fc2_weight_path).astype(np.int8)  # (10, 16)

        #load scale factors
        scales = np.load(scales_path, allow_pickle=True).item()
        self.fc1_scale = float(scales['fc1_scale'])
        self.fc2_scale = float(scales['fc2_scale'])

        #load original float biases from pth file
        state_dict = self._load_state_dict(model_path)
        self.fc1_bias = self._get_tensor(state_dict, 'fc1.bias').astype(np.float32)  # (16,)
        self.fc2_bias = self._get_tensor(state_dict, 'fc2.bias').astype(np.float32)  # (10,)

        #validate shapes
        assert self.fc1_weight.shape == (16, 196)
        assert self.fc2_weight.shape == (10, 16)
        assert self.fc1_bias.shape == (16,)
        assert self.fc2_bias.shape == (10,)

        #validate weight ranges are int4
        assert self.fc1_weight.min() >= -8 and self.fc1_weight.max() <= 7
        assert self.fc2_weight.min() >= -8 and self.fc2_weight.max() <= 7

        self._log(f"FC1: weight {self.fc1_weight.shape}, bias {self.fc1_bias.shape}, scale {self.fc1_scale:.6f}")
        self._log(f"FC2: weight {self.fc2_weight.shape}, bias {self.fc2_bias.shape}, scale {self.fc2_scale:.6f}")
        self._log("Initialization complete")

    def _log(self, msg):
        if self.verbose:
            print(f"[TiledInference] {msg}")

    #load state dict from pth file
    def _load_state_dict(self, model_path):
        state_dict = torch.load(model_path, map_location='cpu')

        #handle nested state dict formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']

        return state_dict

    #extract tensor from state dict
    def _get_tensor(self, state_dict, key):
        if key in state_dict:
            return state_dict[key].numpy()

        #try with module prefix from DataParallel
        module_key = f'module.{key}'
        if module_key in state_dict:
            return state_dict[module_key].numpy()

        available = list(state_dict.keys())
        raise KeyError(f"Key '{key}' not found. Available: {available}")

    #compute 2x2 tile in int32 (matches hardware PE array)
    def matmul_2x2_int32(self, weight_tile, input_tile):
        w = weight_tile.astype(np.int32)
        x = input_tile.astype(np.int32)

        #out[0] = w[0,0]*in[0] + w[0,1]*in[1]
        #out[1] = w[1,0]*in[0] + w[1,1]*in[1]
        return np.array([
            w[0, 0] * x[0] + w[0, 1] * x[1],
            w[1, 0] * x[0] + w[1, 1] * x[1]
        ], dtype=np.int32)

    #matrix-vector multiply using 2x2 tiles with int32 accumulator
    def tiled_matmul_int32(self, weights, inputs):
        out_dim, in_dim = weights.shape

        #pad to even dimensions
        out_padded = out_dim + (out_dim % 2)
        in_padded = in_dim + (in_dim % 2)

        #create padded arrays
        weights_pad = np.zeros((out_padded, in_padded), dtype=np.int8)
        weights_pad[:out_dim, :in_dim] = weights

        inputs_pad = np.zeros(in_padded, dtype=np.int8)
        inputs_pad[:in_dim] = inputs

        #accumulate in int32 (matches hardware)
        accum = np.zeros(out_padded, dtype=np.int32)

        #process 2x2 tiles
        for o in range(0, out_padded, 2):
            for i in range(0, in_padded, 2):
                weight_tile = weights_pad[o:o+2, i:i+2]
                input_tile = inputs_pad[i:i+2]

                partial = self.matmul_2x2_int32(weight_tile, input_tile)
                accum[o:o+2] += partial

        return accum[:out_dim]

    #quantize to int4 range [-8, 7]
    def quantize_int4(self, x):
        return np.clip(np.round(x), -8, 7).astype(np.float32)

    #leaky relu with alpha=0.25, then quantize
    def leaky_relu_int4(self, x):
        #x if x >= 0, else x * 0.25
        activated = np.where(x >= 0, x, x * 0.25)
        return np.clip(np.round(activated), -8, 7).astype(np.float32)

    #compute fc layer using hardware-equivalent tiled matmul
    def fc_layer(self, inputs, weights, bias, scale, apply_relu=True):
        #ensure inputs are int8
        inputs_int = np.clip(np.round(inputs), -8, 7).astype(np.int8)

        #step 1: integer tiled matmul (hardware behavior)
        accum = self.tiled_matmul_int32(weights, inputs_int)

        #step 2: scale and bias (software/float)
        output = accum.astype(np.float32) * scale + bias

        #step 3: activation and quantization
        if apply_relu:
            output = self.leaky_relu_int4(output)

        return output

    #preprocess 14x14 image to int4
    def preprocess_image(self, image):
        #matches qat_model.py: x = quantize_int4(x * 15 - 8)
        x = image.flatten().astype(np.float32)
        x = x * 15.0 - 8.0
        x = self.quantize_int4(x)
        return x

    #run complete forward pass
    def forward(self, image):
        #preprocess
        x = self.preprocess_image(image)
        self._log(f"Preprocessed: shape={x.shape}, range=[{x.min()}, {x.max()}]")

        #fc1 + relu + quantize
        x = self.fc_layer(x, self.fc1_weight, self.fc1_bias, self.fc1_scale, apply_relu=True)
        self._log(f"After FC1: shape={x.shape}, range=[{x.min()}, {x.max()}]")

        #fc2 (no relu on output layer)
        x = self.fc_layer(x, self.fc2_weight, self.fc2_bias, self.fc2_scale, apply_relu=False)
        self._log(f"After FC2: shape={x.shape}, range=[{x.min():.2f}, {x.max():.2f}]")

        return x

    #predict digit class for an image
    def predict(self, image):
        logits = self.forward(image)
        prediction = int(np.argmax(logits))
        return prediction, logits

    #evaluate accuracy on dataset
    def evaluate(self, images, labels, max_samples=None):
        if max_samples is not None:
            images = images[:max_samples]
            labels = labels[:max_samples]

        total = len(labels)
        correct = 0

        for i in range(total):
            pred, _ = self.predict(images[i])
            if pred == labels[i]:
                correct += 1

            if (i + 1) % 1000 == 0:
                acc = 100.0 * correct / (i + 1)
                print(f"Progress: {i+1}/{total}, Accuracy: {acc:.2f}%")

        accuracy = correct / total
        return accuracy, correct, total


#get default paths relative to this script
def get_default_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_dir = os.path.join(script_dir, '..', '..', 'software', 'model', 'weights')
    model_path = os.path.join(weights_dir, 'model_best.pth')
    data_dir = os.path.join(script_dir, '..', '..', 'software', 'data')

    return os.path.abspath(weights_dir), os.path.abspath(model_path), os.path.abspath(data_dir)


def main():
    import argparse

    weights_dir, model_path, data_dir = get_default_paths()

    parser = argparse.ArgumentParser(description='uTPU Tiled Inference')
    parser.add_argument('--weights', type=str, default=weights_dir)
    parser.add_argument('--model', type=str, default=model_path)
    parser.add_argument('--data', type=str, default=data_dir)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--sample', type=int, default=None)
    parser.add_argument('--num-samples', type=int, default=None)
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    print("=" * 60)
    print("uTPU Tiled Inference Engine")
    print("=" * 60)

    #initialize engine
    try:
        engine = TiledInferenceEngine(args.weights, args.model, verbose=args.verbose)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure you have trained and exported:")
        print("  1. python software/model/train.py")
        print("  2. python software/model/export_weights.py")
        return 1

    #load test data
    test_images_path = os.path.join(args.data, 'mnist_14x14_test.npy')
    test_labels_path = os.path.join(args.data, 'test_labels.npy')

    if not os.path.exists(test_images_path):
        print(f"Error: Test images not found at {test_images_path}")
        return 1

    test_images = np.load(test_images_path)
    test_labels = np.load(test_labels_path)

    print(f"Loaded {len(test_labels)} test samples")

    if args.sample is not None:
        #single sample inference
        idx = args.sample
        pred, logits = engine.predict(test_images[idx])
        actual = test_labels[idx]

        print(f"\nSample {idx}:")
        print(f"  Predicted: {pred}")
        print(f"  Actual:    {actual}")
        print(f"  Result:    {'CORRECT' if pred == actual else 'WRONG'}")

    elif args.eval:
        #full evaluation
        print(f"\nEvaluating...")
        accuracy, correct, total = engine.evaluate(test_images, test_labels, args.num_samples)

        print("\n" + "=" * 60)
        print(f"ACCURACY: {100*accuracy:.2f}% ({correct}/{total})")
        print("=" * 60)

    else:
        #quick demo on first 10
        print("\nRunning on first 10 samples:")
        correct = 0
        for i in range(10):
            pred, _ = engine.predict(test_images[i])
            actual = test_labels[i]
            status = "✓" if pred == actual else "✗"
            if pred == actual:
                correct += 1
            print(f"  [{i}] Predicted: {pred}, Actual: {actual} {status}")

        print(f"\nQuick test: {correct}/10 correct")
        print("\nRun with --eval for full evaluation")

    return 0


if __name__ == "__main__":
    sys.exit(main())
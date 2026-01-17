import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../software/model'))

from tiled_inference import TiledInferenceEngine, get_default_paths


#verify tiled inference matches pytorch model
def verify():
    print("=" * 60)
    print("VERIFICATION: Tiled Inference vs PyTorch")
    print("=" * 60)

    weights_dir, model_path, data_dir = get_default_paths()

    #check required files exist
    required_files = [
        (os.path.join(weights_dir, 'fc1_weight.npy'), "FC1 weights"),
        (os.path.join(weights_dir, 'fc2_weight.npy'), "FC2 weights"),
        (os.path.join(weights_dir, 'scales.npy'), "Scale factors"),
        (model_path, "Trained model"),
        (os.path.join(data_dir, 'mnist_14x14_test.npy'), "Test images"),
        (os.path.join(data_dir, 'test_labels.npy'), "Test labels"),
    ]

    print("\n1. Checking required files...")
    missing = []
    for path, name in required_files:
        if os.path.exists(path):
            print(f"   ✓ {name}")
        else:
            print(f"   ✗ {name} NOT FOUND: {path}")
            missing.append(name)

    if missing:
        print(f"\nERROR: Missing files. Run these first:")
        print("  1. python software/preprocesses/downscale.py")
        print("  2. python software/model/train.py")
        print("  3. python software/model/export_weights.py")
        return False

    #load pytorch model
    print("\n2. Loading PyTorch model...")
    from qat_model import MNISTNet
    pytorch_model = MNISTNet()
    pytorch_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    pytorch_model.eval()
    print("   ✓ PyTorch model loaded")

    #load tiled inference engine
    print("\n3. Loading Tiled Inference Engine...")
    engine = TiledInferenceEngine(weights_dir, model_path, verbose=False)
    print("   ✓ Tiled engine loaded")

    #load test data
    print("\n4. Loading test data...")
    test_images = np.load(os.path.join(data_dir, 'mnist_14x14_test.npy'))
    test_labels = np.load(os.path.join(data_dir, 'test_labels.npy'))
    print(f"   ✓ Loaded {len(test_labels)} test samples")

    #compare outputs on subset
    print("\n5. Comparing outputs on 100 samples...")
    num_compare = 100
    max_diff = 0.0
    prediction_matches = 0

    for i in range(num_compare):
        image = test_images[i]

        #pytorch forward
        with torch.no_grad():
            pt_input = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
            pt_output = pytorch_model(pt_input).numpy()[0]

        #tiled forward
        tiled_output = engine.forward(image)

        #compare
        diff = np.abs(pt_output - tiled_output).max()
        max_diff = max(max_diff, diff)

        if np.argmax(pt_output) == np.argmax(tiled_output):
            prediction_matches += 1

    print(f"   Max output difference: {max_diff:.6f}")
    print(f"   Prediction matches: {prediction_matches}/{num_compare}")

    if prediction_matches == num_compare:
        print("   ✓ All predictions match!")
    else:
        print(f"   ⚠ {num_compare - prediction_matches} predictions differ")

    #full accuracy evaluation
    print("\n6. Evaluating full test set...")

    #pytorch accuracy
    pt_correct = 0
    with torch.no_grad():
        for i in range(len(test_labels)):
            pt_input = torch.tensor(test_images[i], dtype=torch.float32).unsqueeze(0)
            pt_output = pytorch_model(pt_input)
            if pt_output.argmax().item() == test_labels[i]:
                pt_correct += 1
    pt_acc = 100.0 * pt_correct / len(test_labels)

    #tiled accuracy
    tiled_acc, tiled_correct, total = engine.evaluate(test_images, test_labels)
    tiled_acc *= 100

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"PyTorch accuracy:         {pt_acc:.2f}% ({pt_correct}/{len(test_labels)})")
    print(f"Tiled inference accuracy: {tiled_acc:.2f}% ({tiled_correct}/{total})")
    print(f"Difference:               {abs(pt_acc - tiled_acc):.4f}%")
    print("=" * 60)

    if abs(pt_acc - tiled_acc) < 1.0:
        print("\n✓ VERIFICATION PASSED")
        print("  Ready for FPGA deployment.")
        return True
    else:
        print("\n✗ VERIFICATION FAILED")
        return False


if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)

import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.qat_model import MNISTNet

def extract_int4_weights(model):
    weights = dict()
    for name, layer in [("fc1", model.fc1), ("fc2", model.fc2)]:

        #learned scale factor
        scale = layer.scale.item() #float
        #weight_matrix
        w = layer.weight.data.clone()

        #quantize
        w_scaled = w/scale
        w_rounded = torch.round(w_scaled)
        w_clamped = torch.clamp(w_rounded, -8, 7)

        #convert to numpy int8 (numpy doesn't have smaller)
        w_int4 = w_clamped.numpy().astype(np.int8)

        w_int4 = w_clamped.numpy().astype(np.int8)

        #bias logic removed (hardware mismatch)

        weights[f'{name}_weight'] = w_int4
        weights[f'{name}_scale'] = scale

        print(f"\n{name} layer:")
        print(f"  Weight shape: {w_int4.shape}")
        print(f"  Weight range: [{w_int4.min()}, {w_int4.max()}]")
        print(f"  Scale factor: {scale:.4f}")
    
    return weights

def int4_to_bytes(int4_array):
    
    #flatten array
    flattened = int4_array.flatten()

    if len(flattened) % 2 == 1:
        flattened = np.append(flattened, np.int8(0))

    packed = []
    for i in range(0, len(flattened), 2):
        low = flattened[i]
        high = flattened[i+1]

        #mask to 4 bits
        low_nibble = int(low) & 0x0F
        high_nibble = int(high) & 0x0F

        packed_byte = (high_nibble << 4) | low_nibble
        packed.append(packed_byte)
    return bytes(packed)

#save weights to binary
def weights_to_binary(weights, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    for name in ['fc1_weight', 'fc2_weight']:
        data = weights[name]
        packed = int4_to_bytes(data)
        filepath = f'{output_dir}/{name}.bin'
        with open(filepath, "wb") as f:
            f.write(packed)
        
        print(f"Saved {filepath}: {len(packed)} bytes (from {data.size} values)")
    
    np.save(f'{output_dir}/fc1_weight.npy', weights['fc1_weight'])
    np.save(f'{output_dir}/fc2_weight.npy', weights['fc2_weight'])
    scales = {
        'fc1_scale': weights['fc1_scale'],
        'fc2_scale': weights['fc2_scale']
    }
    np.save(f'{output_dir}/scales.npy', scales)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(script_dir, 'weights', 'model_best.pth')
    OUTPUT_DIR = os.path.join(script_dir, 'weights')

    #check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train.py first!")
        return
    
    print(f"Loading model from {MODEL_PATH}...")
    model = MNISTNet()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    #quantize
    print("\nExtracting and quantizing weights...")
    weights = extract_int4_weights(model)

    print("\nSaving binary files...")
    weights_to_binary(weights, OUTPUT_DIR)

    print("\n" + "="*50)
    print("Export complete!")
    print("Binary files ready for uTPU:")
    print(f"  {OUTPUT_DIR}/fc1_weight.bin")
    print(f"  {OUTPUT_DIR}/fc2_weight.bin")
    print("="*50)


if __name__ == '__main__':
    main()


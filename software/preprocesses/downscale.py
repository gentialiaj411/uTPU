import numpy as np
from torchvision import datasets
import os 


#downscales 28x28 images to 14x14 by taking average of each 2x2 subaraea
def downscale(images_28x28):
    N = images_28x28.shape[0] #number of images, .shape() returns (numImages, rows per image, cols per image)
    images_14x14 = np.zeros((N, 14, 14), dtype=np.float32) #initialize 14x14 matrix for each image, w/ type 32 bit
    for row in range(14):
        for col in range(14):
            block = images_28x28[:,row*2:row*2+2, col*2:col*2+2]
            images_14x14[:, row, col] = block.mean(axis=(1, 2)) #avg over rows and columns, axis = 0 would be N (numImages)
    return images_14x14

#converts floating-point images to 4-bit signed ints
def quantize(images_float):
    scaled = images_float * 15 - 8 #this is how u convert fp to 4-bit signed int [-8, 7]
    rounded = np.round(scaled) #rounds decimals to ints
    clamped = np.clip(rounded, -8, 7) #values less than -8 become -8 and values greater than 7 become 7
    return clamped.astype(np.int8) #numpy doesn't support 4-bit, so we use 8-bit and only use half of them


def main():
    #create output directory
    output_dir = '../data'
    os.makedirs(output_dir, exist_ok=True)

    #download MNIST dataset
    train_dataset = datasets.MNIST('./mnist_raw', train=True, download=True)
    test_dataset = datasets.MNIST('./mnist_raw', train=False, download=True) 


    #convert tensors to np arrays
    #train_dataset.data is a tensor of shape (60000, 28, 28)
    train_images_raw = train_dataset.data.numpy() #(60000, 28, 28), uint8
    test_images_raw = test_dataset.data.numpy() #(10000, 28, 28), uint8

    #train_dataset.targets is a tensor of shape (60000, 28, 28) with values 0-9
    train_labels = train_dataset.targets.numpy() #(60000, 28, 28) int64
    test_labels = test_dataset.targets.numpy() #(10000, 28, 28), int64

    #normalize 0-255 to 0-1
    print("Normalizing images...")
    train_images_float = train_images_raw.astype(np.float32) / 255.0
    test_images_float = test_images_raw.astype(np.float32) / 255.0

    #downscale 28x28 to 14x14
    train_images_14x14 = downscale(train_images_float)
    test_images_14x14 = downscale(test_images_float)

    #save as .npy files
    np.save(f'{output_dir}/mnist_14x14_train.npy', train_images_14x14)
    np.save(f'{output_dir}/mnist_14x14_test.npy', test_images_14x14)
    np.save(f'{output_dir}/train_labels.npy', train_labels)
    np.save(f'{output_dir}/test_labels.npy', test_labels)

    print(f"\nDataset prepared succesfully!")
    print(f"Training images: {train_images_14x14.shape}") #shape: (60000, 14, 14)
    print(f"Test images: {test_images_14x14.shape}") #shape: (10000, 14, 14)
    print(f"Pixel value range: [{train_images_14x14.min():.3f}, {train_images_14x14.max():.3f}]")

if __name__ == "__main__":
    main()
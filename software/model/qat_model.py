import torch
import torch.nn as nn
import torch.nn.functional as F

class Int4Quantize(torch.autograd.Function):
    #quantizes to int4 for forward pass, unchanged for backward pass
    @staticmethod
    def forward(ctx, x):
        #quantize input to int4 range
        return torch.clamp(torch.round(x), -8, 7)

    @staticmethod
    def backward(ctx, grad_output):
        #pass gradient unchanged
        return grad_output

def quantize_int4(x):
    #quantize tensor to int4
    return Int4Quantize.apply(x)


class QATLinear(nn.Module):
    #layer w/ quantized weights
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(out_features, in_features)*2)
        self.bias= nn.Parameter(torch.zeros(out_features))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        w_quant = quantize_int4(self.weight/self.scale)*self.scale
        return F.linear(x, w_quant, self.bias)

class MNISTNet(nn.Module):
    #neural network

    def __init__(self):
        super().__init__()

        #input: 14x14, output: 16 hidden neurons
        self.fc1 = QATLinear(196, 16)

        #input: 16 (from prev layer), output: 10 (one score per digit 0-9)
        self.fc2 = QATLinear(16, 10)

    def forward(self, x):
        
        #shape before: (64, 14, 14), shape after: (64, 196)
        x = x.view(-1, 196)

        #formula to quantize to int4
        x = quantize_int4(x*15-8)

        #first layer
        x = self.fc1(x)

        #apply ReLU activation
        x = F.leaky_relu(x, negative_slope=0.25)

        #quantize to int4
        x = quantize_int4(x)

        #second layer
        x = self.fc2(x)

        return x
    
if __name__ == "__main__":
    model = MNISTNet()
    print("Model Architecture")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params}")
    test_input = torch.randn(2, 14, 14)
    test_output = model(test_input)
    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")  # Should be (2, 10)

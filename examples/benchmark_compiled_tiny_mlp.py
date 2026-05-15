import os
import sys

import torch
import torch.nn as nn


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HOST_DIR = os.path.join(REPO_ROOT, "firmware", "host")
if HOST_DIR not in sys.path:
    sys.path.insert(0, HOST_DIR)

from pytorch_compiler import compile_mlp_model


class TinyIntegerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                    ]
                )
            )
            self.fc2.weight.copy_(torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]))

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def main():
    model = TinyIntegerMLP().eval()
    x = torch.tensor([[1.0, 2.0, 0.0, 0.0]])
    compiled = compile_mlp_model(model, x, target="cuda")
    bench = compiled.benchmark(x, warmup=5, iters=50)

    print("Compiled tiny MLP benchmark")
    print("===========================")
    print(f"runtime_device={bench['device']}")
    print(f"compile_time_ms={bench['compile_time_ms']:.4f}")
    print(f"setup_time_ms={bench['setup_time_ms']:.4f}")
    print(f"first_call_wall_ms={bench['first_call_wall_ms']:.4f}")
    print(f"steady_state_wall_ms={bench['steady_state_wall_ms']:.4f}")
    print(f"h2d_time_ms={bench['h2d_time_ms']:.4f}")
    print(f"kernel_time_ms={bench['kernel_time_ms']:.4f}")
    print(f"d2h_time_ms={bench['d2h_time_ms']:.4f}")
    print(f"h2d_count={bench.get('h2d_count')}")
    print(f"d2h_count={bench.get('d2h_count')}")
    print(f"adapter_time_ms={bench['adapter_time_ms']:.4f}")
    print(f"backend_linear_ops_executed={bench['backend_linear_ops_executed']}")
    print(f"backend_elementwise_ops_executed={bench['backend_elementwise_ops_executed']}")
    print(f"fallback_ops={bench['fallback_ops']}")
    print(f"max_abs_error={bench['max_abs_error_vs_pytorch']:.8f}")
    print("last_op_traces:")
    for trace in bench["last_op_traces"]:
        print(
            f"- op={trace['graph_op']} engine={trace['engine']} "
            f"latency_ms={trace['latency_ms']:.4f} notes={trace['notes']}"
        )


if __name__ == "__main__":
    main()

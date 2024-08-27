import torch
import argparse
import habana_frameworks.torch.core as htcore
parser = argparse.ArgumentParser()

parser.add_argument(
    "--device",
    default="hpu",
    type=str,
    help="Path to pre-trained model",
)

parser.add_argument(
    "--dtype",
    default=torch.float32,
    type=str,
    help="Path to pre-trained model",
)


args = parser.parse_args()
dtype = args.dtype
if args.dtype == 'bf16':
    dtype = torch.bfloat16
torch.manual_seed(0)
x = torch.rand(2,3,4,5)

print(x)
# FFT
x = x.to(device=torch.device(args.device), dtype=dtype)
x_freq = torch.fft.fftn(x, dim=(-2, -1))
print(x_freq)
x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
print(x_freq)

# IFFT
x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
print(x_freq)
x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
print(x_filtered)
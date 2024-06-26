import torch
from transformers import set_seed
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', '-dev', type=str, default='xpu', help='Device type to use: cpu, xpu,cuda')
args = parser.parse_args()
if args.device == "xpu":
    import intel_extension_for_pytorch as ipex

device = f"{args.device}:0"

set_seed(42)

for src_type in [torch.float, torch.float16]:
    weights = torch.tensor([0, 10, 3, 0], dtype=src_type, device=device) # create a tensor of weights

    a_float = torch.rand((4, 4), dtype=src_type, device=device)
    print(f"a_float = {a_float}")
    b_float = torch.rand((4, 4), dtype=src_type, device=device)
    print(f"b_float = {b_float}")
    c_float = torch.rand((4, 4), dtype=src_type, device=device)
    print(f"c_float = {c_float}")
    d_float = torch.rand((4, 4), dtype=src_type, device=device) 
    print(f"d_float = {d_float}")
    x_cat = torch.load('tensor.pt')
    #print(f"x={x_cat}")
    for dst_type in [torch.bfloat16, torch.float16] :
        print(f"#################{src_type}>>>>>>>>>{dst_type}################")
        with torch.autocast(device_type=args.device, dtype=dst_type):
            # torch.mm is on autocast's list of ops that should run in float16.
            # Inputs are float32, but the op runs in float16 and produces float16 output.
            # No manual casts are required.
            e_float = torch.mm(a_float, b_float)
            print(f"e_float = {e_float}")
            assert e_float.dtype == dst_type
            # Also handles mixed input types
            f_float = torch.mm(d_float, e_float)
            print(f"f_float = {f_float}")
            assert e_float.dtype == dst_type
            h_float = torch.cat((e_float, f_float), 0)
            print(f"h_float = {h_float}")
            assert h_float.dtype == dst_type
            if args.device == "xpu":
                print(f"is_xpu_autocast_enabled={torch.xpu.is_autocast_enabled()}")
            print(f"is_autocast_enabled={torch.is_autocast_enabled()}")
            print(f"get_autocast_dtype={torch.get_autocast_gpu_dtype()}")
            out = torch.multinomial(weights, 2)
            print(f"out={out}")
            #torch.multinomial(weights, 4) # ERROR!
            out2 = torch.multinomial(weights, 4, replacement=True)
            print(f"out={out2}")
            # d1 =  d_float[..., : d_float.shape[-1] // 2]
            # print(f"d1={d1}")
            # d2 = d_float[..., d_float.shape[-1] // 2 :]
            # print(f"d2={d2}")
            # d_cat = torch.cat((-d2, d1), dim=-1)
            # print(f"d_cat={d_cat}")
            x1 = x_cat[..., : x_cat.shape[-1] // 2]
            #print(f"x1={x1}")
            x2 = x_cat[..., x_cat.shape[-1] // 2 :]
            #print(f"x2={x2}")
            torch.cat((-x2, x1), dim=-1)


# After exiting autocast, calls f_float.float() to use with d_float32
print(f"d_float type = {d_float.dtype}")
g_float = torch.mm(d_float, f_float)
print(f"g_float type = {g_float.dtype}")

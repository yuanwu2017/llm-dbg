import torch
import torch.distributed as dist
import os
import argparse
from typing import List
def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default




print("ccl_test start")
parser = argparse.ArgumentParser()
parser.add_argument('--device', '-dev', type=str, default='xpu', help='Device type to use: cpu, xpu')
parser.add_argument('--launch', '-l', type=str, default='torch', help='launcher type to use: torch, mpi')
args = parser.parse_args()

rank = get_int_from_env(["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], 0)
size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1)
local_rank = get_int_from_env(
    ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"], 0
)
local_size = get_int_from_env(
    ["LOCAL_WORLD_SIZE", "MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"],
    1,
)


os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29502'

if args.launch =='torch' :
    os.environ["CCL_PROCESS_LAUNCHER"] = "none"
    os.environ["CCL_LOCAL_SIZE"] = str(local_size)
    os.environ["CCL_LOCAL_RANK"] = str(local_rank)
    

if args.device == 'xpu':
    import intel_extension_for_pytorch
    import oneccl_bindings_for_pytorch
    device = f"xpu:{local_rank}"
    backend='ccl'
    torch.xpu.set_device(local_rank)
elif args.device=='cuda':
    device= f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    backend='nccl'
else:
    import intel_extension_for_pytorch
    import oneccl_bindings_for_pytorch
    device = 'cpu'
    backend='ccl'



print("ccl_test init_process_group device={device}, backend={backend}")
group = dist.init_process_group(backend, rank=rank, world_size=size)
print("ccl_test init_process_group done")
my_rank = dist.get_rank()
my_size = dist.get_world_size()
print("my rank = %d  my size = %d" % (my_rank, my_size))
x = torch.ones([2, 2], device=device)
y = torch.ones([4, 4], device=device)

with torch.autograd.profiler.profile(record_shapes=True) as prof:
    for _ in range(10):
        dist.all_reduce(x)
        dist.all_reduce(y)
        dist.broadcast(x, src=0)
        dist.broadcast(y, src=0)
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))

dist.barrier()

print(f"broadcast and reduce test done!!!!")
target = torch.arange(60, dtype=torch.float16, device=device).chunk(5)
target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
tensors = [tensor.clone() for tensor in target]
process_group = dist.distributed_c10d._get_default_group()
with torch.autograd.profiler.profile(record_shapes=True) as prof:
        try:
           dist._broadcast_coalesced(process_group, tensors, buffer_size=256, src=0)
        except Exception as e:
           print(f"exception={e}")
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))
dist.barrier()
print(f"broadcast_coalesced test done!!!!")
tensors = [torch.tensor(rank, device=device),torch.tensor(-10000., device=device), torch.tensor([[True, False, True, False],[True, False, False, False]], device=device)]
#tensors = [torch.tensor(rank, device=device),torch.tensor(-10000., device=device)]
with torch.autograd.profiler.profile(record_shapes=True) as prof:
        try:
           output = dist._broadcast_coalesced(process_group, tensors, buffer_size=256, src=0)
        except Exception as e:
           print(f"exception={e}")
print(f"output={tensors}")
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))


from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    "lvwerra/gpt2-imdb",
    trust_remote_code=True,
)

model.to(device)
module_states: List[torch.Tensor] = []
for name, param in model.named_parameters():
    module_states.append(param.detach())

for name, buffer in model.named_buffers():
    module_states.append(buffer.detach())
dist._broadcast_coalesced(process_group, module_states, buffer_size=262144000, src=0)
print(f"test done!")    

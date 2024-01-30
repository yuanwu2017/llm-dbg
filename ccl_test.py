import torch
import torch.distributed as dist
import os
import argparse
from typing import List
import traceback
def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
        
print_rank_0("ccl_test start")
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



print_rank_0("ccl_test init_process_group device={device}, backend={backend}")
group = dist.init_process_group(backend, rank=rank, world_size=size)
print_rank_0("ccl_test init_process_group done")
my_rank = dist.get_rank()
my_size = dist.get_world_size()
print_rank_0("my rank = %d  my size = %d" % (my_rank, my_size))
x = torch.ones([2, 2], device=device)
y = torch.ones([4, 4], device=device)

with torch.autograd.profiler.profile(record_shapes=True) as prof:
    for _ in range(10):
        dist.all_reduce(x)
        dist.all_reduce(y)
        dist.broadcast(x, src=0)
        dist.broadcast(y, src=0)
print_rank_0(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))
print_rank_0(f"broadcast and reduce test done!!!!")

dist.barrier()

target = torch.arange(60, dtype=torch.float16, device=device).chunk(5)
target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
tensors = [tensor.clone() for tensor in target]
process_group = dist.distributed_c10d._get_default_group()
try:
    with torch.autograd.profiler.profile(record_shapes=True) as prof:
        dist._broadcast_coalesced(process_group, tensors, buffer_size=256, src=0)
    print_rank_0(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))
    print_rank_0(f"broadcast_coalesced float test done!!!!")
except Exception as e:
           traceback.print_exc()
           print(f"float value test failed!!!!!!!!!!!!!!!!")
dist.barrier()
tensors = [torch.tensor([[True, False, True, False],[True, False, False, False]], device=device)]
#tensors = [torch.tensor(rank, device=device),torch.tensor(-10000., device=device)]
try:
    with torch.autograd.profiler.profile(record_shapes=True) as prof:
            output = dist._broadcast_coalesced(process_group, tensors, buffer_size=256, src=0)
    print_rank_0(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))
    print(f"bool value test done!")
except :
    traceback.print_exc()
    print(f"bool value test failed!!!!!!!!!!!!!!!!")

dist.barrier()

print_rank_0(f"############################################# model synce test #########################################################################")
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
try:
    with torch.autograd.profiler.profile(record_shapes=True) as prof:
        for _ in range(10):
            dist._broadcast_coalesced(process_group, module_states, buffer_size=262144000, src=0)
    print_rank_0(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))
    print_rank_0(f"model sync test done!")    
except :
    traceback.print_exc()
    print_rank_0(f"model sync test failed!")    

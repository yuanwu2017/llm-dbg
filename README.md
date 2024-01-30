


## Run test with accelerate on XPU/GPU 
### run config
```bash
accelerate config
```
### Run test command

#### XPU:
```bash
CCL_ZE_IPC_EXCHANGE=sockets accelerate launch --main_process_port=29502 ccl_test.py --device xpu --launch torch
```
#### GPU:
```bash
accelerate launch --main_process_port=29502  ccl_test.py --device cuda 2>&1 |tee nvidia.log
```

## Run test with mpi on XPU 
```bash
CCL_ZE_IPC_EXCHANGE=sockets mpirun -n 2 -l  python ccl_test.py --device xpu --launch mpi
```


 
### From https://medium.com/@manindersingh120996/multi-gpu-model-training-made-easy-with-distributed-data-parallel-ddp-453ba9f6846e

"""
1. Why Use DDP?
   -a. Handling Large Models:
       - A single GPU often lacks sufficient memory or computational power for training large models. 
         DDP aggregates resources across GPUs to accommodate such tasks.
   -b. Reducing Training Time:
       - By distributing computations across GPUs, DDP accelerates the training process.
   -c. Enabling Scalability:
       - DDP scales training to multiple GPUs and nodes seamlessly, supporting growing datasets and model sizes.

2. Advantages of DDP Over Data Parallel (DP)
   -a. Multi-Process Architecture:
       - DP operates with a single process using multiple threads, which suffers from Global Interpreter Lock (GIL) contention, 
         slowing computation.
       - DDP eliminates GIL contention by assigning one process per GPU, significantly improving efficiency.

   -b. Scalability Across Machines:
       - DP is limited to a single machine and cannot utilize GPUs across multiple systems.
       - DDP supports both single-machine and multi-machine setups, making it more scalable for large-scale distributed training.

   -c. Compatibility with Model Parallelism:
       - DP cannot combine model parallelism with data parallelism.
       - DDP integrates seamlessly with model parallelism, efficiently distributing massive models across GPUs and processes.

3. How DDP Works Internally
   -a. Model Initialization and Replication:
       -1. Process Group Setup: DDP uses a ProcessGroup for inter-process communication (e.g., gloo, nccl).
       -2. Model Broadcast: Parameters are synchronized across all replicas by broadcasting the state_dict from rank 0.
       -3. Reducer Creation: Gradients are grouped into buckets for optimized communication, configurable via bucket_cap_mb.

   -b. Forward Pass:
       -1. Each GPU processes its mini-batch independently using its local model replica.
       -2. With find_unused_parameters=True, DDP tracks unused parameters to avoid unnecessary gradient synchronization 
           (at the cost of overhead).

   -c. Backward Pass and Gradient Synchronization:
       -1. Autograd Hooks: DDP registers hooks to synchronize gradients during the backward pass.
       -2. Bucketed Gradient Reduction:
           - Gradients are grouped into buckets and reduced using asynchronous allreduce operations.
           - Averaged gradients are written back to the corresponding parameters after reduction.

   -d. Optimizer Step:
       The optimizer updates parameters using the synchronized gradients, ensuring all model replicas remain consistent
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# Initializes a distributed environment for training.
# # On Windows platform, the torch.distributed package only
# supports Gloo backend, FileStore and TcpStore.
# For FileStore, set init_method parameter in init_process_group
# to a local file. Example as follow:
# init_method="file:///f:/libtmp/some_file"
# dist.init_process_group(
#    "gloo",
#    rank=rank,
#    init_method=init_method,
#    world_size=world_size)
def setup(rank, world_size):
    torch.distributed.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

# Here we have Created a very simple Neural Network
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 1)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

# creatging the sample datset for testing
class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.data = torch.randn(length, size)
        self.labels = torch.randn(length, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# train loop in which setup is done 
# where world_size is the number of GPUs we want to access
# and rank is the current GPU of interest
def train(rank, world_size):
    setup(rank, world_size)
    torch.backends.cudnn.benchmark = True  # Optional performance optimization

    model = SimpleModel().to(rank)
# Converting each model TO DDP object
    model = DDP(model, device_ids=[rank])

    dataset = RandomDataset(10, 1000)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(5):
        sampler.set_epoch(epoch)  # Ensure proper shuffling
        for batch, (data, labels) in enumerate(dataloader):
            data, labels = data.to(rank), labels.to(rank)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch % 10 == 0 and rank == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch}, Loss {loss.item()}")

    torch.distributed.destroy_process_group()  # Graceful shutdown

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)

-----------------------------------------------------------------------------------------------
# Now after setting up DDP it is required and mandatory to wrap the model in DDP
if ddp:
    model = DDP(model, device_ids = [ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

# ========================

optimizer = raw_model.configure_optimizers(weight_decay = 0.1,
                                       learning_rate = 6e-4,
                                      device_type = device)

-----------------------------------------------------------------------------------------------
class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_process):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_process
        
        # at init load tokens from disk and store them in memory
        # with open('input.txt','r') as f:
        with open(runpod_absolute_path,'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        # making changes in below code to accomodate the DDP and MultiGPU training
        # data splitting
        self.current_position = self.B * self.T * self.process_rank # for each process it's batch will start at rank times B times T

    def next_batch(self):
        # as well as makinng the changes in below code to always load the data on corresponding GPU accordingly 
        # and current position is advanced in such a way that it get's diffent data from every other GPU always
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        # buf.to(dtype = torch.float16)
        x = (buf[:-1]).view(B,T) # inputs
        y = (buf[1:]).view(B,T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x,y



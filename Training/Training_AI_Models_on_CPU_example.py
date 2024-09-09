## From https://towardsdatascience.com/training-ai-models-on-cpu-3903adc9f388
## Just reference
"""
The post explores how AI/ML workloads, traditionally run on GPUs, can be optimized for CPUs, focusing specifically on Intel® Xeon® CPUs and the PyTorch framework.
As GPUs have become harder to acquire, developers are turning to alternative hardware like CPUs,
though CPUs are generally less suited to these tasks. 
The post demonstrates techniques to enhance CPU performance in ML training, particularly for training a ResNet-50 model,
and compares the effectiveness of various optimizations, such as batch size tuning, mixed precision, and memory formats.

Key Optimizations:
1. Batch Size Adjustment: Lower batch sizes seem to perform better on CPUs.
2. Data Loading with Multi-processes: Reducing workers might prevent performance backfire due to resource contention on CPUs.
3. Mixed Precision: Using lower precision datatypes like bfloat16 can improve performance without significantly affecting convergence.
4. Channels Last Memory Format: This optimization is especially beneficial for image models like ResNet-50 and yields significant speed improvements.
5. Torch Compilation: Compiling the model into intermediate machine code with PyTorch’s torch.compile can boost performance, 
                      though its effect is limited when other optimizations are already in place.
6. Intel Extensions: Tools like Intel's ipex.optimize further enhance CPU performance.
7. Distributed Training: Dividing CPU tasks across NUMA nodes can, in theory, improve performance, though the tested case saw no benefit.
"""

####### Base code
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import time

# Additional imports for specific optimizations
import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch as torch_ccl
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os

# Dataset definition
class FakeDataset(Dataset):
    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        rand_image = torch.randn([3, 224, 224], dtype=torch.float32)
        label = torch.tensor(data=index % 10, dtype=torch.uint8)
        return rand_image, label

# Base training loop
def base_training():
    train_set = FakeDataset()
    batch_size = 128
    num_workers = 0
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers
    )
    model = torchvision.models.resnet50()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters())
    model.train()
    
    t0 = time.perf_counter()
    summ = 0
    count = 0
    
    for idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        batch_time = time.perf_counter() - t0
        if idx > 10:  # skip first steps
            summ += batch_time
            count += 1
        t0 = time.perf_counter()
        if idx > 100:
            break
    
    print(f'average step time: {summ/count}')
    print(f'throughput: {count*batch_size/summ}')

# Mixed Precision optimization
def mixed_precision_training():
    # ... (same setup as base_training)
    for idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        with torch.amp.autocast('cpu', dtype=torch.bfloat16):
            output = model(data)
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # ... (same timing code)

# Channels Last Memory Format optimization
def channels_last_training():
    # ... (same setup as base_training)
    for idx, (data, target) in enumerate(train_loader):
        data = data.to(memory_format=torch.channels_last)
        optimizer.zero_grad()
        with torch.amp.autocast('cpu', dtype=torch.bfloat16):
            output = model(data)
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        # ... (same timing code)

# Torch Compilation
def torch_compile_training():
    # ... (same setup as base_training)
    backend = 'inductor'  # or 'ipex'
    model = torch.compile(model, backend=backend)
    # ... (rest of the training loop)

# Intel Extension for PyTorch optimization
def ipex_training():
    # ... (same setup as base_training)
    model, optimizer = ipex.optimize(
        model, 
        optimizer=optimizer,
        dtype=torch.bfloat16
    )
    # ... (rest of the training loop)

# Distributed Training
def distributed_training():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = os.environ.get("PMI_RANK", "0")
    os.environ["WORLD_SIZE"] = os.environ.get("PMI_SIZE", "1")
    dist.init_process_group(backend="ccl", init_method="env://")
    rank = os.environ["RANK"]
    world_size = os.environ["WORLD_SIZE"]

    train_dataset = FakeDataset()
    dist_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=dist_sampler
    )

    model = torchvision.models.resnet50()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters())
    model.train()
    model, optimizer = ipex.optimize(
        model, 
        optimizer=optimizer,
        dtype=torch.bfloat16
    )

    model = torch.nn.parallel.DistributedDataParallel(model)

    # ... (training loop)

    dist.destroy_process_group()

# PyTorch/XLA Training
def xla_training():
    import torch_xla
    import torch_xla.core.xla_model as xm

    device = xm.xla_device()
    model = torchvision.models.resnet50().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters())
    model.train()

    for idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        xm.mark_step()

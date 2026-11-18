import pycuda.driver as cuda
import pycuda.autoinit
import torch


def get_cuda_gpu_info():
    """Get the CUDA GPU info."""
    num_gpus = cuda.Device.count()
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()
    cudnn_version_str = (
        None
        if cudnn_version is None
        else f"{cudnn_version // 1000}.{(cudnn_version % 1000) // 100}"
    )

    # Get torch version
    torch_version = torch.__version__
    pycuda_info = {
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version_str,
        "num_gpus": num_gpus,
        "devices": {},
    }

    for i in range(num_gpus):
        device = cuda.Device(i)
        attributes = device.get_attributes()

        pycuda_info["devices"][f"GPU_{i}"] = {
            "name": device.name(),
            "compute_capability": f"{device.compute_capability()[0]}.{device.compute_capability()[1]}",
            "total_memory_GB": round(device.total_memory() / (1024**3), 2),
            "multiprocessors": attributes.get(
                cuda.device_attribute.MULTIPROCESSOR_COUNT, None
            ),
            "max_threads_per_block": attributes.get(
                cuda.device_attribute.MAX_THREADS_PER_BLOCK, None
            ),
            "max_threads_per_sm": attributes.get(
                cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR, None
            ),
            "shared_memory_per_block_KB": round(
                attributes.get(cuda.device_attribute.SHARED_MEMORY_PER_BLOCK, 0) / 1024,
                2,
            ),
            "L2_cache_size_KB": round(
                attributes.get(cuda.device_attribute.L2_CACHE_SIZE, 0) / 1024, 2
            ),
            "memory_bus_width_bits": attributes.get(
                cuda.device_attribute.GLOBAL_MEMORY_BUS_WIDTH, None
            ),
            "clock_rate_MHz": attributes.get(cuda.device_attribute.CLOCK_RATE, 0)
            / 1000,
            "memory_clock_rate_MHz": attributes.get(
                cuda.device_attribute.MEMORY_CLOCK_RATE, 0
            )
            / 1000,
            "warp_size": attributes.get(cuda.device_attribute.WARP_SIZE, None),
            "max_blocks_per_multiprocessor": attributes.get(
                cuda.device_attribute.MAX_BLOCKS_PER_MULTIPROCESSOR, None
            ),
        }

    return pycuda_info

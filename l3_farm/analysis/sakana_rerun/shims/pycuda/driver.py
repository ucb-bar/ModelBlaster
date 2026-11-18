import torch


class device_attribute:
    MULTIPROCESSOR_COUNT = "multi_processor_count"
    MAX_THREADS_PER_BLOCK = "max_threads_per_block"
    MAX_THREADS_PER_MULTIPROCESSOR = "max_threads_per_multi_processor"
    SHARED_MEMORY_PER_BLOCK = "shared_memory_per_block"
    L2_CACHE_SIZE = "L2_cache_size"
    GLOBAL_MEMORY_BUS_WIDTH = "memory_bus_width"
    CLOCK_RATE = "clock_rate"
    MEMORY_CLOCK_RATE = "memory_clock_rate"
    WARP_SIZE = "warp_size"
    MAX_BLOCKS_PER_MULTIPROCESSOR = "max_blocks_per_multi_processor"


class Device:
    def __init__(self, i):
        self.i = i
        self.p = torch.cuda.get_device_properties(i)

    @staticmethod
    def count():
        return torch.cuda.device_count()

    def name(self):
        return self.p.name

    def compute_capability(self):
        return (self.p.major, self.p.minor)

    def total_memory(self):
        return self.p.total_memory

    def get_attributes(self):
        return {k: getattr(self.p, k, None) for k in vars(device_attribute).values() if isinstance(k, str) and not k.startswith("__")}


def init():
    torch.cuda.init()

"""ctypes binding of the ModelBlaster Vulkan backend (vulkan/libmb_vk.so).

Device memory is host-mapped (mb_vk_alloc), so a buffer is a numpy array
viewing GPU-visible memory; kernels take its address. The library is found at
$MB_VK_LIB, else vulkan/build/libmb_vk.so next to this file.

    vk = VkBackend()               # raises VkUnavailable without a usable GPU
    x = vk.array((4, 8), np.float32); x[:] = ...
    vk.lib.kernel_relu(vk.ptr(x), vk.ptr(y), x.size); vk.sync()
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_LIB = HERE / "vulkan" / "build" / "libmb_vk.so"


class VkUnavailable(RuntimeError):
    """No libmb_vk.so, no Vulkan loader, or no suitable GPU."""


def lib_path() -> Path:
    return Path(os.environ.get("MB_VK_LIB", DEFAULT_LIB))


class VkBackend:
    def __init__(self, path: str | os.PathLike | None = None):
        p = Path(path) if path else lib_path()
        if not p.exists():
            raise VkUnavailable(f"{p} not built (python gpu_eval/harness/vulkan/build.py)")
        try:
            self.lib = ctypes.CDLL(str(p))
        except OSError as e:
            raise VkUnavailable(f"cannot load {p}: {e}") from e
        L = self.lib
        for name, res, args in [
            ("mb_vk_init", ctypes.c_int, []), ("mb_vk_device_name", ctypes.c_char_p, []),
            ("mb_vk_info", ctypes.c_char_p, []), ("mb_last_error", ctypes.c_int, []),
            ("mb_vk_error", ctypes.c_char_p, []), ("mb_vk_clear_error", None, []),
            ("mb_vk_alloc", ctypes.c_void_p, [ctypes.c_size_t]), ("mb_vk_mark", ctypes.c_size_t, []),
            ("mb_vk_release", None, [ctypes.c_size_t]), ("mb_vk_sync", None, []),
            ("mb_vk_devaddr", ctypes.c_uint64, [ctypes.c_void_p]),
            ("mb_vk_bytes_in_use", ctypes.c_size_t, []),
            ("mb_vk_timestamps", None, [ctypes.c_int]), ("mb_vk_ts", ctypes.c_int, []),
            ("mb_vk_ts_count", ctypes.c_int, []), ("mb_vk_ts_ns", ctypes.c_double, [ctypes.c_int]),
            ("mb_vk_ts_reset", None, []), ("mb_vk_dispatch_count", ctypes.c_long, []),
            ("mb_vk_capture_begin", ctypes.c_int, []), ("mb_vk_capture_end", ctypes.c_int, []),
            ("mb_vk_replay", None, [ctypes.c_int]), ("mb_vk_graph_cbs", ctypes.c_int, [ctypes.c_int]),
            ("mb_vk_set_packed", None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]),
            ("mb_vk_packed_count", ctypes.c_int, []),
        ]:
            if not hasattr(L, name):        # an older library
                continue
            f = getattr(L, name)
            f.restype, f.argtypes = res, args
        if L.mb_vk_init() != 0:
            raise VkUnavailable(f"mb_vk_init: {L.mb_vk_error().decode()}")
        self.device = L.mb_vk_device_name().decode()
        self.info = L.mb_vk_info().decode()

    # -- memory
    def array(self, shape, dtype=np.float32) -> np.ndarray:
        dt = np.dtype(dtype)
        n = int(np.prod(shape)) if len(tuple(np.atleast_1d(shape))) else 1
        nbytes = max(n * dt.itemsize, 1)
        addr = self.lib.mb_vk_alloc(nbytes)
        if not addr:
            raise MemoryError(f"mb_vk_alloc({nbytes}): {self.lib.mb_vk_error().decode()}")
        buf = (ctypes.c_char * nbytes).from_address(addr)
        return np.frombuffer(buf, dtype=dt, count=n).reshape(shape)

    def upload(self, a: np.ndarray, dtype=None) -> np.ndarray:
        a = np.ascontiguousarray(a, dtype=dtype or a.dtype)
        d = self.array(a.shape, a.dtype)
        d[...] = a
        return d

    @staticmethod
    def ptr(a: np.ndarray | None) -> int | None:
        return None if a is None else a.ctypes.data

    def mark(self) -> int:
        return self.lib.mb_vk_mark()

    def release(self, mark: int) -> None:
        self.lib.mb_vk_release(mark)

    # -- execution
    def sync(self) -> None:
        self.lib.mb_vk_sync()
        self.check()

    def replay(self, graph: int) -> None:
        self.lib.mb_vk_replay(graph)
        self.check("replay")

    def check(self, where: str = "") -> None:
        if self.lib.mb_last_error():
            msg = self.lib.mb_vk_error().decode()
            self.lib.mb_vk_clear_error()
            raise RuntimeError(f"vulkan error{(' in ' + where) if where else ''}: {msg}")

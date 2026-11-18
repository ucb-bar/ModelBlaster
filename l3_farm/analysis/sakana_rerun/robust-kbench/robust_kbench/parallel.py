from typing import List, Any, Dict, Optional
import multiprocessing as mp
import os
import json
import torch
import tempfile
import uuid
import shutil

# Import the evaluation functions from evaluate.py
from robust_kbench.primitives.evaluate import (
    compile_cuda_kernel,
    correct_cuda_kernel,
    eval_cuda_kernel,
    prof_cuda_kernel,
    eval_torch_runtime,
)


class ParallelKernelExecutor:
    """
    Class to execute compilation, testing, evaluation, and profiling
    of CUDA kernels in parallel. Each method accepts a list of CUDA kernel
    file paths and runs operations concurrently using multiprocessing with
    the spawn start method.
    """

    def __init__(
        self,
        task_dir: str,
        op_atol: float = 1e-5,
        op_rtol: float = 1e-5,
        num_correct_trials: int = 5,
        warmup_time: int = 25,
        repetition_time: int = 10000,
        eval_type: str = "kernelbench",
        multi_init_settings: bool = False,
        multi_input_settings: bool = False,
        timeout: int = 300,
        torch_prof: bool = False,
        ncu_prof: bool = False,
        clang_tidy: bool = False,
        forward: bool = True,
        config_fname: Optional[str] = None,
    ):
        self.task_dir = task_dir
        self.op_atol = op_atol
        self.op_rtol = op_rtol
        self.num_correct_trials = num_correct_trials
        self.warmup_time = warmup_time
        self.repetition_time = repetition_time
        self.eval_type = eval_type
        self.multi_init_settings = multi_init_settings
        self.multi_input_settings = multi_input_settings
        self.timeout = timeout
        self.torch_prof = torch_prof
        self.ncu_prof = ncu_prof
        self.clang_tidy = clang_tidy
        self.forward = forward  # to run forward or backward ops
        self.config_fname = config_fname

        # Determine the number of available GPUs.
        self.available_gpus = (
            torch.cuda.device_count() if torch.cuda.is_available() else 1
        )

        # Create a unique extension directory for each GPU.
        self.ext_dirs = self.setup_extension_dirs()

    def setup_extension_dirs(self) -> List[str]:
        ext_dirs = []
        for i in range(self.available_gpus):
            unique_id = str(uuid.uuid4())[:8]
            ext_dir = os.path.join(
                tempfile.gettempdir(), f"torch_extensions_{unique_id}"
            )
            os.makedirs(ext_dir, exist_ok=True)
            ext_dirs.append(ext_dir)
        return ext_dirs

    def cleanup_extension_dirs(self):
        for ext_dir in self.ext_dirs:
            try:
                shutil.rmtree(ext_dir)
            except Exception as e:
                print(f"Warning: Could not clean up directory {ext_dir}: {e}")

    def _cleanup_gpu(self, gpu_id: int):
        """Clean up GPU memory and processes for a specific GPU."""
        try:
            torch.cuda.set_device(gpu_id)
            # Delete all cached allocators
            torch.cuda.empty_cache()
            # Force synchronize the device
            torch.cuda.synchronize(gpu_id)
            # Reset the peak memory stats
            torch.cuda.reset_peak_memory_stats(gpu_id)
            # Clear IPC handles
            torch.cuda.ipc_collect()
        except Exception as e:
            print(f"Warning: GPU cleanup failed for GPU {gpu_id}: {str(e)}")

    def _process_cleanup(self, gpu_id: int):
        """Cleanup to be called within each worker process"""
        try:
            # Set the device for this process
            torch.cuda.set_device(gpu_id)
            # Clear CUDA memory for this process
            torch.cuda.empty_cache()
            torch.cuda.synchronize(gpu_id)
            # Delete all PyTorch CUDA state
            if torch.cuda.is_initialized():
                torch.cuda._lazy_init()
                torch.cuda.synchronize(gpu_id)
        except Exception as e:
            print(f"Warning: GPU cleanup failed for GPU {gpu_id}: {str(e)}")

    def _worker_wrapper(self, func, kwargs):
        """Wrapper to ensure proper cleanup in worker processes"""
        try:
            gpu_id = kwargs["gpu_id"]  # Extract gpu_id from kwargs
            torch.cuda.set_device(gpu_id)  # Set device before running function
            return func(**kwargs)
        finally:
            self._process_cleanup(kwargs["gpu_id"])

    def compile(self, cuda_files: List[str], debug: bool = False) -> List[Any]:
        """Compiles kernels in parallel using compile_cuda_kernel."""
        results = [None] * len(cuda_files)
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=len(cuda_files))
        try:
            async_results = []
            for i, cuda_file in enumerate(cuda_files):
                gpu_id = i % self.available_gpus
                ext_dir = self.ext_dirs[gpu_id]
                async_res = pool.apply_async(
                    self._worker_wrapper,
                    args=(
                        compile_cuda_kernel,
                        {
                            "task_dir": self.task_dir,
                            "cuda_code_path": cuda_file,
                            "gpu_id": gpu_id,
                            "ext_dir": ext_dir,
                            "timeout": self.timeout,
                            "debug": debug,
                        },
                    ),
                )
                async_results.append((i, async_res))
            for i, async_res in async_results:
                try:
                    results[i] = async_res.get(timeout=self.timeout)
                except mp.TimeoutError:
                    print(
                        f"Compilation timed out after {self.timeout}s for file: {cuda_files[i]}"
                    )
                    results[i] = {
                        "summary": {"correct": False},
                        "cuda_fname": cuda_files[i],
                        "stderr": f"Compilation timed out after {self.timeout}s",
                        "stdout": "",
                    }
                except Exception as e:
                    results[i] = {
                        "summary": {"correct": False},
                        "cuda_fname": cuda_files[i],
                        "stderr": str(e),
                        "stdout": "",
                    }
        finally:
            pool.terminate()
            pool.join()
            for i in range(min(len(cuda_files), self.available_gpus)):
                self._cleanup_gpu(i)
        return results

    def test(self, cuda_files: List[str], debug: bool = False) -> List[Any]:
        """Tests kernels in parallel using test_cuda_kernel."""
        results = [None] * len(cuda_files)
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=len(cuda_files))
        try:
            async_results = []
            for i, cuda_file in enumerate(cuda_files):
                gpu_id = i % self.available_gpus
                ext_dir = self.ext_dirs[gpu_id]
                kwargs = {
                    "task_dir": self.task_dir,
                    "cuda_code_path": cuda_file,
                    "config_fname": self.config_fname,
                    "op_atol": self.op_atol,
                    "op_rtol": self.op_rtol,
                    "num_correct_trials": self.num_correct_trials,
                    "multi_init_settings": True,
                    "multi_input_settings": True,
                    "gpu_id": gpu_id,
                    "ext_dir": ext_dir,
                    "timeout": self.timeout,
                    "debug": debug,
                    "forward": self.forward,
                }
                async_res = pool.apply_async(
                    self._worker_wrapper,
                    args=(correct_cuda_kernel, kwargs),
                )
                async_results.append((i, async_res))

            for i, async_res in async_results:
                try:
                    results[i] = async_res.get(timeout=self.timeout)
                except mp.TimeoutError:
                    print(
                        f"Testing timed out after {self.timeout}s for file: {cuda_files[i]}"
                    )
                    results[i] = {
                        "summary": {"correct": False, "max_diff": float("nan")},
                        "cuda_fname": cuda_files[i],
                        "error": f"Testing timed out after {self.timeout}s",
                        "stderr": f"Testing timed out after {self.timeout}s",
                        "stdout": "",
                    }
                except Exception as e:
                    results[i] = {
                        "summary": {"correct": False, "max_diff": float("nan")},
                        "cuda_fname": cuda_files[i],
                        "error": str(e),
                        "stderr": str(e),
                        "stdout": "",
                    }
        finally:
            pool.terminate()
            pool.join()
            # Clean up each GPU that was used
            for i in range(min(len(cuda_files), self.available_gpus)):
                self._cleanup_gpu(i)
        return results

    def evaluate(self, cuda_files: List[str], debug: bool = False) -> List[Any]:
        """Evaluates kernels in parallel using eval_cuda_kernel."""
        results = [None] * len(cuda_files)
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=len(cuda_files))
        try:
            async_results = []
            for i, cuda_file in enumerate(cuda_files):
                gpu_id = i % self.available_gpus
                ext_dir = self.ext_dirs[gpu_id]
                async_res = pool.apply_async(
                    self._worker_wrapper,
                    args=(
                        eval_cuda_kernel,
                        {
                            "task_dir": self.task_dir,
                            "cuda_code_path": cuda_file,
                            "config_fname": self.config_fname,
                            "warmup_time": self.warmup_time,
                            "repetition_time": self.repetition_time,
                            "eval_type": self.eval_type,
                            "multi_init_settings": self.multi_init_settings,
                            "multi_input_settings": self.multi_input_settings,
                            "gpu_id": gpu_id,
                            "ext_dir": ext_dir,
                            "timeout": self.timeout,
                            "debug": debug,
                            "forward": self.forward,
                        },
                    ),
                )
                async_results.append((i, async_res))
            for i, async_res in async_results:
                try:
                    results[i] = async_res.get(timeout=self.timeout)
                except mp.TimeoutError:
                    print(
                        f"Evaluation timed out after {self.timeout}s for file: {cuda_files[i]}"
                    )
                    results[i] = {
                        "summary": {"correct": False},
                        "cuda_fname": cuda_files[i],
                        "error": f"Evaluation timed out after {self.timeout}s",
                        "stderr": f"Evaluation timed out after {self.timeout}s",
                        "stdout": "",
                    }
                except Exception as e:
                    results[i] = {
                        "summary": {"correct": False},
                        "cuda_fname": cuda_files[i],
                        "error": str(e),
                        "stderr": str(e),
                        "stdout": "",
                    }
        finally:
            pool.terminate()
            pool.join()
            for i in range(min(len(cuda_files), self.available_gpus)):
                self._cleanup_gpu(i)
        return results

    def profile(self, cuda_files: List[str], debug: bool = False) -> List[Any]:
        """Profiles kernels in parallel using prof_cuda_kernel."""
        results = [None] * len(cuda_files)
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=len(cuda_files))
        try:
            async_results = []
            for i, cuda_file in enumerate(cuda_files):
                gpu_id = i % self.available_gpus
                ext_dir = self.ext_dirs[gpu_id]
                async_res = pool.apply_async(
                    self._worker_wrapper,
                    args=(
                        prof_cuda_kernel,
                        {
                            "task_dir": self.task_dir,
                            "cuda_code_path": cuda_file,
                            "config_fname": self.config_fname,
                            "torch_prof": self.torch_prof,
                            "ncu_prof": self.ncu_prof,
                            "clang_tidy": self.clang_tidy,
                            "gpu_id": gpu_id,
                            "ext_dir": ext_dir,
                            "debug": debug,
                            "forward": self.forward,
                            "repetition_time": self.repetition_time,
                        },
                    ),
                )
                async_results.append((i, async_res))
            for i, async_res in async_results:
                try:
                    results[i] = async_res.get(timeout=self.timeout)
                    results[i]["cuda_fname"] = cuda_files[i]
                except mp.TimeoutError:
                    print(
                        f"Profiling timed out after {self.timeout}s for file: {cuda_files[i]}"
                    )
                    results[i] = {
                        "summary": {"correct": False},
                        "cuda_fname": cuda_files[i],
                        "stderr": f"Profiling timed out after {self.timeout}s",
                        "stdout": "",
                    }
                except Exception as e:
                    results[i] = {
                        "summary": {"correct": False},
                        "cuda_fname": cuda_files[i],
                        "stderr": str(e),
                        "stdout": "",
                    }
        finally:
            pool.terminate()
            pool.join()
            for i in range(min(len(cuda_files), self.available_gpus)):
                self._cleanup_gpu(i)
        return results

    def torch_eval(self, gpu_id: int = 0, debug: bool = False) -> Dict[str, Any]:
        """Evaluates native PyTorch vs torch.compile performance on a single GPU.

        Args:
            gpu_id: GPU ID to use for evaluation (default: 0)

        Returns:
            Dictionary containing evaluation results for both versions
        """
        if not (0 <= gpu_id < self.available_gpus):
            raise ValueError(f"GPU ID must be between 0 and {self.available_gpus - 1}")

        # Check if task_dir exists and if the results are already cached
        if not os.path.exists(self.task_dir):
            raise FileNotFoundError(f"Task directory not found: {self.task_dir}")

        # Check if the results are already cached
        native_results_fname = os.path.join(
            self.task_dir,
            "eval_results",
            "forward" if self.forward else "backward",
            "torch_native_results.json",
        )
        compile_results_fname = os.path.join(
            self.task_dir,
            "eval_results",
            "forward" if self.forward else "backward",
            "torch_compile_results.json",
        )
        if os.path.exists(compile_results_fname) and os.path.exists(
            native_results_fname
        ):
            with open(compile_results_fname, "r") as f:
                torch_compile_results = json.load(f)
            with open(native_results_fname, "r") as f:
                torch_native_results = json.load(f)
            print(f"Found cached torch results for {self.task_dir}")
            return {
                "native": torch_native_results,
                "compiled": torch_compile_results,
            }

        try:
            native_results, compile_results = eval_torch_runtime(
                task_dir=self.task_dir,
                multi_init_settings=self.multi_init_settings,
                multi_input_settings=self.multi_input_settings,
                warmup_time=self.warmup_time,
                repetition_time=self.repetition_time,
                eval_type=self.eval_type,
                gpu_id=gpu_id,
                ext_dir=self.ext_dirs[gpu_id],
                timeout=self.timeout,
                debug=debug,
                forward=self.forward,
            )
            return {"native": native_results, "compiled": compile_results}
        except Exception as e:
            return {"error": str(e)}

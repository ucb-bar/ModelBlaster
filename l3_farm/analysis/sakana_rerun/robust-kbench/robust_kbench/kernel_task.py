import importlib.util
import os
from typing import Optional
from robust_kbench.config_task import ConfigTask


class KernelTask(object):
    def __init__(
        self,
        task_dir: str,
        multi_input_settings: bool = False,
        multi_init_settings: bool = False,
        forward: bool = True,
        config_fname: Optional[str] = None,
    ):
        if forward:
            import_fname = "func_forward.py"
            cuda_fname = "forward.cu"
        else:
            import_fname = "func_backward.py"
            cuda_fname = "backward.cu"

        self.multi_input_settings = multi_input_settings
        self.multi_init_settings = multi_init_settings

        # Import the task module dynamically
        self.task_dir = task_dir
        self.task_name = os.path.basename(self.task_dir)

        try:
            # Use importlib.util to import the module directly from its file path
            module_path = os.path.join(self.task_dir, import_fname)
            spec_name = import_fname.replace(".py", "")
            spec = importlib.util.spec_from_file_location(spec_name, module_path)
            self.task_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.task_module)
        except Exception as e:
            raise ImportError(
                f"Could not find task module for task {self.task_name}: {e}"
            )

        # Check if default correct kernel exists
        self.correct_kernel_path = os.path.join(self.task_dir, cuda_fname)
        if os.path.exists(self.correct_kernel_path):
            # Load the correct kernel as text
            with open(self.correct_kernel_path, "r") as f:
                self.correct_kernel = f.read()
        else:
            self.correct_kernel = None

        # Load text of the task module
        forward_task_file = os.path.join(self.task_dir, import_fname)
        with open(forward_task_file, "r") as f:
            self.module_text = f.read()

        # Get module function string
        self.forward_fn_str = extract_forward_fn(self.module_text)
        self.operation_info = extract_operation_info(self.module_text)

        if not forward:
            self.autograd_fn_str = extract_autograd_fn(self.module_text)

        # Get task config from directory
        self.config_task = ConfigTask(
            task_dir,
            multi_input_settings=self.multi_input_settings,
            multi_init_settings=self.multi_init_settings,
            forward=forward,
            config_fname=config_fname,
        )
        self.init_configs, self.input_configs, self.configs_str = (
            self.config_task.get_configs()
        )

    @property
    def autograd_fn(self):
        """Get the autograd function from the task module."""
        return self.task_module.AutogradFunction

    @property
    def forward_fn(self):
        """Get the forward function from the task module."""
        return self.task_module.forward_fn

    @property
    def model(self):
        """Get the model class from the task module."""
        return self.task_module.Model

    def get_input_settings(self):
        """Get the input settings from the task module."""
        return self.input_configs

    def get_init_settings(self):
        """Get the initialization input settings from the task module."""
        return self.init_configs

    def get_configs_str(self):
        """Get the configuration strings from the task module."""
        return self.configs_str

    def get_inputs(self, **kwargs):
        """Get inputs from the task module with optional settings."""
        return self.task_module.get_inputs(**kwargs)

    def get_input_names(self):
        """Get the input names from the task module."""
        return self.task_module.input_names


def extract_forward_fn(code_str: str) -> str:
    """Extract the forward_fn definition from the code"""
    # Find the start of forward_fn
    start = code_str.find("def forward_fn")
    if start == -1:
        return None

    # Find the next function definition or class definition
    next_def = code_str.find("\ndef ", start + 1)
    next_class = code_str.find("\nclass ", start + 1)

    # Get the earlier of the two (if they exist)
    end_candidates = [pos for pos in [next_def, next_class] if pos != -1]
    end = min(end_candidates) if end_candidates else len(code_str)

    # Extract the function definition
    fn_str = code_str[start:end].strip()
    return fn_str


def extract_operation_info(code_str: str) -> str:
    """Extract the operation info from the code"""
    # Find docstring of forward_fn
    start = code_str.find('"""') + 3
    if start == -1:
        return None
    end = code_str.find("Args:", start + 3)
    if end == -1:
        return None
    docstring = code_str[start:end].strip()
    # Only take first lines before Args:
    return docstring


def extract_autograd_fn(code_str: str) -> str:
    """Extract the forward_fn definition from the code"""
    # Find the start of forward_fn
    start = code_str.find("class AutogradFunction")
    if start == -1:
        return None

    # Find the next function definition or class definition
    next_def = code_str.find("\ndef ", start + 1)
    next_class = code_str.find("\nclass ", start + 1)

    # Get the earlier of the two (if they exist)
    end_candidates = [pos for pos in [next_def, next_class] if pos != -1]
    end = min(end_candidates) if end_candidates else len(code_str)

    # Extract the function definition
    fn_str = code_str[start:end].strip()
    return fn_str

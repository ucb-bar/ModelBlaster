import os
import json
from typing import List, Dict, Tuple, Optional


class ConfigTask:
    def __init__(
        self,
        task_dir: str,
        multi_input_settings: bool = False,
        multi_init_settings: bool = False,
        forward: bool = True,
        config_fname: Optional[str] = None,
    ):
        self.task_dir = task_dir
        self.multi_input_settings = multi_input_settings
        self.multi_init_settings = multi_init_settings
        self.forward = forward
        # Get filepath to configs
        if config_fname is None:
            config_fname = os.path.join(
                self.task_dir,
                f"config_{'forward' if self.forward else 'backward'}.json",
            )
        else:
            config_fname = os.path.join(self.task_dir, config_fname)
        if not os.path.exists(config_fname):
            raise FileNotFoundError(f"Config file not found: {config_fname}")

        # Load json settings dict
        with open(config_fname, "r") as f:
            self.configs = json.load(f)

    def get_configs(self) -> Tuple[List[Dict], List[Dict], List[str]]:
        # Get input settings
        if self.multi_input_settings:
            input_configs = self.configs["multi_input_configs"]
        else:
            input_configs = self.configs["single_input_configs"]
        # Get init settings
        if self.multi_init_settings:
            init_configs = self.configs["multi_init_configs"]
        else:
            init_configs = self.configs["single_init_configs"]

        # Get shared settings
        if self.multi_input_settings and self.multi_init_settings:
            shared_configs = self.configs["multi_shared_configs"]
        else:
            shared_configs = self.configs["single_shared_configs"]

        all_init_configs = []
        all_input_configs = []
        all_configs_str = []
        for init_config in init_configs:
            for input_config in input_configs:
                for shared_config in shared_configs:
                    # Shared config should be applied last to allow overriding
                    init_c = {**init_config, **shared_config}
                    input_c = {**input_config, **shared_config}
                    all_init_configs.append(init_c)
                    all_input_configs.append(input_c)
                    # Combine init and input configs into a single string
                    config_str = "_".join(
                        [f"{k}_{v}" for k, v in init_c.items()]
                        + [f"{k}_{v}" for k, v in input_c.items()]
                    )
                    # Replace space, dots, -, = with _
                    config_str = (
                        config_str.replace(" ", "_")
                        .replace(".", "_")
                        .replace("-", "_")
                        .replace("=", "_")
                    )
                    all_configs_str.append(config_str)
        return all_init_configs, all_input_configs, all_configs_str

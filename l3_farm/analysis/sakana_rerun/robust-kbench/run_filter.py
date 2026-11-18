import argparse
import json
import os
from robust_kbench.kernel_task import KernelTask
from robust_kbench.filter.forward import (
    filter_output_range,
    filter_output_std,
    filter_output_axes,
    filter_input_impact,
    filter_llm_sanity,
)


def main(task_dir: str, num_seeds: int = 5):
    """Filter out tasks that are not robust to the given filter."""
    task = KernelTask(
        task_dir,
        multi_input_settings=False,
        multi_init_settings=False,
    )
    # Filter check 1: Output range not only in (-0.01, 0.01)
    should_filter_output_range = filter_output_range(task, num_seeds)
    # Filter check 2: Output std not close to 0
    should_filter_output_std = filter_output_std(task, num_seeds)
    # Filter check 3: Output axes variation not close to 0
    should_filter_output_axes = filter_output_axes(task, num_seeds)
    # Filter check 4: Model weights don't affect the output
    # Only relevant for tasks with model weights
    # should_filter_init_impact = filter_init_impact(task, num_seeds)
    # Filter check 5: Input variation doesn't affect the output
    should_filter_input_impact = filter_input_impact(task, num_seeds)
    # Filter check 6: LLM sanity
    is_redundant, is_inefficient, llm_output = filter_llm_sanity(
        task.module_text,
        model_name="claude-3-7-sonnet-20250219",
        temperature=0.0,
        max_tokens=8192,
    )
    return {
        "filter_output_range": should_filter_output_range,
        "filter_output_std": should_filter_output_std,
        "filter_output_axes": should_filter_output_axes,
        "filter_input_impact": should_filter_input_impact,
        "filter_llm_redundancy": is_redundant,
        "filter_llm_inefficiency": is_inefficient,
        "filter_llm_assessment": llm_output,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--task_dir", type=str, default="tasks")
    args = parser.parse_args()

    if args.task_dir in "tasks":
        # get all tasks in tasks/
        tasks = os.listdir("tasks")
        for task in tasks:
            task_dir = os.path.join("tasks", task)
            print(f"Filtering task: {task_dir}")
            filter_results = main(task_dir, args.num_seeds)
            print(json.dumps(filter_results, indent=4))

            # Store json in task_dir/filter_results.json
            with open(os.path.join(task_dir, "filter_results.json"), "w") as f:
                json.dump(filter_results, f, indent=4)
    elif args.task_dir in ["level_1", "level_2"]:
        tasks = os.listdir(f"tasks/kernelbench/{args.task_dir}")
        print(tasks)
        # sort tasks by number
        tasks.sort(key=lambda x: int(x.split("_")[1]))
        for task in tasks:
            task_dir = os.path.join(f"tasks/kernelbench/{args.task_dir}", task)
            # check if filter_results.json exists
            print(f"Filtering task: {task_dir}")
            filter_results = main(task_dir, args.num_seeds)
            print(json.dumps(filter_results, indent=4))

            # Store json in task_dir/filter_results.json
            with open(os.path.join(task_dir, "filter_results.json"), "w") as f:
                json.dump(filter_results, f, indent=4)
    else:
        print(f"Filtering task: {args.task_dir}")
        filter_results = main(args.task_dir, args.num_seeds)
        print(json.dumps(filter_results, indent=4))

        # Store json in task_dir/filter_results.json
        with open(os.path.join(args.task_dir, "filter_results.json"), "w") as f:
            json.dump(filter_results, f, indent=4)

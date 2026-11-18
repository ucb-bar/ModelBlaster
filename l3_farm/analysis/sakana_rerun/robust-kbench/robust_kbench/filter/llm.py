"""LLM client for the LLM2CUDA pipeline."""

import time
import random
from typing import Any, Dict, Tuple, List, Union
import os
import backoff
import openai
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import anthropic
import multiprocessing as mp
from dotmap import DotMap

MAX_O_SERIES_TOKENS = 16384
NUM_QUERY_RETRIES = 3


class LLMClient:
    def __init__(
        self,
        model_names: Union[List[str], str] = "gpt-4o-2024-05-13",
        temperatures: Union[float, List[float]] = 0.75,
        max_tokens: int = 4096,
        reasoning_effort: str = "high",
        extract: bool = False,
    ):
        self.temperatures = temperatures
        self.max_tokens = max_tokens
        self.model_names = model_names
        self.reasoning_effort = reasoning_effort
        self.extract = extract

    def sequence_query(
        self,
        msg: str,
        system_message: str,
        msg_history: List = [],
        print_debug: bool = False,
        n: int = 1,
        num_reflexion_iters: int = 0,
        reflexion_message: str = None,
    ) -> Tuple[List[Union[str, Dict, List, None]], List[List], List[str]]:
        """Sequential query implementation that ensures diverse proposals.

        Args:
            Same as query() method, plus:
            num_reflexion_iters (int): Number of reflection iterations
            reflexion_message (str): Template for reflection prompt

        Returns:
            Same format as query() method
        """
        if n == 1:
            content, history, model, temp = query(
                self.model_names,
                msg,
                system_message,
                self.temperatures,
                self.max_tokens,
                msg_history,
                print_debug,
                self.reasoning_effort,
            )
            if num_reflexion_iters > 0 and reflexion_message:
                content, history = reflect(
                    model,
                    reflexion_message,
                    system_message,
                    num_reflexion_iters,
                    history,
                    temp,
                    self.max_tokens,
                    print_debug=print_debug,
                )
            return content, history, model, temp

        contents = []
        histories = []
        models = []
        temps = []
        # Modify prompt for each iteration to encourage diversity
        base_msg = msg
        start_t = time.time()
        print(f"Sampling {n} diverse kernel proposals sequentially.")
        for i in range(n):
            if i == 0:
                diversity_msg = (
                    base_msg
                    + f"\nProposal {i+1} of {n}.\nImprove the runtime of the kernel, while maintaining the same functionality."
                )
            else:
                diversity_msg = (
                    f"Proposal {i+1} of {n}.\n"
                    f"Make this proposal different from previous ones. "
                    f"Focus on a unique implementation strategy. "
                )

            content, new_history, model, temp = self.sample_query(
                diversity_msg,
                system_message,
                msg_history,
                print_debug,
            )

            # Add reflection if requested
            if num_reflexion_iters > 0 and reflexion_message:
                content, new_history = reflect(
                    model,
                    reflexion_message,
                    system_message,
                    num_reflexion_iters,
                    new_history,
                    self.temperatures,
                    self.max_tokens,
                    print_debug=print_debug,
                )

            print(
                f"==> {i+1}/{n} - kernel proposal: {model} - Time: {time.time() - start_t:.2f}"
            )
            contents.append(content)
            histories.append(new_history)
            models.append(model)
            temps.append(temp)
            # Add previous proposal to message history to inform next one
            if i < n - 1:  # Don't need to update for last iteration
                msg_history = new_history

        return contents, histories, models, temps

    @backoff.on_exception(
        backoff.expo,
        (
            openai.RateLimitError,
            openai.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.APIStatusError,
        ),
    )
    def sample_query(
        self,
        msg: Union[str, List[str]],
        system_message: Union[str, List[str]],
        msg_history: Union[List, List[List]] = [],
        print_debug: bool = False,
        n: int = 1,
        num_reflexion_iters: int = 0,
        reflexion_message: str = None,
        verbose: bool = True,
    ) -> Tuple[
        Union[str, Dict, List[str], List[Dict]],
        Union[List, List[List]],
        List[str],
        List[float],
    ]:
        """Query the LLM with optional batching and reflection.

        Args:
            msg (Union[str, List[str]]): Single message or list of n messages to query with
            system_message (Union[str, List[str]]): Single system message or list of n system messages
            msg_history (List, optional): Message history. Defaults to [].
            print_debug (bool, optional): Whether to print debug info. Defaults to False.

            n (int, optional): Number of completions to generate. Defaults to 1.
            num_reflexion_iters (int, optional): Number of reflection iterations. Defaults to 0.
            reflexion_message (str, optional): Template for reflection prompt. Defaults to None.

        Returns:
            If n=1: (completion, message_history, model, temp)
            If n>1: (list of completions, list of message histories, list of models, list of temps)
        """
        if n == 1:
            content, history, model, temp = query(
                self.model_names,
                msg,
                system_message,
                self.temperatures,
                self.max_tokens,
                msg_history,
                print_debug,
                verbose,
                self.reasoning_effort,
            )
            if num_reflexion_iters > 0 and reflexion_message:
                content, history = reflect(
                    model,
                    reflexion_message,
                    system_message,
                    num_reflexion_iters,
                    history,
                    temp,
                    self.max_tokens,
                    print_debug=print_debug,
                    verbose=verbose,
                )
            if self.extract:
                content = self.extract_dict(content)
            return content, history, model, temp

        # Handle single message/system_message case
        if isinstance(msg, str):
            msg = [msg] * n
        if isinstance(system_message, str):
            system_message = [system_message] * n
        if len(msg_history) == 0:
            msg_history = [[]] * n
        if len(msg_history) == 1:
            msg_history = msg_history * n

        # Validate lengths
        if len(msg) != n or len(system_message) != n:
            raise ValueError(
                f"Length of messages ({len(msg)}) and system messages ({len(system_message)}) must match n={n}"
            )
        # print(msg_history[0])
        worker_args = [
            (
                self.model_names,
                msg[i],
                system_message[i],
                self.temperatures,
                self.max_tokens,
                msg_history[i],
                print_debug,
                num_reflexion_iters,
                reflexion_message,
                verbose,
            )
            for i in range(n)
        ]

        # Create pool and run queries in parallel with error handling
        start_t = time.time()
        print(f"Sampling {n} diverse kernel proposals in parallel.")
        with mp.Pool() as pool:
            results = list(pool.imap_unordered(query_worker, worker_args))
        print(f"Time taken for {n} parallel queries: {time.time() - start_t:.2f}")

        # Filter out failed results
        results = [r for r in results if r[0] is not None]
        if len(results) == 0:
            raise RuntimeError("All parallel queries failed")

        # Unzip results
        contents, histories, models, temps = zip(*results)

        contents = list(contents)
        histories = list(histories)
        models = list(models)
        temps = list(temps)
        if self.extract:
            contents = self.extract_dict(contents)

        # Loop over contents and only use content, hist, models, temps where all three keys are in content: name, code, thought otherwise remove from lists
        for i, content in enumerate(contents):
            if (
                "name" not in content
                or "code" not in content
                or "thought" not in content
            ):
                contents.pop(i)
                histories.pop(i)
                models.pop(i)
                temps.pop(i)
        return contents, histories, models, temps

    def extract_dict(
        self, llm_output: Union[str, List[str]]
    ) -> Union[Dict, List[Dict], None]:
        """Extract the dictionary from the LLM output.

        Args:
            llm_output (Union[str, List[str]]): The output(s) from the LLM.

        Returns:
            Union[Dict, List[Dict], None]: The dictionary/dictionaries from the LLM output(s).
            Returns None if parsing fails, a single dict for single input,
            or a list of dicts (with possible Nones) for batch input.
        """
        # Handle batch case
        if isinstance(llm_output, list) or isinstance(llm_output, tuple):
            return [self.extract_dict(output) for output in llm_output]

        # Single output case
        return DotMap(**extract_kernel_components(llm_output))


def extract_kernel_components(llm_output: str) -> dict:
    """Extract the name, thought and code components from LLM output.

    Args:
        llm_output (str): The output string containing NAME:, THOUGHT: and CODE: sections

    Returns:
        dict: Dictionary containing 'name', 'thought' and 'code' keys with extracted strings
    """
    components = {}

    # Extract name
    name_start = llm_output.find("NAME:") + 5
    name_end = llm_output.find("THOUGHT:")
    if name_start > 4 and name_end != -1:
        components["name"] = llm_output[name_start:name_end].strip()

    # Extract thought
    thought_start = llm_output.find("THOUGHT:") + 8
    thought_end = llm_output.find("CODE:")
    if thought_start > 7 and thought_end != -1:
        components["thought"] = llm_output[thought_start:thought_end].strip()

    # Extract code
    code_start = llm_output.find("```cpp") + 6
    code_end = llm_output.rfind("```")
    if code_start > 5 and code_end != -1:
        components["code"] = llm_output[code_start:code_end].strip()

    return components


def query(
    model_names: Union[str, List[str]],
    msg: str,
    system_message: str,
    temperatures: Union[float, List[float]] = 0.75,
    max_tokens: int = 4096,
    msg_history: List = [],
    print_debug: bool = False,
    verbose: bool = True,
    reasoning_effort: str = "high",
) -> Tuple[Union[str, Dict, None], List, str, float]:
    """Query the LLM.

    Args:
        msg (str): The message to query the LLM with.
        msg_history (list, optional): The message history. Defaults to [].
        print_debug (bool, optional): Whether to print the debug information. Defaults to False.

    Raises:
        ValueError: If the model is not supported.

    Returns:
        Tuple[str, List]: The response from the LLM and the message history.
    """
    # sample a model from the list
    if isinstance(model_names, list):
        model_name = random.choice(model_names)
    else:
        model_name = model_names
    client, model = get_client_llm(model_name)

    # perform temperature sampling if provided
    if isinstance(temperatures, list):
        # o1 only allows temp = 1.0
        if (
            "o1-" in model_name
            or "o3-" in model_name
            or "reasoner" in model_name
            or "r1" in model_name
            or "R1" in model_name
        ):
            temp = 1.0
        else:
            temp = random.choice(temperatures)
    else:
        temp = temperatures
    if verbose:
        print(f"==> Querying with model {model} & temp {temp}.")
    if "claude" in model:
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temp,
            system=system_message,
            messages=new_msg_history,
        )
        content = response.content[0].text
        new_msg_history.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": content,
                    }
                ],
            }
        )
    elif model in [
        "o1-preview-2024-09-12",
        "o1-mini-2024-09-12",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            n=1,
            seed=0,
            max_completion_tokens=MAX_O_SERIES_TOKENS,
        )
        content = response.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": content})
    elif model in [
        "o1-2024-12-17",
        "o3-mini-2025-01-31",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            n=1,
            seed=0,
            max_completion_tokens=MAX_O_SERIES_TOKENS,
            reasoning_effort=reasoning_effort,
        )
        content = response.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": content})
    elif model in [
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temp,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            seed=0,
        )
        content = response.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": content})
    elif model in [
        "deepseek-chat",
        "deepseek/deepseek-chat",
        "deepseek-ai/DeepSeek-V3",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temp,
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": content})
    elif model in [
        "deepseek-reasoner",
        "deepseek/deepseek-r1",
        "deepseek-ai/DeepSeek-R1",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": content})
    elif model in [
        "meta-llama/llama-3.1-405b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "meta-llama/llama-3.1-405b",
        "meta-llama/llama-3.1-70b",
        "qwen/qwen-2.5-coder-32b-instruct",
    ]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temp,
            max_tokens=max_tokens,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history.append({"role": "assistant", "content": content})
    elif "gemini" in model:
        new_msg_history = msg_history + [{"role": "user", "parts": msg}]
        response = client.generate_content(
            contents=[
                {"role": "user", "parts": system_message},
                *new_msg_history,
            ],
            generation_config=GenerationConfig(
                temperature=temp,
                max_output_tokens=max_tokens,
                candidate_count=1,
            ),
        )
        content = response.text
        new_msg_history.append({"role": "model", "parts": content})
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()
    return content, new_msg_history, model_name, temp


def reflect(
    model_name: str,
    reflexion_message: str,
    system_message: str,
    num_reflexion_iters: int,
    msg_history: List,
    temperature: Union[float, List[float]] = 0.75,
    max_tokens: int = 4096,
    end_cond_str: str = "I am done",
    print_debug: bool = False,
    verbose: bool = True,
) -> Tuple[
    Union[str, Dict, None],
    List,
]:
    """Reflect on previous responses to improve them.

    Args:
        reflexion_message (str): The reflexion prompt
        system_message (str): The system message
        num_reflexion_iters (int): Number of reflection iterations
        msg_history (Union[List, List[List]]): Single or batch of message histories
        end_cond_str (str): String that indicates reflection should stop
        print_debug (bool): Whether to print debug info

    Returns:
        If single history: (final_completion, final_history)
        If batch: (list of final_completions, list of final_histories)
    """
    for i in range(1, num_reflexion_iters + 1):
        if verbose:
            print(f"Reflection iteration {i}/{num_reflexion_iters}")
        reflexion_prompt = reflexion_message.format(
            current_round=i, num_reflections=num_reflexion_iters
        )
        content, msg_history, _, _ = query(
            model_name,
            reflexion_prompt,
            system_message,
            temperature,
            max_tokens,
            msg_history,
            print_debug=print_debug,
            verbose=verbose,
        )
        if end_cond_str in content:
            if print_debug:
                print(f"End condition met after {i} reflections.")
            break

    return content, msg_history


def get_client_llm(model_name: str) -> Tuple[Any, str]:
    """Get the client and model for the given model name.

    Args:
        model_name (str): The name of the model to get the client and model for.

    Raises:
        ValueError: If the model is not supported.

    Returns:
        The client and model for the given model name.
    """
    # print(f"Getting client for model {model_name}")
    if model_name in [
        "claude-3-5-haiku-20241022",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-7-sonnet-20250219",
    ]:
        model = model_name
        client = anthropic.Anthropic()
    elif model_name.startswith("bedrock") and "claude" in model_name:
        # Expects: bedrock/<MODEL_ID>
        # bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0
        # bedrock/anthropic.claude-3-5-haiku-20241022-v1:0
        # bedrock/anthropic.claude-3-opus-20240229-v1:0
        # bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0
        model = model_name.split("/")[-1]
        client = anthropic.AnthropicBedrock(
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION_NAME"),
        )
    elif model_name.startswith("vertex_ai") and "claude" in model_name:
        # Expects: vertex_ai/<MODEL_ID>
        model = model_name.split("/")[-1]
        client = anthropic.AnthropicVertex()
    elif model_name in [
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini-2024-07-18",
        "o1-mini-2024-09-12",
        "o1-preview-2024-09-12",
        "o1-2024-12-17",
        "o3-mini-2025-01-31",
    ]:
        model = model_name
        client = openai.OpenAI()
    elif model_name.startswith("azure-"):
        # get rid of the azure- prefix
        model = model_name.split("azure-")[-1]
        client = openai.AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_API_ENDPOINT"),
        )
    elif model_name in [
        "deepseek-coder-v2-0724",
        "deepseek-chat",
        "deepseek-reasoner",
    ]:
        model = model_name
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )
    elif model_name in [
        "llama-3.1-70b",
        "llama-3.1-70b-instruct",
        "llama-3.1-405b",
        "llama-3.1-405b-instruct",
    ]:
        model = f"meta-llama/{model_name}"
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    elif model_name in [
        "openrouter-deepseek-chat",
    ]:
        model = "deepseek/deepseek-chat"
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    elif model_name in [
        "openrouter-deepseek-reasoner",
    ]:
        model = "deepseek/deepseek-r1"
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    elif model_name in [
        "together-deepseek-chat",
    ]:
        model = "deepseek-ai/DeepSeek-V3"
        client = openai.OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ["TOGETHER_API_KEY"],
        )
    elif model_name in [
        "together-deepseek-reasoner",
    ]:
        model = "deepseek-ai/DeepSeek-R1"
        client = openai.OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ["TOGETHER_API_KEY"],
        )
    elif "gemini" in model_name:
        # gemini-1.5-flash
        # gemini-1.5-pro
        # gemini-2.0-flash-thinking-exp-01-21
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        model = model_name.split("/")[-1]
        client = genai.GenerativeModel(model)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    return client, model


def query_worker(args):
    """Helper function for parallel query processing."""
    try_count = 0
    while try_count < NUM_QUERY_RETRIES:
        try:
            (
                model_names,
                msg,
                system_message,
                temperature,
                max_tokens,
                msg_history,
                print_debug,
                num_reflexion_iters,
                reflexion_message,
                verbose,
            ) = args

            content, history, model, temp = query(
                model_names,
                msg,
                system_message,
                temperature,
                max_tokens,
                msg_history,
                print_debug,
                verbose=verbose,
            )
            if num_reflexion_iters > 0 and reflexion_message:
                content, history = reflect(
                    model,
                    reflexion_message,
                    system_message,
                    num_reflexion_iters,
                    history,
                    temperature,
                    max_tokens,
                    print_debug=print_debug,
                    verbose=verbose,
                )
            return content, history, model, temp
        except Exception as e:
            try_count += 1
            print(
                f"[{try_count}/{NUM_QUERY_RETRIES}]: Error in worker process: {str(e)}"
            )
    return None, None, None, None


if __name__ == "__main__":
    # Setup the LLM client
    llm = LLMClient(
        model_names=[
            # "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            # "gpt-4o-mini-2024-07-18",
            # "deepseek-chat",
            "deepseek-reasoner",
        ],
        temperatures=0.75,
        max_tokens=4096,
    )
    # Single query with reflection
    content, history, model, temp = llm.sample_query(
        msg="Write a poem about CUDA.",
        system_message="You are a poet.",
        num_reflexion_iters=2,
        reflexion_message="Reflect on your previous response. Iteration {current_round}/{num_reflections}",
        n=2,
    )
    print(content)
    print(history)
    print(model)
    print(temp)
    # Sequence queries with reflection
    contents, histories, models, temps = llm.sequence_query(
        msg="Write a poem about CUDA.",
        system_message="You are a poet.",
        num_reflexion_iters=2,
        reflexion_message="Reflect on your previous response. Iteration {current_round}/{num_reflections}",
        n=2,
    )
    for content, history, model, temp in zip(contents, histories, models, temps):
        print(content)
        print(history)
        print(model)
        print(temp)

    # Batch queries with different messages and reflection
    contents, histories, models, temps = llm.sample_query(
        msg=[
            "Write a happy poem about CUDA.",
            "Write a sad poem about CUDA.",
        ],
        system_message="You are a poet.",
        n=2,
        num_reflexion_iters=2,
        reflexion_message="Reflect on your previous response. Iteration {current_round}/{num_reflections}",
    )
    for content, history, model, temp in zip(contents, histories, models, temps):
        print(content)
        print(history)
        print(model)
        print(temp)

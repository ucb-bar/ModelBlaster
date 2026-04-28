"""Smoke test the Bedrock client against the configured Llama model.

Run:
    source ../../../set_api_keys.sh
    python -m agents.pipeline.smoke_test
"""

from __future__ import annotations

from agents.pipeline.bedrock_client import BedrockClient


def main() -> None:
    client = BedrockClient()
    print(f"model={client.model_id} region={client.region}")
    res = client.converse(
        system="You are a terse C programmer. Reply with code only, no prose.",
        user=(
            "Write a C function `int add(int a, int b)` that returns a+b. "
            "Wrap it in a ```c fenced code block."
        ),
        max_tokens=256,
        temperature=0.0,
    )
    print(f"stop_reason={res.stop_reason} tokens={res.input_tokens}+{res.output_tokens}")
    print("--- response ---")
    print(res.text)


if __name__ == "__main__":
    main()

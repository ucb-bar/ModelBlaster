"""Stub LLM for the halfway-sync GPU test (no real model, no key): `stub:hw` answers every codegen / fix request with the
current slot source plus a deterministic marker comment (so every candidate is a new, valid, equally fast kernel);
reflection / upkeep calls get the same kind of (parseable-as-nothing) answer. Registered through kb_graph_opt --plugin."""
import hashlib

from kernelblaster.agents.opt_multislot import extract_code
from kernelblaster.agents.utils import agentic_llm


def scripted(msgs):
    user = msgs[-1]["content"]
    cur = extract_code(user.split("Current source:")[-1] if "Current source:" in user
                       else user.split("Current slot source:")[-1]) or ""
    tag = hashlib.sha1(user.encode()).hexdigest()[:8]
    return "```cuda\n" + cur.rstrip("\n") + f"\n// hw-stub {tag}\n```"


agentic_llm.register_stub("hw", scripted)

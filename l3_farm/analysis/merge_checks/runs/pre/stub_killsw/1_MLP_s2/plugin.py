from kernelblaster.agents.opt_multislot import extract_code
from kernelblaster.agents.utils import agentic_llm
def scripted(msgs):
    user = msgs[-1]["content"]
    cur = extract_code(user.split("Current source:")[-1] if "Current source:" in user else user.split("Current slot source:")[-1])
    return "```cuda\n" + (cur or "") + "```"
agentic_llm.register_stub("graphopt", scripted)

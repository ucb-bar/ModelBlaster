import asyncio, importlib.util, sys, json
sp = importlib.util.spec_from_file_location("agentic_llm", "/Users/krisd/Documents/ModelBlaster-clone/ModelBlaster/third_party/KernelBlaster/src/kernelblaster/agents/utils/agentic_llm.py")
m = importlib.util.module_from_spec(sp); sys.modules["agentic_llm"] = m; sp.loader.exec_module(m)
msgs = [{"role": "system", "content": "You are a terse assistant. Reply with only the requested text."},
        {"role": "user", "content": "Reply with exactly: SMOKE-OK"}]
txt, usage = asyncio.run(m._claude_cli_once(msgs, sys.argv[1] if len(sys.argv) > 1 else "claude-opus-5-5"))
print(repr(txt)); print(json.dumps(usage))

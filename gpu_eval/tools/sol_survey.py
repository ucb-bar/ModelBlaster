"""Survey SOL-ExecBench problems without allocating them: build get_inputs()
on the meta device and report per-problem input bytes, dtypes, scalar args,
and whether a meta-device forward works.  Usage: sol_survey.py FILE... > out.tsv"""
import contextlib, importlib.util, sys, traceback
import torch

NAMES = ["rand", "randn", "zeros", "ones", "empty", "full", "randint", "arange",
         "tensor", "eye", "linspace", "randperm", "rand_like", "randn_like",
         "zeros_like", "ones_like", "empty_like", "full_like", "as_tensor", "normal"]


@contextlib.contextmanager
def on_meta():
    saved = {n: getattr(torch, n) for n in NAMES if hasattr(torch, n)}

    def wrap(fn):
        def inner(*a, **k):
            if "device" in k or fn.__name__ not in ("tensor", "as_tensor"):
                k["device"] = "meta"
            return fn(*a, **k)
        return inner
    try:
        for n, fn in saved.items():
            setattr(torch, n, wrap(fn))
        yield
    finally:
        for n, fn in saved.items():
            setattr(torch, n, fn)


def load(path):
    spec = importlib.util.spec_from_file_location("_sol", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    print("file\tin_MB\tdtypes\tn_tensor\tn_scalar\tout\tnote")
    for p in sys.argv[1:]:
        stem = p.rsplit("/", 2)[-2] + "/" + p.rsplit("/", 1)[-1][:-3]
        note, out = "", ""
        try:
            mod = load(p)
            with on_meta():
                ins = mod.get_inputs()
            ts = [t for t in ins if torch.is_tensor(t)]
            mb = sum(t.numel() * t.element_size() for t in ts) / 2**20
            dts = ",".join(sorted({str(t.dtype).replace("torch.", "") for t in ts}))
            nsc = len(ins) - len(ts)
            try:
                m = mod.Model(*(mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []))
                with torch.no_grad():
                    o = m(*ins)
                os_ = o if isinstance(o, (tuple, list)) else [o]
                out = ";".join("x".join(map(str, t.shape)) + ":" + str(t.dtype).replace("torch.", "")
                               if torch.is_tensor(t) else type(t).__name__ for t in os_)
            except Exception as e:
                note = "fwd:" + type(e).__name__ + ":" + str(e).splitlines()[0][:80] if str(e) else type(e).__name__
            print(f"{stem}\t{mb:.0f}\t{dts}\t{len(ts)}\t{nsc}\t{out}\t{note}", flush=True)
        except Exception as e:
            print(f"{stem}\t\t\t\t\t\tload:{type(e).__name__}:{str(e)[:80]}", flush=True)


if __name__ == "__main__":
    main()

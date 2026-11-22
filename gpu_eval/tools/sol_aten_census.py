"""Record the aten ops each (adapted) SOL problem's forward dispatches, flagging
data-dependent-shape ops and in-place mutation.  Usage:
sol_aten_census.py OUT.json FILE..."""
import importlib.util, json, sys
import torch
from torch.utils._python_dispatch import TorchDispatchMode

DATA_DEP = {"nonzero", "_local_scalar_dense", "unique", "_unique2", "unique_consecutive",
            "bincount", "masked_select", "repeat_interleave", "argwhere", "histc"}


class Rec(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        n = func._schema.name.split("::")[-1]
        ov = func._overloadname
        key = n + ("." + ov if ov and ov != "default" else "")
        self.ops[key] = self.ops.get(key, 0) + 1
        return func(*args, **(kwargs or {}))


def main():
    out = {}
    for p in sys.argv[2:]:
        stem = p.split("/")[-2] + "/" + p.split("/")[-1][:-3]
        try:
            spec = importlib.util.spec_from_file_location("_m", p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            torch.manual_seed(0)
            ins = m.get_inputs()
            model = m.Model(*(m.get_init_inputs() if hasattr(m, "get_init_inputs") else []))
            r = Rec()
            with r, torch.no_grad():
                model(*ins)
            names = {k.split(".")[0] for k in r.ops}
            out[stem] = {"ops": r.ops,
                         "data_dep": sorted(n for n in names if n in DATA_DEP),
                         "inplace": sorted(n for n in names if n.endswith("_") and not n.startswith("_"))}
        except Exception as e:
            out[stem] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
    json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()

"""Adapt SOL-ExecBench problems (KernelBench layout) for ModelBlaster's
--bench-file extractor.

SOL files differ from KernelBench in two ways that matter to MB:
  * shapes are locals inside get_inputs() (MB's shrinker halves module-level
    int attrs), and the default sizes are GPU-scale (GBs of bf16 inputs);
  * tensors are bf16/fp16 while MB's verified pipeline is fp32.

So per file we
  1. rewrite torch.bfloat16 / torch.float16 / torch.half -> torch.float32 and
     .bfloat16() / .half() -> .float()   (the problem is evaluated in fp32);
  2. hoist `name = <int>` statements at the top level of get_inputs() /
     get_init_inputs() to module globals (the function then reads the global);
  3. halve the largest shrinkable int global until the inputs fit TARGET_MB,
     keeping only steps after which get_inputs() + forward still run (meta
     device when the problem supports it, else CPU at the shrunk size).

Writes OUT/<level>/<stem>.py and OUT/<level>/<stem>.adapt.json (original -> new dims).
Usage: sol_adapt.py OUT TARGET_MB FILE...
"""
import ast, contextlib, importlib.util, json, os, re, sys, time
import torch

# dims that define the op rather than its size; never shrunk
STRUCT = re.compile(r"(^|_)(head_?dim|top_?k|k|kernel|stride|pad|padding|dilation|window|"
                    r"groups?|patch|rank|bits|block|order|stage|num_layers|layers|spatial_merge|"
                    r"merge|temporal|ratio|factor|eps|seed|n_?rep)(_|$|s$)", re.I)
BATCHY = re.compile(r"(batch|bs$|num_seqs|num_seq|n_seq)", re.I)
SEQY = re.compile(r"(seq|token|len|length|time|frames)", re.I)


def group(k):
    """shrink order: batch-like dims first, then sequence-like, then the rest"""
    return 1 if SEQY.search(k) else 0 if BATCHY.search(k) else 2


def floor(k):
    if group(k) == 2 and re.match(r"_*(num|n)_", k, re.I):   # counts (heads, experts, ...)
        return 2
    return (1, 16, 32)[group(k)]

DTYPES = {"bfloat16": "float32", "float16": "float32", "half": "float32"}
CREATE = ["rand", "randn", "zeros", "ones", "empty", "full", "randint", "arange",
          "tensor", "eye", "linspace", "randperm", "normal", "as_tensor"]


class Rewrite(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if (isinstance(node.value, ast.Name) and node.value.id == "torch"
                and node.attr in DTYPES):
            return ast.copy_location(ast.Attribute(node.value, DTYPES[node.attr], node.ctx), node)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in ("bfloat16", "half") and not node.args:
            f.attr = "float"
        return node


def hoist(tree):
    """Move top-level int-literal assignments out of get_inputs/get_init_inputs."""
    module_names = {t.id for s in tree.body if isinstance(s, ast.Assign)
                    for t in s.targets if isinstance(t, ast.Name)}
    hoisted, new_globals = {}, []
    for fn in tree.body:
        if not (isinstance(fn, ast.FunctionDef) and fn.name in ("get_inputs", "get_init_inputs")):
            continue
        # a name assigned again later in the body (loop counters, rebinding) stays local
        counts = {}
        for s in ast.walk(fn):
            for t in getattr(s, "targets", []) + ([s.target] if isinstance(s, (ast.AugAssign, ast.For, ast.AnnAssign)) else []):
                if isinstance(t, ast.Name):
                    counts[t.id] = counts.get(t.id, 0) + 1
        keep = []
        for s in fn.body:
            if (isinstance(s, ast.Assign) and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                    and isinstance(s.value, ast.Constant) and type(s.value.value) is int
                    and counts.get(s.targets[0].id) == 1
                    and s.targets[0].id not in module_names
                    and s.targets[0].id not in hoisted):
                hoisted[s.targets[0].id] = s.value.value
                new_globals.append(s)
            else:
                keep.append(s)
        fn.body = keep or [ast.Pass()]
    # insert hoisted globals right after the imports
    at = max([i + 1 for i, s in enumerate(tree.body) if isinstance(s, (ast.Import, ast.ImportFrom))] or [0])
    tree.body[at:at] = new_globals
    return hoisted


@contextlib.contextmanager
def creation_on(device):
    saved = {n: getattr(torch, n) for n in CREATE if hasattr(torch, n)}

    def wrap(fn):
        def inner(*a, **k):
            if device != "cpu" or "device" in k:   # CPU: only redirect explicit (cuda) devices
                k["device"] = device
            return fn(*a, **k)
        return inner
    try:
        for n, fn in saved.items():
            setattr(torch, n, wrap(fn))
        yield
    finally:
        for n, fn in saved.items():
            setattr(torch, n, fn)


def load(src, tag):
    spec = importlib.util.spec_from_loader(tag, loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, tag, "exec"), mod.__dict__)
    return mod


def int_globals(tree):
    """top-level `NAME = <int literal>` statements (the shrinkable dims)"""
    return {s.targets[0].id: s.value.value for s in tree.body
            if isinstance(s, ast.Assign) and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name) and isinstance(s.value, ast.Constant)
            and type(s.value.value) is int}


def with_dims(tree, dims):
    """source with the given top-level int literals substituted; derived globals
    (`_I = _H * _D`) are recomputed when the module is executed"""
    t = ast.parse(ast.unparse(tree))
    for s in t.body:
        if (isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name)
                and s.targets[0].id in dims and isinstance(s.value, ast.Constant)):
            s.value = ast.Constant(dims[s.targets[0].id])
    return ast.unparse(t)


def input_bytes(mod, device):
    torch.manual_seed(0)
    with creation_on(device):
        ins = mod.get_inputs()
    return sum(t.numel() * t.element_size() for t in ins if torch.is_tensor(t))


def probe(mod, device):
    """Input bytes at the module's current dims; raises if get_inputs/forward fail."""
    torch.manual_seed(0)
    with creation_on(device):
        ins = mod.get_inputs()
        m = mod.Model(*(mod.get_init_inputs() if hasattr(mod, "get_init_inputs") else []))
        with torch.no_grad():
            m(*ins)
    return sum(t.numel() * t.element_size() for t in ins if torch.is_tensor(t))


CPU_CHECK_MAX = 1 << 30   # cpu-validated problems: skip the forward above this size


def check(tree, dims, device):
    """bytes at `dims`, or raise if the problem no longer runs there. device:
    'meta' (forward runs on meta), 'cpu' (forward needs data; meta sizing),
    'cpu!' (even get_inputs needs data)."""
    mod = load(with_dims(tree, dims), "_sol_probe")
    if device == "meta":
        return probe(mod, "meta")
    nb = input_bytes(mod, "meta") if device == "cpu" else None
    if nb is None or nb <= CPU_CHECK_MAX:
        nb = probe(mod, "cpu")
    return nb


def shrink(tree, dims, target, device):
    dims = dict(dims)
    hist, frozen = {}, set()
    nbytes = check(tree, dims, device)
    for _ in range(300):
        if nbytes <= target:
            break
        cands = {k: v for k, v in dims.items()
                 if k not in frozen and not STRUCT.search(k.strip("_")) and v > floor(k)}
        if not cands:
            break
        g = min(group(k) for k in cands)
        k = max((k for k in cands if group(k) == g), key=cands.get)
        old = dims[k]
        dims[k] = max(floor(k), old // 2)
        try:
            nb = check(tree, dims, device)
        except Exception:
            dims[k] = old
            frozen.add(k)
            continue
        hist.setdefault(k, old)
        nbytes = nb
    return dims, hist, nbytes


def main():
    out, target = sys.argv[1], int(float(sys.argv[2]) * 2**20)
    for p in sys.argv[3:]:
        level, stem = p.rstrip("/").split("/")[-2], os.path.basename(p)[:-3]
        t0 = time.time()
        tree = ast.parse(open(p).read())
        tree = Rewrite().visit(tree)
        hoisted = hoist(tree)
        ast.fix_missing_locations(tree)
        src = ast.unparse(tree)
        dims0 = {k: v for k, v in int_globals(tree).items() if v > 1}
        rec = {"hoisted": hoisted}
        mod = load(src, f"_sol_{stem}")
        dev = "meta"
        try:
            probe(mod, "meta")
        except Exception:
            try:
                input_bytes(mod, "meta")
                dev = "cpu"
            except Exception:
                dev = "cpu!"
        dims = dims0
        try:
            dims, hist, nb = shrink(tree, dims0, target, dev)
            try:
                nb = probe(load(with_dims(tree, dims), "_sol_final"), "cpu")   # real data at the final size
            except Exception:
                if dev != "meta":
                    raise
                # meta tensors skip data checks (embedding/gather bounds):
                # redo the shrink validating every step on real data
                dev = "cpu"
                dims, hist, nb = shrink(tree, dims0, target, dev)
                nb = probe(load(with_dims(tree, dims), "_sol_final"), "cpu")
            rec.update(device=dev, shrunk={k: [v, dims[k]] for k, v in hist.items()},
                       in_mb=round(nb / 2**20, 2), ok=True)
        except Exception as e:
            rec.update(device=dev, ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
        out_src = with_dims(tree, dims)
        os.makedirs(f"{out}/{level}", exist_ok=True)
        hdr = (f"# Adapted from SOL-ExecBench {level}/{stem}.py by gpu_eval/tools/sol_adapt.py:\n"
               f"# bf16/fp16 -> fp32; get_inputs dims hoisted; shrunk {rec.get('shrunk', {})}\n")
        with open(f"{out}/{level}/{stem}.py", "w") as f:
            f.write(hdr + out_src + "\n")
        rec["secs"] = round(time.time() - t0, 1)
        print(f"{level}/{stem}\t{rec.get('ok')}\t{rec.get('in_mb')}\t{dev}\t{rec.get('shrunk', rec.get('error'))}", flush=True)
        json.dump(rec, open(f"{out}/{level}/{stem}.adapt.json", "w"), indent=1, sort_keys=True)


if __name__ == "__main__":
    main()

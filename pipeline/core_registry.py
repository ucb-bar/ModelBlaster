"""Heterogeneous Core registry — describes what hardware is available.

The registry is consumed by:
  * extract_graph (validates per-dispatch hardware_target values),
  * XPU-RT (picks a target per dispatch from the registry's list,
    balancing capability + cost),
  * the runtime stub generator (maps each `kind` to a worker queue;
    today's collapse-to-one-pthreadpool runtime ignores everything
    except whether a target is reachable).

A registry is a small JSON document. See agents/cores/*.json for
shipped examples (spike_rocket_quad, spike_rvv_quad, and a heterogeneous
chipyard_hetero_example with rocket+boom+rvv+gemmini).

JSON schema (informal):

    {
      "system": "<short tag>",
      "description": "<human-readable>",
      "default_dispatch_target": "any",
      "cores": [
        {
          "name": "<unique-name>",
          "kind": "<scalar|rvv|boom|gemmini|...>",
          "isa": "<isa string>",          // optional, for documentation
          "harts": [<int>...],            // hart ids this core covers
          "capabilities": ["<op>", ...],   // op kinds the core can execute
          "capabilities_ref": "<core-name>" // optional alias to share with another core
        },
        ...
      ]
    }

Capabilities form the contract between the registry and the IR's
per-dispatch hardware_target field:
  * "any" — XPU-RT picks any core whose capabilities include the
    dispatch's `op`.
  * "<kind>" — must run on a core of that kind that supports the op.
  * "<core-name>" — must run on this specific core.

The loader resolves capabilities_ref hops (so registries can avoid
duplicating long capability lists) and validates uniqueness +
capability coverage.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Core:
    name: str
    kind: str
    harts: tuple[int, ...]
    capabilities: tuple[str, ...]
    isa: str = ""
    description: str = ""


@dataclass
class CoreRegistry:
    system: str
    description: str
    default_dispatch_target: str
    cores: tuple[Core, ...]

    # Convenience indices — built by load() so consumers don't repeat
    # the work.
    by_name: dict[str, Core] = field(default_factory=dict)
    by_kind: dict[str, tuple[Core, ...]] = field(default_factory=dict)

    def kinds(self) -> set[str]:
        return set(self.by_kind.keys())

    def names(self) -> set[str]:
        return set(self.by_name.keys())

    def cores_for_op(self, op: str) -> tuple[Core, ...]:
        """Return cores whose capability list covers `op`."""
        return tuple(c for c in self.cores if op in c.capabilities)

    def resolve_target(self, hardware_target: str, op: str) -> tuple[Core, ...]:
        """Given a per-dispatch hardware_target ("any" / kind / name),
        return the candidate cores that satisfy it for the given op.
        Empty tuple = no core can run this dispatch as constrained."""
        if hardware_target == "any":
            return self.cores_for_op(op)
        if hardware_target in self.by_name:
            c = self.by_name[hardware_target]
            return (c,) if op in c.capabilities else ()
        if hardware_target in self.by_kind:
            return tuple(c for c in self.by_kind[hardware_target]
                         if op in c.capabilities)
        return ()


def load(path: str) -> CoreRegistry:
    """Load a registry JSON, resolve capabilities_ref aliases, validate
    invariants. Raises ValueError on malformed input."""
    with open(path) as f:
        doc = json.load(f)

    if "cores" not in doc or not isinstance(doc["cores"], list):
        raise ValueError(f"{path}: missing or non-list 'cores'")

    raw_by_name: dict[str, dict] = {}
    for entry in doc["cores"]:
        if "name" not in entry:
            raise ValueError(f"{path}: every core needs a 'name'")
        if entry["name"] in raw_by_name:
            raise ValueError(f"{path}: duplicate core name {entry['name']!r}")
        raw_by_name[entry["name"]] = entry

    cores: list[Core] = []
    for entry in doc["cores"]:
        caps = entry.get("capabilities")
        if caps is None and "capabilities_ref" in entry:
            ref = entry["capabilities_ref"]
            if ref not in raw_by_name:
                raise ValueError(
                    f"{path}: {entry['name']}.capabilities_ref={ref!r} "
                    f"points to unknown core")
            caps = raw_by_name[ref].get("capabilities")
            if caps is None:
                raise ValueError(
                    f"{path}: {entry['name']}.capabilities_ref={ref!r} "
                    f"resolves to a core without explicit capabilities")
        if caps is None:
            raise ValueError(
                f"{path}: core {entry['name']!r} has neither capabilities "
                f"nor capabilities_ref")
        if "kind" not in entry:
            raise ValueError(
                f"{path}: core {entry['name']!r} missing 'kind'")
        cores.append(Core(
            name=entry["name"],
            kind=entry["kind"],
            harts=tuple(entry.get("harts", [])),
            capabilities=tuple(caps),
            isa=entry.get("isa", ""),
            description=entry.get("description", ""),
        ))

    by_name = {c.name: c for c in cores}
    by_kind: dict[str, list[Core]] = {}
    for c in cores:
        by_kind.setdefault(c.kind, []).append(c)

    return CoreRegistry(
        system=doc.get("system", os.path.basename(path).rsplit(".", 1)[0]),
        description=doc.get("description", ""),
        default_dispatch_target=doc.get("default_dispatch_target", "any"),
        cores=tuple(cores),
        by_name=by_name,
        by_kind={k: tuple(v) for k, v in by_kind.items()},
    )


def validate_dispatch_targets(reg: CoreRegistry,
                              ops: Iterable[dict]) -> list[str]:
    """Check that every non-view op's (op, hardware_target) pair is
    realisable on the given registry. Returns a list of human-readable
    errors; empty = all good."""
    errors: list[str] = []
    valid_targets = {"any"} | reg.kinds() | reg.names()
    for op in ops:
        if op.get("op") == "view":
            continue
        tgt = op.get("hardware_target", "any")
        if tgt not in valid_targets:
            errors.append(
                f"dispatch {op.get('dispatch_id')} ({op.get('name')}): "
                f"unknown hardware_target {tgt!r}; "
                f"must be one of {sorted(valid_targets)}"
            )
            continue
        if not reg.resolve_target(tgt, op.get("op", "")):
            errors.append(
                f"dispatch {op.get('dispatch_id')} ({op.get('name')}): "
                f"no core satisfies hardware_target={tgt!r} "
                f"for op={op.get('op')!r}"
            )
    return errors


# --- CLI for ad-hoc inspection / validation -----------------------------

def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="Inspect / validate an agents/cores/*.json registry")
    ap.add_argument("registry", help="path to a registry JSON")
    ap.add_argument("--ir", default=None,
                    help="optional graph.json — checks every dispatch's "
                         "hardware_target is realisable on this registry")
    args = ap.parse_args()
    reg = load(args.registry)
    print(f"system: {reg.system}")
    print(f"  default_dispatch_target: {reg.default_dispatch_target}")
    print(f"  {len(reg.cores)} cores in {len(reg.kinds())} kinds: "
          f"{sorted(reg.kinds())}")
    for c in reg.cores:
        print(f"    {c.name:<10} kind={c.kind:<8} harts={list(c.harts)} "
              f"caps={len(c.capabilities)}")
    if args.ir:
        ir = json.load(open(args.ir))
        errs = validate_dispatch_targets(reg, ir.get("ops", []))
        if errs:
            print(f"\n{len(errs)} dispatch-target errors:")
            for e in errs:
                print(f"  {e}")
            raise SystemExit(1)
        print(f"\nIR ok: {len(ir.get('ops', []))} ops match registry")


if __name__ == "__main__":
    _main()

import json, re
SYS="You compare two lessons learned while optimizing GPU kernels. Answer with JSON only."
def meta(c):
    s=f"[{c['kind']}] op={c['op_class']} technique={c['technique']} state={c['state']} scope={'problem '+str(c['problem']) if c['problem'] else 'global'}"
    if c.get('pol'): s+=f" {json.dumps(c['pol'])}"
    return s
def prompt_v1(a,b):
    return (f"A: {meta(a)}\n{a['text']}\n\nB: {meta(b)}\n{b['text']}\n\n"
      "Relation of A to B? Labels: same (same technique, condition and effect; different example shapes, numbers or wording do not matter) | "
      "a_covers_b (A is the more general statement and fully contains B) | b_covers_a | related (same topic but a distinct lesson: different condition, mechanism or effect) | "
      "different | contradict (opposite effect under the same condition).\n"
      'JSON: {"label":"...","why":"<=15 words"}')
def prompt_v2(a,b):
    return (f"Statement 1: {meta(a)}\n{a['text']}\n\nStatement 2: {meta(b)}\n{b['text']}\n\n"
      "Would a knowledge base lose anything useful if these were stored as ONE entry? Pick one: "
      "identical (same lesson) | one_general (statement 1 is broader and includes statement 2) | two_general (statement 2 is broader and includes statement 1) | "
      "distinct (different lessons, even if related) | opposed (they disagree).\n"
      'JSON: {"verdict":"...","why":"<=15 words"}')
M1={'same':'same','a_covers_b':'a_covers_b','b_covers_a':'b_covers_a','related':'related','different':'different','contradict':'contradict'}
M2={'identical':'same','one_general':'a_covers_b','two_general':'b_covers_a','distinct':'related','opposed':'contradict'}
def parse(t, m, key):
    mm=re.search(r'\{.*\}',t or '',re.S)
    if not mm: return None
    try: d=json.loads(mm.group(0))
    except Exception: return None
    return m.get(str(d.get(key,'')).strip().lower())
FOLD={'same','a_covers_b','b_covers_a'}
def swap(l):
    return {'a_covers_b':'b_covers_a','b_covers_a':'a_covers_b'}.get(l,l)

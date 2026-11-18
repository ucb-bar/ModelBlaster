cd /tmp/krisd/cl/runs
for d in *; do [ -f $d/recompose.json ] && python3 -c "
import json
r=json.load(open('$d/recompose.json'))
m=r['median_us']; g=r.get('graph',{})
rep=r.get('reuse') or {}
print('$d', {k: round(v,1) if v else v for k,v in m.items()}, 'identical', r.get('as_shipped_identical'), 'evals', rep.get('evals'), 'reuse_s', r.get('reuse_s'),
 'kinds', [(k['kind'], k.get('before_us'), k.get('after_us'), [w['from'] for w in k.get('written',[])]) for k in rep.get('kinds',[])],
 'graph', {k:(v.get('e2e_basis'), v.get('graph_nodes'), v.get('passed')) for k,v in g.items()},
 'conc', (r.get('concurrency_C') or {}).get('max_width'), (r.get('concurrency_C') or {}).get('lanes'))
"; done

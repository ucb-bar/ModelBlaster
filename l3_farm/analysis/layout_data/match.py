import json, glob, collections
R = {}
for l in open('conv.out'):
    if l.startswith('@@R@@'): R.update(json.loads(l[5:]))
RES = '/Users/krisd/Documents/ModelBlaster-clone/l3_farm/results'
out = {}
for n, r in R.items():
    L = {}
    for x in r['layers']:
        L[(tuple(x['inp']), x['oc'], x['k'][0], x['s'][0], x['g'])] = x
    rs = sorted(glob.glob(f'{RES}/{n}/work/r*/profile/*/profile.json'), key=lambda p: int(p.split('/work/r')[1][:2]))
    d = json.load(open(rs[-1])); mr = json.load(open(f'{RES}/{n}/work/model_runtime.json'))[n]
    ours = n_ = cn = cc = best = 0; miss = 0; rows = []
    for o in d['ops']:
        if 'conv2d' not in o['op'] or 'transpose' in o['op']: continue
        s = o['shape'].get('a', o['shape'])
        g = s.get('G', s.get('groups', 1))
        if 'dw' in o['op']: g = s['IC']
        k = ((s['N'], s['IC'], s['IH'], s['IW']), s['OC'], s['KH'], s['SH'], g)
        x = L.get(k)
        if not x: miss += 1; continue
        ours += o['us']; cn += x['nchw_us']; cc += x['nhwc_us']; best += min(o['us'], x['nhwc_us'])
        rows.append((k, o['us'], x['nchw_us'], x['nhwc_us']))
    out[n] = dict(ours_e2e=mr['model_e2e_us'], torch_nchw_graph=r['nchw_bench1_us'], torch_cl_graph=r['cl_bench1_us'],
                  cudnn_nchw_layout_kernels=r['nchw_prof'][0].get('layout/copy'), ours_conv=round(ours), cudnn_nchw_conv=round(cn),
                  cudnn_nhwc_conv=round(cc), ours_or_nhwc_best=round(best), saving=round(ours - best), missing=miss,
                  bound_e2e=round(mr['model_e2e_us'] - (ours - best)), bound_speedup=round(mr['model_e2e_us'] / (mr['model_e2e_us'] - (ours - best)), 3))
    print(n, json.dumps(out[n]))
    if n in ('11_VGG16', '10_ResNet101'):
        for row in sorted(rows, key=lambda r: -r[1])[:8]: print('    ', row)
json.dump(out, open('conv_match.json', 'w'), indent=1)

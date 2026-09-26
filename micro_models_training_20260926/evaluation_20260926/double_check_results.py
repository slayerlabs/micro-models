"""Independent aggregation and bookkeeping checks; never fits a model."""
import json,math,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;root=P.parent
r=json.loads((P/'results.json').read_text());a=np.load(P/'token_scores.npz');f=json.loads((P/'frozen_protocol.json').read_text())
checks={};maxerror=0
for split in ['validation','test']:
    records=[json.loads(l) for l in (root/f'data/{split}.jsonl').read_text().splitlines()]
    target=a[split+'__target'];owner=a[split+'__owner'];body=a[split+'__body']
    for name,v in r[split]['models'].items():
        losses=a[split+'__'+name+'__nll']
        nll=math.fsum(float(x) for x,b in zip(losses,body) if b)/sum(bool(x) for x in body)
        maxerror=max(maxerror,abs(math.exp(nll)-v['body_ppl']))
        for i,record in enumerate(records):
            ix=(owner==i)&body;actual=math.fsum(map(float,losses[ix]))
            expected=v['per_record'][i]
            assert expected['id']==record['id'] and int(ix.sum())==expected['body_chars']==len(record['body'])
            assert abs(actual-expected['body_nll'])<1e-9
    checks[split+'_per_record_alignment']=True
checks['scalar_vs_vector_aggregation_max_PPL_difference']=maxerror;assert maxerror<1e-12
test=r['test']['models'];families=sorted({x['family_id'] for x in test['NPLM']['per_record']})
def group(model,fam,field):return math.fsum(row[field] for row in test[model]['per_record'] if row['family_id']==fam)
rng=np.random.default_rng(f['uncertainty']['seed']);ix=rng.integers(len(families),size=(10000,len(families)))
delta=[]
for chosen in ix:
    count=math.fsum(group('NPLM',families[i],'body_chars') for i in chosen)
    na=math.fsum(group('NPLM',families[i],'body_nll') for i in chosen)
    nb=math.fsum(group('GPT_B',families[i],'body_nll') for i in chosen)
    delta.append((na-nb)/count)
ci=np.quantile(delta,[.025,.975]);ref=r['test']['comparisons'][0]['delta_ci95']
checks['primary_bootstrap_independent_CI_max_difference']=float(np.max(np.abs(ci-ref)))
assert checks['primary_bootstrap_independent_CI_max_difference']<1e-12
g=json.loads((P/'generation.json').read_text());seen=set()
for s in g['samples']:
    key=(s['model'],s['prompt_index'],s['seed']);assert key not in seen;seen.add(key)
    assert hashlib.sha256((P/s['file']).read_bytes()).hexdigest()==s['sha256']
assert len(seen)==90
for m in f['generation']['models']:
    expected={(m,p,s) for p in [1,2,3] for s in f['generation']['seeds']};assert expected<=seen
checks['all_90_predeclared_samples_present_and_unchanged']=True
for path,digest in f['checkpoint_sha256'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
checks['frozen_weights_unchanged']=True
proof=json.loads((P/'notebook_execution.json').read_text());assert proof['errors']==0 and proof['figure_outputs']==5
checks['notebook_outputs_present']=True
checks['passed']=True
(P/'double_check_results.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))

"""Validation-only CKA and identity-stitch control; no fitting."""
import json
import numpy as np
import torch
from evaluate_campaign import OUT, load_models, load_data, windows, save
from gpt import GPT

def cka(x,y):
    x=x-x.mean(0);y=y-y.mean(0)
    return float(np.square(x.T@y).sum()/np.sqrt(np.square(x.T@x).sum()*np.square(y.T@y).sum()))

@torch.no_grad()
def reps(model,data):
    output=[[] for _ in model.blocks]
    for i in range(0,len(data['x']),16):
        x=data['x'][i:i+16];mask=data['body'][i:i+16]
        h=model.drop(model.tok_emb(x)+model.pos_emb(torch.arange(x.shape[1])))
        for j,b in enumerate(model.blocks):
            h=b(h);output[j].append(h[mask].double().numpy())
    output=[np.concatenate(o) for o in output]
    ix=np.linspace(0,len(output[0])-1,min(1536,len(output[0])),dtype=int)
    return [o[ix] for o in output]

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    models,_=load_models();rows,vocab,_=load_data('validation');data=windows(rows,vocab)
    for n,s in [('random_A',9901),('random_B',9902)]:
        torch.manual_seed(s);models[n]=GPT(models['GPT_A'].cfg).eval()
    r={n:reps(m,data) for n,m in models.items() if n!='NPLM'}
    perm=np.random.default_rng(9903).permutation(len(r['GPT_A'][0]))
    pairs=[('GPT_A','GPT_B'),('GPT_A','random_A'),('random_A','random_B')]
    res={a+'__'+b:[cka(x,y) for x,y in zip(r[a],r[b])] for a,b in pairs}
    res['shuffled_alignment']=[cka(x,y[perm]) for x,y in zip(r['GPT_A'],r['GPT_B'])]
    res['self']=[cka(x,x) for x in r['GPT_A']]
    checks={}
    x=r['GPT_A'][0][:256];y=r['GPT_B'][0][:256];x-=x.mean(0);y-=y.mean(0)
    a=x@x.T;b=y@y.T
    independent=float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
    checks['feature_vs_gram_formula_difference']=abs(cka(x,y)-independent)
    assert checks['feature_vs_gram_formula_difference']<1e-12
    # E0 identity control, not trained mapper experiment.
    m=models['GPT_A'];x=data['x'][:8]
    with torch.no_grad():
        baseline=m(x)[0];h=m.drop(m.tok_emb(x)+m.pos_emb(torch.arange(x.shape[1])))
        for i,b in enumerate(m.blocks):
            if i==2:h=h@torch.eye(m.cfg.n_embd)
            h=b(h)
        stitched=m.head(m.ln_f(h))
    checks['E0_identity_max_absolute_logit_difference']=float((baseline-stitched).abs().max())
    assert checks['E0_identity_max_absolute_logit_difference']==0
    save('representations.json',{'split':'validation','observations':len(perm),'layers':[1,2,3,4],
        'CKA':res,'mean_CKA':{n:float(np.mean(v)) for n,v in res.items()},'checks':checks,
        'limitations':['One trained pair, one random pair, no scale sweep or uncertainty estimates',
        'Shared tokens and positions contribute to representation similarity',
        'Identity control checks wiring, not ability to learn a mapper or cross-model stitching']})
    print(json.dumps({n:np.mean(v) for n,v in res.items()}),flush=True)

if __name__=='__main__':main()

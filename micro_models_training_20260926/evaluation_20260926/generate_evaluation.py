"""All 90 frozen prompt/seed samples. No retries, filtering or music repair."""
import json, math, warnings
import numpy as np
import torch
import music21
from music21 import converter
from evaluate_campaign import OUT, ROOT, load_models, save, sha

@torch.no_grad()
def generate(model,vocab,prompt,seeds,cfg):
    initial=[1]+[vocab['stoi'][c] for c in prompt]
    x=torch.tensor([initial for _ in seeds])
    rngs=[torch.Generator().manual_seed(s) for s in seeds]
    outputs=[[] for _ in seeds];done=[False]*len(seeds)
    for step in range(cfg['max_new_tokens']):
        logits=model(x[:,-128:])[0][:,-1,:]/cfg['temperature']
        logits[:,[0,1]]=-torch.inf
        thresholds=logits.topk(cfg['top_k'],dim=1).values[:,-1:]
        logits[logits<thresholds]=-torch.inf
        ps=logits.softmax(-1);tokens=[]
        for i,rng in enumerate(rngs):
            t=2 if done[i] else int(torch.multinomial(ps[i],1,generator=rng))
            if not done[i]:outputs[i].append(t)
            if t==2:done[i]=True
            tokens.append(t)
        x=torch.cat([x,torch.tensor(tokens)[:,None]],1)
        if all(done):break
    inv={v:k for k,v in vocab['stoi'].items()}
    return [(prompt+''.join(inv[t] for t in out if t in inv),out,'EOS' if d else 'token_limit') for out,d in zip(outputs,done)]

def inspect(text):
    out={'parser_accepted':False,'interior_measures':0,'bad_interior_measures':0,'has_testable_meter':False}
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            score=converter.parseData(text,format='abc')
            notes=list(score.recurse().notes)
            if not notes:raise ValueError('No notes')
            ms=list(score.recurse().getElementsByClass('Measure'))[1:-1]
            bad=[{'measure':i+1,'duration':float(m.duration.quarterLength),'expected':float(m.barDuration.quarterLength)}
                 for i,m in enumerate(ms) if not math.isclose(float(m.duration.quarterLength),float(m.barDuration.quarterLength),abs_tol=1e-8)]
            out.update(parser_accepted=True,notes=len(notes),interior_measures=len(ms),bad_interior_measures=len(bad),
                bad_details=bad,has_testable_meter=bool(ms),warnings=[str(w.message) for w in seen])
    except Exception as e:out['error']=f'{type(e).__name__}: {e}'
    return out

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    models,_=load_models();f=json.loads((OUT/'frozen_protocol.json').read_text());cfg=f['generation']
    vocab=json.loads((ROOT/'data/vocab.json').read_text());dest=OUT/'raw_samples';dest.mkdir(exist_ok=True)
    allrows=[]
    for name in cfg['models']:
        for pi,prompt in enumerate(cfg['prompts']):
            samples=generate(models[name],vocab,prompt,cfg['seeds'],cfg)
            for seed,(text,tokens,reason) in zip(cfg['seeds'],samples):
                path=dest/f'{name}_prompt{pi+1}_seed{seed}.abc';path.write_text(text)
                row={'model':name,'prompt_index':pi+1,'seed':seed,'stop':reason,'tokens':len(tokens),
                    'file':str(path.relative_to(OUT)),'sha256':sha(path),**inspect(text)}
                row['complete_and_meter_check_pass']=reason=='EOS' and row['parser_accepted'] and row['has_testable_meter'] and row['bad_interior_measures']==0
                allrows.append(row)
            print(name,'prompt',pi+1,'completed',flush=True)
            save('generation.json',{'status':'running','samples':allrows})
    summary={}
    for name in cfg['models']:
        s=[r for r in allrows if r['model']==name]
        summary[name]={'samples':len(s),'parser_accepted':sum(r['parser_accepted'] for r in s),'EOS':sum(r['stop']=='EOS' for r in s),
                      'with_interior_meter_errors':sum(r['bad_interior_measures']>0 for r in s),
                      'without_testable_meter':sum(not r['has_testable_meter'] for r in s),
                      'bad_measures':sum(r['bad_interior_measures'] for r in s),
                      'interior_measures':sum(r['interior_measures'] for r in s),
                      'complete_and_meter_check_pass':sum(r['complete_and_meter_check_pass'] for r in s),
                      'median_generated_tokens':float(np.median([r['tokens'] for r in s]))}
    save('generation.json',{'status':'completed','music21':music21.__version__,'torch':str(torch.__version__),
        'protocol_sha256':sha(OUT/'frozen_protocol.json'),'summary':summary,'samples':allrows,
        'limitations':['90 fixed samples, 3 prompts, no population-wide musical quality estimate',
            'Parser acceptance is not strict ABC validation; missing/extra bars may be normalized',
            'First/last measures excluded; extremely short pieces do not pass composite measure',
            'No human listening study, no learned judge, no grammar correction',
            'Composite complete-and-meter check is diagnostic, not musicality']})
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

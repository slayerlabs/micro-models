"""Frozen-model held-out evaluation; no optimizer, fitting or data reassignment."""
import json, math, pickle, sys, datetime, platform
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT/'src'))
from protocol import load_data, windows, evaluate, sha, check_gate, encode
from train import NPLM
from gpt import GPT, GPTConfig

def save(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')

def load_models():
    specs = {'NPLM':'nplm_92c0801c_seed_20260926',
             'GPT_A':'gpt_92c0801c_seed_20260926', 'GPT_B':'gpt_92c0801c_seed_20260927'}
    models = {}
    for name, run in specs.items():
        c = torch.load(ROOT/'runs'/run/'checkpoint_best.pt', map_location='cpu', weights_only=True)
        model = GPT(GPTConfig(**c['model_config'])) if c['model_type']=='gpt' else NPLM(**c['model_config'])
        model.load_state_dict(c['model']); model.eval()
        models[name] = model
    return models, specs

@torch.no_grad()
def neural_probs(model, data, batch=32):
    result=[]
    for i in range(0,len(data['x']),batch):
        logits,_ = model(data['x'][i:i+batch])
        result.append(F.log_softmax(logits, -1)[data['y'][i:i+batch]!=-100].double().numpy())
    return np.concatenate(result)

def ngram_probs(rows,vocab,counts,order):
    size=vocab['size'];uni=counts[0][()]
    base=(np.array([uni.get(i,0) for i in range(size)],dtype=float)+.01)/(sum(uni.values())+.01*size)
    output=[]
    for r in rows:
        ids=encode(r,vocab)
        for j in range(1,len(ids)-1):
            p=base.copy();den=1.
            for n in range(1,min(order,j)+1):
                cnt=counts[n].get(tuple(ids[j-n:j]))
                if cnt:
                    w=2.**n;total=sum(cnt.values())
                    for t,c in cnt.items():p[t]+=w*c/total
                    den+=w
            output.append(np.log(p/den))
    return np.array(output)

def summarize(logp, target, owners, body, rows):
    nll=-logp[np.arange(len(target)),target]
    correct=logp.argmax(1)==target
    per=[]
    for i,r in enumerate(rows):
        mask=owners==i;bm=mask&body
        assert mask.sum()==len(r['text']) and bm.sum()==len(r['body'])
        per.append({'id':r['id'],'family_id':r['family_id'],'chars':int(mask.sum()),
                    'body_chars':int(bm.sum()),'nll':float(nll[mask].sum()),'body_nll':float(nll[bm].sum())})
    return {'full_ppl':float(np.exp(nll.mean())), 'body_ppl':float(np.exp(nll[body].mean())),
            'body_nll':float(nll[body].mean()),'body_bpc':float(nll[body].mean()/math.log(2)),
            'body_accuracy':float(correct[body].mean()),'n_characters':len(target),'n_body':int(body.sum()),
            'per_record':per},nll,correct

def bootstrap(summaries,rows,freeze):
    names=list(summaries);groups=sorted({r['family_id'] for r in rows})
    gidx={g:i for i,g in enumerate(groups)};loss=np.zeros((len(groups),len(names)));count=np.zeros(len(groups))
    for k,n in enumerate(names):
        for r in summaries[n]['per_record']:
            i=gidx[r['family_id']];loss[i,k]+=r['body_nll']
            if k==0:count[i]+=r['body_chars']
    rng=np.random.default_rng(freeze['uncertainty']['seed'])
    ix=rng.integers(len(groups),size=(freeze['uncertainty']['replicates'],len(groups)))
    b=loss[ix].sum(1)/count[ix].sum(1)[:,None]
    for k,n in enumerate(names):
        summaries[n]['body_ppl_ci95']=np.exp(np.quantile(b[:,k],[.025,.975])).tolist()
        summaries[n]['family_macro_nll']=float((loss[:,k]/count).mean())
        summaries[n]['family_macro_ppl']=float(np.exp((loss[:,k]/count).mean()))
    comparisons=[('NPLM','GPT_B'),('GPT_B','ngram_context6'),('NPLM','ngram_context6'),
                 ('GPT_AB_prob','GPT_B'),('GPT_AB_logits','GPT_B'),
                 ('NPLM_GPT_B_prob','NPLM'),('NPLM_GPT_B_logits','NPLM')]
    out=[]
    for a,c in comparisons:
        ia,ic=names.index(a),names.index(c);delta=b[:,ia]-b[:,ic]
        point=summaries[a]['body_nll']-summaries[c]['body_nll']
        out.append({'A':a,'B':c,'delta_NLL_A_minus_B':point,'delta_ci95':np.quantile(delta,[.025,.975]).tolist(),
                    'PPL_ratio_A_over_B':math.exp(point),'PPL_ratio_ci95':np.exp(np.quantile(delta,[.025,.975])).tolist(),
                    'families_A_better':int((loss[:,ia]<loss[:,ic]).sum()),'families':len(groups),
                    'primary':a=='NPLM' and c=='GPT_B'})
    return out

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    f=json.loads((OUT/'frozen_protocol.json').read_text());check_gate()
    for p,d in f['checkpoint_sha256'].items():assert sha(ROOT/p)==d
    assert sha(ROOT/'runs/ngram_92c0801c/counts.pkl')==f['ngram_sha256']
    models,specs=load_models()
    with (ROOT/'runs/ngram_92c0801c/counts.pkl').open('rb') as h:counts=pickle.load(h)
    checks={};results={};npz={}
    for split in ['validation','test']:
        if split=='test':
            assert all(v<1e-6 for v in checks['validation_reproduction_absolute_errors'].values())
            save('test_access.json',{'first_scoring_started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'frozen_protocol_sha256':sha(OUT/'frozen_protocol.json'),'checkpoint_choice_frozen':True})
        rows,vocab,_=load_data(split);data=windows(rows,vocab)
        valid=data['y']!=-100
        target=data['y'][valid].numpy();body=data['body'][valid].numpy()
        owners=data['owners'][:,None].expand_as(data['y'])[valid].numpy()
        assert np.array_equal(target,np.concatenate([np.array(encode(r,vocab)[1:-1]) for r in rows]))
        probs={n:neural_probs(m,data) for n,m in models.items()}
        for o in [1,3,6]:probs[f'ngram_context{o}']=ngram_probs(rows,vocab,counts,o)
        for name,(a,b) in f['exploratory_ensembles'].items():
            probs[name+'_prob']=np.logaddexp(probs[a],probs[b])-math.log(2)
            lp=(probs[a]+probs[b])/2
            probs[name+'_logits']=lp-np.logaddexp.reduce(lp,axis=1)[:,None]
        summary={};losses={};correct={}
        for name,lp in probs.items():
            assert np.max(np.abs(np.exp(lp).sum(1)-1))<1e-5
            summary[name],losses[name],correct[name]=summarize(lp,target,owners,body,rows)
        if split=='validation':
            checks['validation_reproduction_absolute_errors']={n:abs(summary[n]['body_ppl']-json.loads((ROOT/'runs'/r/'best_validation.json').read_text())['body_ppl']) for n,r in specs.items()}
            prior=json.loads((ROOT/'runs/ngram_92c0801c/validation.json').read_text())
            for r in prior['results']:
                checks['validation_reproduction_absolute_errors'][f"ngram{r['order']}"]=abs(r['body_ppl']-summary[f"ngram_context{r['order']}"]['body_ppl'])
            # Independent original aggregator, and prefix-only causal check.
            checks['original_evaluator_agreement']={n:abs(evaluate(m,data,rows)['body_ppl']-summary[n]['body_ppl']) for n,m in models.items()}
            for n,m in models.items():
                with torch.no_grad():
                    x=data['x'][:2].clone();a=m(x)[0][:,:20]
                    x[:,20:]=(x[:,20:]+1)%vocab['size'];b=m(x)[0][:,:20]
                    assert torch.equal(a,b), 'Noncausal predictions'
            checks['causal_future_perturbation_passed']=True
        comparisons=bootstrap(summary,rows,f)
        charlist=np.array([vocab['chars'][int(t)-3] for t in target])
        classes={'note_letters':np.isin(charlist,list('ABCDEFGabcdefgz'))&body,
                 'duration_digits_slash':np.isin(charlist,list('0123456789/'))&body,
                 'space':(charlist==' ')&body,'barline':(charlist=='|')&body,
                 'accidental':np.isin(charlist,list('=^_'))&body}
        detail={n:{c:{'chars':int(mask.sum()),'nll':float(v[mask].mean()),'accuracy':float(correct[n][mask].mean())} for c,mask in classes.items()} for n,v in losses.items()}
        pair_diagnostics={}
        for a,b in [('GPT_A','GPT_B'),('NPLM','GPT_B')]:
            ca,cb=correct[a][body],correct[b][body]
            pair_diagnostics[a+'__'+b]={'both_correct':int((ca&cb).sum()),'A_only':int((ca&~cb).sum()),
                'B_only':int((~ca&cb).sum()),'neither':int((~ca&~cb).sum()),
                'argmax_disagreement':float((probs[a][body].argmax(1)!=probs[b][body].argmax(1)).mean()),
                'loss_correlation':float(np.corrcoef(losses[a][body],losses[b][body])[0,1])}
        results[split]={'records':len(rows),'families':len({r['family_id'] for r in rows}),'models':summary,
                        'comparisons':comparisons,'token_classes':detail,'complementarity':pair_diagnostics}
        for n,v in losses.items():npz[split+'__'+n+'__nll']=v
        for n,v in correct.items():npz[split+'__'+n+'__correct']=v
        npz[split+'__target']=target;npz[split+'__owner']=owners;npz[split+'__body']=body
        print(split, {n:round(v['body_ppl'],6) for n,v in summary.items()}, flush=True)
    save('results.json',results);np.savez_compressed(OUT/'token_scores.npz',**npz)
    checks['all_targets_scored_once']=True;checks['all_distributions_normalized']=True
    checks['frozen_weights_unchanged']=all(sha(ROOT/p)==d for p,d in f['checkpoint_sha256'].items())
    checks['runtime']={'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__}
    save('evaluation_checks.json',checks)

if __name__=='__main__':main()

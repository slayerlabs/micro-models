"""Compare uninterrupted and restored diagnostic updates in RAM; leave runs unchanged."""
import io
import json
import torch
from gpt import GPT,GPTConfig
from train import NPLM
from protocol import ROOT,check_gate,load_data,windows,sha

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
gate=check_gate()
rows,vocab,_=load_data('train')
data=windows(rows,vocab,include_eos=True)
def build(ck):
    model=GPT(GPTConfig(**ck['model_config'])) if ck['model_type']=='gpt' else NPLM(**ck['model_config'])
    model.load_state_dict(ck['model']);model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=ck['config']['base_lr'],betas=(.9,.99),weight_decay=.1)
    opt.load_state_dict(ck['optimizer'])
    rng=torch.Generator();rng.set_state(ck['sampler_rng'])
    torch.set_rng_state(ck['torch_rng'])
    return model,opt,rng
def update(model,opt,rng):
    indices=torch.randint(len(data['x']),(32,),generator=rng)
    _,loss=model(data['x'][indices],data['y'][indices])
    opt.zero_grad(set_to_none=True);loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(loss) and torch.isfinite(norm)
    opt.step()
    return indices,float(loss.detach())

out={'manifest_sha256':gate['manifest_sha256'],'method':'Two disposable updates: continuous vs torch.save/torch.load restore between updates',
     'diagnostic_only':True,'actual_run_steps_changed':False,'test_evaluated':False,'runs':[]}
for run in sorted((ROOT/'runs').glob('*_seed_*')):
    if not run.is_dir():continue
    original_hash=sha(run/'checkpoint_last.pt')
    ck=torch.load(run/'checkpoint_last.pt',weights_only=True,map_location='cpu')
    assert ck['step']==ck['planned_steps']==2000
    model,opt,rng=build(ck)
    update(model,opt,rng)
    buf=io.BytesIO()
    state={**{k:ck[k] for k in ['model_type','model_config','config']},
           'model':model.state_dict(),'optimizer':opt.state_dict(),
           'sampler_rng':rng.get_state(),'torch_rng':torch.get_rng_state()}
    torch.save(state,buf)
    continuous_indices,continuous_loss=update(model,opt,rng)
    expected={k:v.detach().clone() for k,v in model.state_dict().items()}
    buf.seek(0);restored=torch.load(buf,weights_only=True,map_location='cpu')
    model_b,opt_b,rng_b=build(restored)
    restored_indices,restored_loss=update(model_b,opt_b,rng_b)
    difference=max(float((expected[k]-v).abs().max()) for k,v in model_b.state_dict().items())
    assert torch.equal(continuous_indices,restored_indices)
    assert continuous_loss==restored_loss and difference==0
    assert sha(run/'checkpoint_last.pt')==original_hash
    result={'run':run.name,'checkpoint_unchanged':True,'sampler_matches':True,
            'loss_difference':abs(continuous_loss-restored_loss),'maximum_parameter_difference':difference,
            'optimizer_steps_after_disposable_updates':sorted({int(s['step']) for s in opt_b.state.values()}),
            'passed':True}
    out['runs'].append(result);print(json.dumps(result),flush=True)
out['passed']=len(out['runs'])==3 and all(r['passed'] for r in out['runs'])
assert out['passed']
(ROOT/'audit/resume_verification.json').write_text(json.dumps(out,indent=2)+'\n')

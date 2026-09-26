"""Verify final saved weights and reproduce validation, never open test data."""
import json
import math
import time
import torch
from gpt import GPT, GPTConfig
from train import NPLM
from protocol import ROOT, check_gate, load_data, windows, evaluate, sha

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
gate = check_gate()
rows, vocab, manifest = load_data('validation')
val_data = windows(rows, vocab)
output = {'manifest_sha256':gate['manifest_sha256'], 'test_evaluated':False,
          'method':'Fresh weights_only checkpoint load, hashes, optimizer step, and full validation reproduction',
          'runs':[]}
for run in sorted((ROOT/'runs').glob('*_seed_*')):
    if not run.is_dir(): continue
    cfg=json.loads((run/'config.json').read_text())
    status=json.loads((run/'status.json').read_text())
    assert status['state']=='completed' and status['step']==status['planned_steps']==2000
    assert not status['test_evaluated'] and not cfg['test_file_opened_by_trainer']
    assert cfg['manifest_sha256']==gate['manifest_sha256']
    assert cfg['audit_sha256']==sha(ROOT/'audit/double_check.json')
    for name, digest in cfg['source_code_sha256'].items():
        assert sha(ROOT/'src'/name)==digest, f'Source changed: {name}'
    best=torch.load(run/'checkpoint_best.pt',weights_only=True,map_location='cpu')
    last=torch.load(run/'checkpoint_last.pt',weights_only=True,map_location='cpu')
    assert best['manifest_sha256']==last['manifest_sha256']==gate['manifest_sha256']
    assert last['step']==last['planned_steps']==2000
    assert best['step']==last['best_step']==status['best_step']
    assert best['vocab']==last['vocab']==vocab
    assert best['config']==last['config']==cfg
    for ck in [best,last]:
        assert all(torch.isfinite(t).all().item() for t in ck['model'].values())
        assert all(t.dtype==torch.float32 for t in ck['model'].values() if t.is_floating_point())
    assert all(int(s['step'])==2000 for s in last['optimizer']['state'].values())
    assert len(last['sampler_rng'])>0 and len(last['torch_rng'])>0
    model=GPT(GPTConfig(**cfg['model_config'])) if cfg['model_type']=='gpt' else NPLM(**cfg['model_config'])
    model.load_state_dict(best['model'],strict=True)
    assert sum(p.numel() for p in model.parameters())==cfg['parameters']
    t=time.perf_counter()
    actual=evaluate(model,val_data,rows)
    stored=json.loads((run/'best_validation.json').read_text())
    assert [r['id'] for r in actual['per_record']]==[r['id'] for r in stored['per_record']]
    differences={k:abs(actual[k]-stored[k]) for k in ['nll_per_char','body_nll_per_char','ppl','body_ppl']}
    assert max(differences.values())<1e-6, differences
    events=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
    assert [e['step'] for e in events]==list(range(0,2001,200))
    selected=min(events,key=lambda e:e['validation']['body_nll_per_char'])
    assert selected['step']==best['step']
    assert abs(math.log(status['best_validation_body_ppl'])-actual['body_nll_per_char'])<1e-6
    item={'run':run.name,'completed_steps':last['step'],'best_step':best['step'],
          'best_checkpoint_sha256':sha(run/'checkpoint_best.pt'),
          'last_checkpoint_sha256':sha(run/'checkpoint_last.pt'),
          'maximum_reproduced_metric_difference':max(differences.values()),
          'validation_body_ppl':actual['body_ppl'],'parameters':cfg['parameters'],
          'optimizer_steps_match':True,'finite_FP32_weights':True,'source_hashes_match':True,
          'checkpoint_selection_matches_log':True,'validation_reloaded_seconds':time.perf_counter()-t,'passed':True}
    output['runs'].append(item)
    print(json.dumps(item),flush=True)
assert len(output['runs'])==3
output['passed']=all(r['passed'] for r in output['runs'])
(ROOT/'audit/run_integrity.json').write_text(json.dumps(output,indent=2)+'\n')
print('Saved audit/run_integrity.json')

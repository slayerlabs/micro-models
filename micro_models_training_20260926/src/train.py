"""Bounded, resumable FP32 CPU training on audited train/validation files only."""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from gpt import GPT,GPTConfig
from protocol import ROOT,check_gate,load_data,windows,evaluate,sha


class NPLM(nn.Module):
    def __init__(self,vocab_size,context=16,embedding=32,hidden=128):
        super().__init__();self.context=context
        self.embedding=nn.Embedding(vocab_size,embedding)
        self.fc=nn.Linear(context*embedding,hidden)
        self.head=nn.Linear(hidden,vocab_size)
    def forward(self,x,targets=None):
        contexts=F.pad(x,(self.context-1,0),value=0).unfold(1,self.context,1)
        h=self.embedding(contexts).flatten(-2)
        logits=self.head(torch.tanh(self.fc(h)))
        loss=None if targets is None else F.cross_entropy(logits.flatten(0,1),targets.flatten(),ignore_index=-100)
        return logits,loss


def atomic_save(obj,path):
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,temp);os.replace(temp,path)


def write_json(path,obj):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');os.replace(temp,path)


def compact(metrics):return {k:v for k,v in metrics.items() if k!='per_record'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['gpt','nplm'],default='gpt')
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--steps',type=int,default=2000)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.manual_seed(args.seed)
    gate=check_gate()
    train_rows,vocab,manifest=load_data('train')
    val_rows,_,_=load_data('validation')
    assert not {r['family_id'] for r in train_rows}&{r['family_id'] for r in val_rows}
    train_data=windows(train_rows,vocab,include_eos=True)
    val_data=windows(val_rows,vocab,include_eos=False)
    if args.model=='gpt':
        cfg=GPTConfig(vocab_size=vocab['size'],block_size=128,n_layer=4,n_head=4,n_embd=128,dropout=.1)
        model=GPT(cfg);model_config=dataclasses.asdict(cfg);lr=3e-4
    else:
        model_config={'vocab_size':vocab['size'],'context':16,'embedding':32,'hidden':128}
        model=NPLM(**model_config);lr=1e-3
    assert all(p.dtype==torch.float32 for p in model.parameters())
    opt=torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.99),weight_decay=.1)
    generator=torch.Generator().manual_seed(args.seed+1)
    run=ROOT/f"runs/{args.model}_{gate['manifest_sha256'][:8]}_seed_{args.seed}"
    if run.exists() and (run/'checkpoint_last.pt').exists() and not args.resume:
        raise RuntimeError('Existing run: use --resume rather than overwrite')
    run.mkdir(parents=True,exist_ok=True)
    config={'model_type':args.model,'model_config':model_config,'seed':args.seed,
      'steps':args.steps,'batch':32,'block':128,'stride':64,'base_lr':lr,'warmup':100,
      'precision':'FP32 parameters, forward, backward, optimizer; no AMP/FP8/quantization',
      'torch':str(torch.__version__),'python':platform.python_version(),'device':'cpu','threads':args.threads,
      'parameters':sum(p.numel() for p in model.parameters()),'source_commit':'da30da3d7486a4b6b3de8d2d4aaa0f183617f751',
      'manifest_sha256':gate['manifest_sha256'],'audit_sha256':sha(ROOT/'audit/double_check.json'),
      'source_code_sha256':{p.name:sha(p) for p in (ROOT/'src').glob('*.py')},
      'data_files_used':{s:manifest['data_files'][f'{s}.jsonl'] for s in ['train','validation']},
      'training_windows':len(train_data['x']),'training_target_positions_per_pass':int(train_data['y'].ne(-100).sum()),
      'validation_windows':len(val_data['x']),'test_file_opened_by_trainer':False,
      'checkpoint_selection':'minimum full-validation body_nll_per_char',
      'sampler':'uniform replacement sampling of fixed record-bounded chunks; every target has one labeled chunk',
      'loss':'all nonmasked training characters plus EOS; no target across record boundaries'}
    write_json(run/'config.json',config)
    first_step=0;best=float('inf');best_step=None;processed=0;elapsed_before=0.
    if args.resume:
        ck=torch.load(run/'checkpoint_last.pt',weights_only=True,map_location='cpu')
        assert ck['manifest_sha256']==gate['manifest_sha256']
        assert ck['model_config']==model_config and ck['planned_steps']==args.steps
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        generator.set_state(ck['sampler_rng']);torch.set_rng_state(ck['torch_rng'])
        first_step=ck['step'];best=ck['best_val_body_nll'];best_step=ck['best_step']
        processed=ck['processed_target_tokens'];elapsed_before=ck['elapsed_seconds']
    t0=time.perf_counter();recent=[]
    def checkpoint(step):
        return {'format_version':1,'model_type':args.model,'model_config':model_config,
          'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'planned_steps':args.steps,
          'best_val_body_nll':best,'best_step':best_step,'sampler_rng':generator.get_state(),
          'torch_rng':torch.get_rng_state(),'processed_target_tokens':processed,
          'elapsed_seconds':elapsed_before+time.perf_counter()-t0,'seed':args.seed,
          'manifest_sha256':gate['manifest_sha256'],'vocab':vocab,'config':config}
    def validate(step):
        nonlocal best,best_step
        metrics=evaluate(model,val_data,val_rows)
        improved=metrics['body_nll_per_char']<best
        if improved:
            best=metrics['body_nll_per_char'];best_step=step
            atomic_save({'format_version':1,'model_type':args.model,'model_config':model_config,
             'model':model.state_dict(),'step':step,'validation':metrics,'vocab':vocab,
             'manifest_sha256':gate['manifest_sha256'],'seed':args.seed,'config':config},run/'checkpoint_best.pt')
            write_json(run/'best_validation.json',metrics)
        event={'step':step,'elapsed_seconds':elapsed_before+time.perf_counter()-t0,
          'training_loss_recent':sum(recent)/len(recent) if recent else None,
          'processed_target_tokens':processed,'validation':compact(metrics),'best_step':best_step,'improved':improved}
        with (run/'metrics.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
        atomic_save(checkpoint(step),run/'checkpoint_last.pt')
        write_json(run/'status.json',{'state':'running','step':step,'planned_steps':args.steps,
          'best_step':best_step,'best_validation_body_ppl':math.exp(best),
          'elapsed_seconds':event['elapsed_seconds'],'test_evaluated':False})
        print(json.dumps({'model':args.model,'seed':args.seed,**event}),flush=True)
        model.train();recent.clear()
    if first_step==0:validate(0)
    model.train()
    for step in range(first_step+1,args.steps+1):
        # Warmup reaches base_lr at step 100; deterministic cosine schedule.
        if step<=100:current_lr=lr*step/100
        else:current_lr=lr*(.1+.45*(1+math.cos(math.pi*(step-100)/max(1,args.steps-100))))
        for group in opt.param_groups:group['lr']=current_lr
        ix=torch.randint(len(train_data['x']),(32,),generator=generator)
        x=train_data['x'][ix];y=train_data['y'][ix]
        _,loss=model(x,y)
        if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('Non-finite gradients')
        opt.step();recent.append(float(loss.detach()));processed+=int(y.ne(-100).sum())
        if step%200==0 or step==args.steps:validate(step)
        elif step%50==0:
            print(json.dumps({'model':args.model,'seed':args.seed,'step':step,
             'loss':sum(recent[-50:])/len(recent[-50:]),'elapsed_seconds':elapsed_before+time.perf_counter()-t0}),flush=True)
    ck=torch.load(run/'checkpoint_best.pt',weights_only=True,map_location='cpu')
    model.load_state_dict(ck['model']);model.eval()
    training_metrics=evaluate(model,windows(train_rows,vocab),train_rows)
    write_json(run/'best_training_metrics.json',training_metrics)
    write_json(run/'status.json',{'state':'completed','step':args.steps,'planned_steps':args.steps,
      'best_step':best_step,'best_validation_body_ppl':math.exp(best),
      'best_training_body_ppl':training_metrics['body_ppl'],
      'elapsed_seconds':elapsed_before+time.perf_counter()-t0,'processed_target_tokens':processed,
      'test_evaluated':False})
    print(json.dumps({'state':'completed','model':args.model,'seed':args.seed,
       'best_step':best_step,'validation_body_ppl':math.exp(best),
       'training_body_ppl':training_metrics['body_ppl'],'test_evaluated':False}),flush=True)


if __name__=='__main__':main()

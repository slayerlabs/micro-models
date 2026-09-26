"""Shared, record-aware training/evaluation protocol; no test loading in trainers."""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 128
STRIDE = 64


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(split):
    assert split in ('train', 'validation', 'test')
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    path=ROOT/f'data/{split}.jsonl'
    assert sha(path)==manifest['data_files'][path.name], 'Frozen dataset changed'
    vp=ROOT/'data/vocab.json'
    assert sha(vp)==manifest['vocab_sha256'], 'Frozen vocabulary changed'
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    assert all(r['split']==split for r in rows)
    return rows,json.loads(vp.read_text()),manifest


def check_gate():
    path=ROOT/'audit/double_check.json'
    result=json.loads(path.read_text())
    assert result['passed'], 'Independent audit did not pass'
    assert result['manifest_sha256']==sha(ROOT/'data/manifest.json'), 'Manifest changed since audit'
    assert result['protocol_sha256']==sha(Path(__file__)), 'Protocol changed since audit'
    return result


def encode(row,vocab):
    return [vocab['special_tokens']['BOS']] + [vocab['stoi'][c] for c in row['text']] + [vocab['special_tokens']['EOS']]


def windows(rows,vocab,include_eos=False,block=BLOCK,stride=STRIDE):
    """Every target appears once, with causal left context; windows never mix records.

    Training may include EOS. Evaluation scores every actual text character,
    including the first one conditioned on BOS; EOS is excluded from bpc/ppl.
    """
    xx=[]; yy=[]; body_masks=[]; owners=[]; spans=[]
    for owner,row in enumerate(rows):
        ids=encode(row,vocab)
        stop=len(ids) if include_eos else len(ids)-1
        header=len(row['text'])-len(row['body'])
        for p in range(1,stop,stride):
            end=min(p+stride,stop)
            start=max(0,end-1-block)
            x=ids[start:end-1]
            y=ids[start+1:end]
            positions=list(range(start+1,end))
            masks=[p<=j<end for j in positions]
            bmask=[m and header <= j-1 < len(row['text']) for m,j in zip(masks,positions)]
            y=[t if m else -100 for t,m in zip(y,masks)]
            real=len(x)
            xx.append(x+[0]*(block-real));yy.append(y+[-100]*(block-real))
            body_masks.append(bmask+[False]*(block-real))
            owners.append(owner);spans.append([start,p,end])
    return {'x':torch.tensor(xx,dtype=torch.long),'y':torch.tensor(yy,dtype=torch.long),
            'body':torch.tensor(body_masks,dtype=torch.bool),
            'owners':torch.tensor(owners,dtype=torch.long),'spans':spans,
            'record_ids':[r['id'] for r in rows]}


@torch.no_grad()
def evaluate(model, data, rows, batch=32):
    model.eval()
    nll=torch.zeros(len(rows),dtype=torch.float64)
    cnt=torch.zeros(len(rows),dtype=torch.float64)
    body_nll=nll.clone();body_cnt=cnt.clone()
    for start in range(0,len(data['x']),batch):
        x=data['x'][start:start+batch];y=data['y'][start:start+batch]
        logits,_=model(x)
        losses=F.cross_entropy(logits.transpose(1,2),y,reduction='none',ignore_index=-100).double()
        valid=y.ne(-100);bm=data['body'][start:start+batch]
        owner=data['owners'][start:start+batch]
        nll.index_add_(0,owner,(losses*valid).sum(1));cnt.index_add_(0,owner,valid.sum(1).double())
        body_nll.index_add_(0,owner,(losses*bm).sum(1));body_cnt.index_add_(0,owner,bm.sum(1).double())
    assert cnt.tolist()==[len(r['text']) for r in rows]
    assert body_cnt.tolist()==[len(r['body']) for r in rows]
    ce=nll.sum().item()/cnt.sum().item()
    bce=body_nll.sum().item()/body_cnt.sum().item()
    return {'nll_per_char':ce,'ppl':math.exp(ce),'bits_per_char':ce/math.log(2),
            'body_nll_per_char':bce,'body_ppl':math.exp(bce),
            'characters_scored':int(cnt.sum()),'body_characters_scored':int(body_cnt.sum()),
            'record_count':len(rows),'per_record':[
                {'id':r['id'],'family_id':r['family_id'],'nll':float(nll[i]),'chars':int(cnt[i]),
                 'body_nll':float(body_nll[i]),'body_chars':int(body_cnt[i])} for i,r in enumerate(rows)]}

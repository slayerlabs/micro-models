"""Second audit, recomputed from frozen files, without importing split builder.

Checks source provenance, cross-split overlap, alternative sequence matching,
ABC round-trip and exact target coverage. Does not calculate any model test loss.
"""
import argparse
import collections
import difflib
import hashlib
import itertools
import json
import math
import re
import unicodedata
from fractions import Fraction
from pathlib import Path

import torch
from music21 import converter

from protocol import ROOT, encode, evaluate, load_data, windows
import protocol


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def normalize(s):
    s=unicodedata.normalize('NFKD',s.lower().replace('ß','ss'))
    return re.sub('[^a-z0-9]','',s)


def musical_events(score):
    """Independent reconstruction from the serialized ABC, merging tied notes."""
    result=[]
    for obj in score.recurse().notesAndRests:
        if obj.isChord:raise ValueError('Unexpected chord')
        p=None if obj.isRest else int(obj.pitch.midi)
        duration=Fraction(obj.quarterLength).limit_denominator(4096)
        t=getattr(obj,'tie',None)
        if t and t.type in ('stop','continue'):
            if not result or result[-1][0]!=p:raise ValueError('Roundtrip broken tie')
            result[-1][1]+=duration
        else:result.append([p,duration])
    return [[p,d.numerator,d.denominator] for p,d in result]


def divergence(a,b):
    keys=set(a)|set(b);sa=sum(a.values());sb=sum(b.values())
    p=[a.get(k,0)/sa for k in keys];q=[b.get(k,0)/sb for k in keys]
    m=[(x+y)/2 for x,y in zip(p,q)]
    kl=lambda x:sum(v*math.log2(v/w) for v,w in zip(x,m) if v)
    return {'total_variation':sum(abs(x-y) for x,y in zip(p,q))/2,
            'jensen_shannon_bits':(kl(p)+kl(q))/2}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(2)
    report={'audit_version':1,'checks':{},'details':{},'test_loss_computed':False}
    def check(name,ok,detail=None):
        report['checks'][name]=bool(ok)
        if detail is not None:report['details'][name]=detail
    splits={};vocab=None
    for s in ('train','validation','test'):
        splits[s],vocab,manifest=load_data(s)
    rows=sum(splits.values(),[])
    ids=[r['id'] for r in rows]
    check('unique_representative_ids',len(set(ids))==len(ids))
    original=[];bad_hash=[];by_source={}
    for r in rows:
        for entry in [r]+r['aliases']:
            original.append(entry['id']);by_source[entry['id']]=r['split']
            p=args.source/'kern'/entry['source_file']
            if sha(p)!=entry['source_sha256']:bad_hash.append(entry['id'])
    expected={p.stem for p in (args.source/'kern').glob('chor*.krn')}
    check('all_370_sources_accounted_once',len(original)==370 and set(original)==expected and len(set(original))==370)
    check('all_source_hashes_match',not bad_hash,bad_hash)
    check('all_record_lengths_match',all(r['chars']==len(r['text']) for r in rows))
    check('all_text_hashes_match',all(hashlib.sha256(r['text'].encode()).hexdigest()==r['text_sha256'] for r in rows))
    check('no_unrecognized_characters',all(set(r['text'])<=set(vocab['stoi']) for r in rows))
    check('headers_do_not_contain_identity',all(r['text'].startswith('X:1\nM:') and r['id'] not in r['text'] and 'BWV' not in r['text'] for r in rows))
    check('soprano_is_highest_mean_voice',all(r['part_mean_midi'][r['soprano_part_index']]==max(r['part_mean_midi']) for r in rows))
    overlap={}
    for a,b in itertools.combinations(splits,2):
        k=f'{a}:{b}';x=splits[a];y=splits[b]
        detail={}
        for field in ['id','family_id','text_sha256','melody_sha256','transposed_sha256']:
            detail[field]=len({r[field] for r in x}&{r[field] for r in y})
        for what in ['title','work']:
            def keys(rs):
                out=set()
                for r in rs:
                    for e in [r]+r['aliases']:
                        source=(args.source/'kern'/e['source_file']).read_text()
                        if what=='title':
                            title=re.search(r'^!!!OTL@@DE:\s*(.+)$',source,re.M).group(1)
                            out.add(normalize(title))
                        else:out.update(re.findall(r'^!!!SCT:\s*BWV\s+(\d+)',source,re.M))
                return out
            detail[what]=len(keys(x)&keys(y))
        # Recreate sequences from events, rather than trusting stored intervals.
        def contours(rs):
            out=[]
            for r in rs:
                p=[e[0] for e in r['events'] if e[0] is not None]
                out.append([q-p for p,q in zip(p,p[1:])])
            return out
        cx,cy=contours(x),contours(y)
        def fragments(seq,size):return {tuple(z[i:i+size]) for z in seq for i in range(len(z)-size+1)}
        detail['shared_24_interval_fragments']=len(fragments(cx,24)&fragments(cy,24))
        sx={r['body'][i:i+128] for r in x for i in range(len(r['body'])-127)}
        sy={r['body'][i:i+128] for r in y for i in range(len(r['body'])-127)}
        detail['shared_128_character_fragments']=len(sx&sy)
        max_similarity=0;best_pair=None;near_count=0
        for i,j in itertools.product(range(len(x)),range(len(y))):
            if min(len(cx[i]),len(cy[j]))<15:continue
            sim=difflib.SequenceMatcher(None,cx[i],cy[j],autojunk=False).ratio()
            if sim>max_similarity:max_similarity=sim;best_pair=[x[i]['id'],y[j]['id']]
            if sim>=.80:near_count+=1
        detail['near_matches_alternative_algorithm']=near_count
        check(f'zero_overlap_{k}',all(v==0 for v in detail.values()),detail)
        overlap[k]={'maximum_alternative_contour_similarity':max_similarity,'nearest_pair':best_pair}
    report['details']['nearest_cross_split_pairs']=overlap
    distribution={}
    for field in ['mode','meter','key_sharps']:
        total=collections.Counter(str(r[field]) for r in rows)
        distribution[field]={}
        for split,rs in splits.items():
            c=collections.Counter(str(r[field]) for r in rs)
            distribution[field][split]={'counts':dict(c),**divergence(c,total)}
    total_chars=collections.Counter(''.join(r['text'] for r in rows))
    distribution['characters']={s:divergence(collections.Counter(''.join(r['text'] for r in rs)),total_chars) for s,rs in splits.items()}
    check('character_JS_under_0_06_bits',all(v['jensen_shannon_bits']<.06 for v in distribution['characters'].values()))
    check('mode_TV_under_0_15',all(v['total_variation']<.15 for v in distribution['mode'].values()))
    check('meter_TV_under_0_15',all(v['total_variation']<.15 for v in distribution['meter'].values()))
    total_n=len(rows);total_c=sum(r['chars'] for r in rows)
    shares={s:{'records':len(rs)/total_n,'chars':sum(r['chars'] for r in rs)/total_c} for s,rs in splits.items()}
    check('split_shares_within_5_percentage_points',all(abs(v-manifest['split_assignment']['target_ratios'][s])<=.05 for s,d in shares.items() for v in d.values()),shares)
    report['details']['distribution']=distribution
    # Every kept record is roundtripped, rather than only a favorable sample.
    failures=[]
    for i,r in enumerate(rows,1):
        try:
            score=converter.parseData(r['text'],format='abc')
            ev=musical_events(score)
            if ev!=r['events']:
                failures.append({'id':r['id'],'reason':'pitch/duration sequence differs','expected_events':len(r['events']),'actual_events':len(ev),
                                 'first_difference':next(([a,b] for a,b in zip(r['events'],ev) if a!=b),None)})
        except Exception as e:failures.append({'id':r['id'],'reason':repr(e)})
        if i%50==0:print(json.dumps({'roundtrip_checked':i,'failures':len(failures)}),flush=True)
    check('ABC_roundtrip_all_pitches_and_durations',not failures,failures)
    # Exact coverage checks for production windows, including every suffix.
    coverage_errors=[]
    for split,rs in splits.items():
        for eos in (False,True):
            data=windows(rs,vocab,include_eos=eos)
            seen=[collections.Counter() for _ in rs]
            for k,(start,p,end) in enumerate(data['spans']):
                owner=int(data['owners'][k]);seq=encode(rs[owner],vocab)
                length=end-1-start
                if data['x'][k,:length].tolist()!=seq[start:end-1]:coverage_errors.append('input span')
                for pos in range(length):
                    target=int(data['y'][k,pos]);global_pos=start+pos+1
                    if target!=-100:
                        if target!=seq[global_pos]:coverage_errors.append('target mismatch')
                        seen[owner][global_pos]+=1
            for i,r in enumerate(rs):
                expected_positions=set(range(1,len(r['text'])+1+int(eos)))
                if set(seen[i])!=expected_positions or any(v!=1 for v in seen[i].values()):coverage_errors.append(r['id'])
    check('all_targets_once_no_tail_omission_no_record_crossing',not coverage_errors,coverage_errors[:20])
    class Uniform:
        def eval(self):return self
        def __call__(self,x):return torch.zeros((*x.shape,vocab['size'])),None
    uniform_cases=[]
    for n in [1,2,63,64,65,127,128,129,130,191,192,193,255,256,257,1000]:
        r={'id':f'len{n}','family_id':'synthetic_control','text':'a'*n,'body':'a'*n}
        metrics=evaluate(Uniform(),windows([r],vocab),[r])
        uniform_cases.append({'chars':n,'scored':metrics['characters_scored'],'ppl':metrics['ppl']})
    check('uniform_metric_matches_analytic_value_all_lengths',all(x['chars']==x['scored'] and abs(x['ppl']-vocab['size'])<1e-4 for x in uniform_cases),uniform_cases)
    from gpt import GPT,GPTConfig
    torch.manual_seed(97)
    model=GPT(GPTConfig(vocab_size=vocab['size'],block_size=128)).eval()
    x=torch.randint(vocab['size'],(2,64));y=x.clone();y[:,40:]=(y[:,40:]+1)%vocab['size']
    with torch.no_grad():
        before,_=model(x);after,_=model(y)
    diff=float((before[:,:40]-after[:,:40]).abs().max())
    check('causal_mask_does_not_expose_future_tokens',diff==0.0,{'max_logit_difference_before_changed_future':diff})
    report['manifest_sha256']=sha(ROOT/'data/manifest.json')
    report['protocol_sha256']=sha(Path(protocol.__file__))
    report['audit_script_sha256']=sha(__file__)
    report['passed']=all(report['checks'].values())
    report['limitations']=['Finite similarity rules cannot prove the absence of every musicological relationship.',
      'Short shared motifs are expected in this domain and are not treated as full-record leakage.',
      'Distribution matching uses metadata only; test prediction quality remains unmeasured.',
      'Rare modes and keys may not occur in every split; inspect count tables, not only aggregate divergence.']
    (ROOT/'audit/double_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'checks':report['checks'],'failed':[k for k,v in report['checks'].items() if not v]},indent=2),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()

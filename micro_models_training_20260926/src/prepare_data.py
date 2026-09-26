"""Pinned real Bach corpus -> grouped, stratified, frozen train/val/test.

No split is selected with model scores. All derived versions of a melody stay
in one component. The alphabet is a declared ABC grammar, not fitted on test.
"""
import argparse
import collections
import hashlib
import html
import itertools
import json
import math
import re
import subprocess
import unicodedata
from fractions import Fraction
from pathlib import Path

import numpy as np
from music21 import converter
from rapidfuzz.fuzz import ratio

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = '0fd9e00542445a522c6030c80c687b874aa569d5'
SOURCE_URL = 'https://github.com/craigsapp/bach-370-chorales'
SEED = 20260926
NEAR_THRESHOLD = 80.0
CHAR_WINDOW = 128
NOTE_WINDOW = 24
SPLITS = ('train', 'validation', 'test')
RATIOS = np.array([0.8, 0.1, 0.1])
ABC_ALPHABET = sorted(set("\n ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/^_=,'|#[]-:.()"))


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def norm_title(s):
    s = unicodedata.normalize('NFKD', html.unescape(s).casefold().replace('ß', 'ss'))
    return ''.join(c for c in s if c.isalnum() and not unicodedata.combining(c))


def duration(quarter_length):
    # L:1/4; exact rational representation. No quantization of durations.
    q = Fraction(quarter_length).limit_denominator(4096)
    if q == 1:
        return ''
    return str(q.numerator) if q.denominator == 1 else f'{q.numerator}/{q.denominator}'


def pitch(p):
    alter = int(p.accidental.alter) if p.accidental is not None else 0
    if alter not in (-2, -1, 0, 1, 2):
        raise ValueError('Unsupported accidental')
    acc = {-2: '__', -1: '_', 0: '=', 1: '^', 2: '^^'}[alter]
    name = p.step.lower() + "'" * (p.octave - 5) if p.octave >= 5 else p.step.upper() + ',' * (4 - p.octave)
    return acc + name


def merge_events(part):
    events = []
    for n in part.recurse().notesAndRests:
        if n.isChord:
            raise ValueError('Soprano contains chord')
        q = Fraction(n.quarterLength).limit_denominator(4096)
        if q <= 0:
            raise ValueError('Zero/negative duration')
        p = None if n.isRest else int(n.pitch.midi)
        tie = getattr(n, 'tie', None)
        if tie and tie.type in ('continue', 'stop'):
            if not events or events[-1][0] != p:
                raise ValueError('Broken tie')
            events[-1][1] += q
        else:
            events.append([p, q])
    return [[p, q.numerator, q.denominator] for p, q in events]


def parse_record(path):
    raw = path.read_bytes()
    source = raw.decode('utf-8')
    score = converter.parse(str(path), forceSource=True)
    means = []
    for part in score.parts:
        notes = [n for n in part.recurse().notes if n.isNote]
        means.append(sum(n.pitch.midi for n in notes) / len(notes) if notes else -1)
    if len(means) != 4:
        raise ValueError(f'Expected four voices, got {len(means)}')
    soprano_index = int(np.argmax(means))
    sop = score.parts[soprano_index]
    events = merge_events(sop)
    notes = [p for p, _, _ in events if p is not None]
    if len(notes) < 8:
        raise ValueError('Too few notes')
    meters = list(sop.recurse().getElementsByClass('TimeSignature'))
    meter = meters[0].ratioString if meters else '4/4'
    # music21 omits Key objects for modal Humdrum labels. Read the explicit
    # source label instead of silently dropping Dorian/Mixolydian/Phrygian music.
    key_token = next((line.split('\t')[-1] for line in source.splitlines()
                      if re.fullmatch(r'\*[A-Ga-g][#-]*:[A-Za-z]*',line.split('\t')[-1])),None)
    if key_token is None:raise ValueError('Missing explicit source key label')
    key_match=re.fullmatch(r'\*([A-Ga-g])([#-]*):([A-Za-z]*)',key_token)
    tonic,acc,modal=key_match.groups()
    mode={'dor':'dorian','mix':'mixolydian','phr':'phrygian'}.get(modal,
         'minor' if tonic.islower() else 'major')
    suffix={'major':'','minor':'m','dorian':'dor','mixolydian':'mix','phrygian':'phr'}[mode]
    key_name=tonic.upper()+acc.replace('-','b')+suffix
    signatures=list(sop.recurse().getElementsByClass('KeySignature'))
    key_sharps=signatures[0].sharps if signatures else 0
    bars = []
    active_meter = meter
    for measure in sop.getElementsByClass('Measure'):
        tokens = []
        ts = list(measure.getElementsByClass('TimeSignature'))
        if ts and ts[0].ratioString != active_meter:
            active_meter = ts[0].ratioString
            tokens.append(f'[M:{active_meter}]')
        for n in measure.recurse().notesAndRests:
            t = ('z' if n.isRest else pitch(n.pitch)) + duration(n.quarterLength)
            if getattr(n, 'tie', None) and n.tie.type in ('start', 'continue'):
                t += '-'
            tokens.append(t)
        bars.append(' '.join(tokens))
    if not bars:
        raise ValueError('No measures')
    body = ' | '.join(bars) + ' |\n'
    text = f'X:1\nM:{meter}\nL:1/4\nK:{key_name}\n' + body
    unknown = set(text) - set(ABC_ALPHABET)
    if unknown:
        raise ValueError(f'Outside predefined alphabet: {unknown}')
    title_match = re.search(r'^!!!OTL@@DE:\s*(.+)$', source, re.M)
    title = html.unescape(title_match.group(1)) if title_match else ''
    if not title:
        raise ValueError('Missing title')
    bwv = sorted(set(re.findall(r'^!!!SCT:\s*BWV\s+(\d+)', source, re.M)))
    # Transposition-invariant pitch contour, tied note fragments merged.
    intervals = [b - a for a, b in zip(notes, notes[1:])]
    first = notes[0]
    normalized = [[None if p is None else p-first, a, b] for p, a, b in events]
    return {'id': path.stem, 'source_file': path.name, 'source_sha256': digest(raw),
            'title': title, 'normalized_title': norm_title(title), 'work_roots': bwv,
            'soprano_part_index': soprano_index, 'part_mean_midi': means,
            'meter': meter, 'all_meters': sorted(set(t.ratioString for t in meters)),
            'mode': mode, 'key': key_name, 'key_sharps': key_sharps,
            'notes': len(notes), 'bars': len(bars), 'chars': len(text),
            'events': events, 'intervals': intervals, 'normalized_events': normalized,
            'text': text, 'body': body, 'text_sha256': digest(text.encode()),
            'melody_sha256': digest(events), 'transposed_sha256': digest(normalized)}


class UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def root(self, x):
        while x != self.p[x]:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        if a == b: return False
        self.p[max(a,b)] = min(a,b)
        return True


def make_components(records):
    uf = UnionFind(len(records)); edges = []
    def join(a, b, why):
        if uf.join(a,b): edges.append({'a':records[a]['id'], 'b':records[b]['id'], 'reason':why})
    for field in ['normalized_title', 'melody_sha256', 'transposed_sha256', 'text_sha256']:
        seen = {}
        for i,r in enumerate(records):
            v=r[field]
            if v in seen: join(i,seen[v],field)
            else: seen[v]=i
    seen_work={}
    for i,r in enumerate(records):
        for work in r['work_roots']:
            if work in seen_work:join(i,seen_work[work],'work_root')
            else:seen_work[work]=i
    # All matching long character windows and musical-contour fragments are
    # grouped before splitting; short ubiquitous musical motifs may still recur.
    for key,size in [('body',CHAR_WINDOW),('intervals',NOTE_WINDOW)]:
        seen={}
        for i,r in enumerate(records):
            seq=r[key]
            for start in range(len(seq)-size+1):
                token=seq[start:start+size]
                if isinstance(token,list):token=tuple(token)
                if token in seen:join(i,seen[token],f'{key}_window_{size}')
                else:seen[token]=i
    near=[]
    for i,j in itertools.combinations(range(len(records)),2):
        a,b=records[i]['intervals'],records[j]['intervals']
        if min(len(a),len(b)) < 15:continue
        sim=ratio(a,b)
        if sim >= NEAR_THRESHOLD:
            join(i,j,'near_contour_80pct')
            near.append({'a':records[i]['id'],'b':records[j]['id'],'similarity':sim})
    groups=collections.defaultdict(list)
    for i,r in enumerate(records):groups[uf.root(i)].append(r['id'])
    for i,r in enumerate(records):
        members=sorted(groups[uf.root(i)])
        r['family_id']='family_'+digest(members)[:16]
    return edges,near


def assign(records):
    groups=sorted(set(r['family_id'] for r in records)); gi={g:i for i,g in enumerate(groups)}
    meter_counts=collections.Counter(r['meter'] for r in records)
    length_cuts=np.quantile([r['chars'] for r in records],[.25,.5,.75])
    for r in records:
        r['length_bin']=int(np.searchsorted(length_cuts,r['chars']))
        r['meter_stratum']=r['meter'] if meter_counts[r['meter']]>=20 else 'other'
    columns=['rows','chars']
    for field in ['mode','meter_stratum','length_bin','key_sharps']:
        columns.extend(f'{field}:{v}' for v in sorted(set(r[field] for r in records),key=str))
    mat=np.zeros((len(groups),len(columns)))
    for r in records:
        a=mat[gi[r['family_id']]];a[0]+=1;a[1]+=r['chars']
        for c,column in enumerate(columns[2:],2):
            field,v=column.split(':');a[c]+=str(r[field])==v
    totals=mat.sum(axis=0);target=RATIOS[:,None]*totals[None,:]
    scale=np.maximum(target,2.)
    weights=np.array([8.,8.]+[2. if c.startswith(('mode:','meter_stratum:')) else 1. for c in columns[2:]])
    rng=np.random.default_rng(SEED);n=len(groups)
    nval=max(1,round(n*.1));ntest=nval
    best=None
    # This search uses only population metadata, never model accuracy/loss.
    for _ in range(12000):
        perm=rng.permutation(n)
        parts=[perm[nval+ntest:],perm[:nval],perm[nval:nval+ntest]]
        counts=np.stack([mat[p].sum(axis=0) for p in parts])
        score=float((((counts-target)/scale)**2*weights).mean())
        if best is None or score<best[0]:best=(score,parts,counts)
    # Refine metadata balance by pairwise group swaps. The number of groups per
    # split is fixed and no prediction result is ever used by this objective.
    labels=np.zeros(n,dtype=int)
    for k,ix in enumerate(best[1]):labels[ix]=k
    score,counts=best[0],best[2].copy()
    for _ in range(30000):
        i,j=rng.integers(n,size=2);a,b=labels[i],labels[j]
        if a==b:continue
        candidate=counts.copy();d=mat[j]-mat[i];candidate[a]+=d;candidate[b]-=d
        s=float((((candidate-target)/scale)**2*weights).mean())
        if s<score:
            score=s;counts=candidate;labels[i],labels[j]=b,a
    best=(score,[np.flatnonzero(labels==k) for k in range(3)],counts)
    for split,idxs in zip(SPLITS,best[1]):
        members={groups[i] for i in idxs}
        for r in records:
            if r['family_id'] in members:r['split']=split
    return {'seed':SEED,'target_ratios':dict(zip(SPLITS,RATIOS.tolist())),
            'selection':'metadata only: 12000 deterministic candidates plus 30000 group-swap proposals; no model scores',
            'objective':best[0],'feature_columns':columns,'counts':best[2].tolist(),
            'length_quartile_cuts':length_cuts.tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    args=ap.parse_args();source=args.source.resolve()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert head==SOURCE_COMMIT,(head,SOURCE_COMMIT)
    files=sorted((source/'kern').glob('chor*.krn'));assert len(files)==370
    data=ROOT/'data';audit=ROOT/'audit';data.mkdir(exist_ok=True);audit.mkdir(exist_ok=True)
    cache=data/'all_records_pre_dedup.jsonl'
    if cache.exists():
        records=[json.loads(s) for s in cache.read_text().splitlines()]
        assert len(records)==370
        for r in records:assert digest((source/'kern'/r['source_file']).read_bytes())==r['source_sha256']
    else:
        records=[];errors=[]
        for k,p in enumerate(files,1):
            try:records.append(parse_record(p))
            except Exception as e:errors.append({'file':p.name,'error':repr(e)})
            if k%50==0:print(json.dumps({'parsed':k,'accepted':len(records),'errors':len(errors)}),flush=True)
        write_json(audit/'parse_errors.json',errors)
        if errors:raise RuntimeError('Review parsing failures before splitting')
        cache.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
    edges,near=make_components(records)
    # Identical soprano targets have one representative; provenance of every
    # original score is retained, including alternate titles/work identifiers.
    unique={};duplicates=[]
    for r in records:
        k=r['melody_sha256']
        if k in unique:
            base=unique[k]
            assert base['family_id']==r['family_id']
            base['aliases'].append({z:r[z] for z in ('id','title','normalized_title','work_roots','source_file','source_sha256')})
            duplicates.append({'removed':r['id'],'representative':base['id']})
        else:
            r['aliases']=[];unique[k]=r
    kept=sorted(unique.values(),key=lambda r:r['id'])
    assignment=assign(kept)
    for split in SPLITS:
        rows=[r for r in kept if r['split']==split]
        (data/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    vocab={'special_tokens':{'PAD':0,'BOS':1,'EOS':2},'chars':ABC_ALPHABET,
           'stoi':{c:i+3 for i,c in enumerate(ABC_ALPHABET)},
           'size':len(ABC_ALPHABET)+3,'policy':'ABC grammar declared before data inspection; no fitted tokenizer'}
    write_json(data/'vocab.json',vocab)
    profile={}
    for split in SPLITS:
        rows=[r for r in kept if r['split']==split]
        profile[split]={'records':len(rows),'families':len(set(r['family_id'] for r in rows)),
          'characters':sum(r['chars'] for r in rows),'notes':sum(r['notes'] for r in rows),
          'length_quantiles':np.quantile([r['chars'] for r in rows],[0,.25,.5,.75,1]).tolist(),
          'mode':dict(collections.Counter(r['mode'] for r in rows)),
          'meter':dict(collections.Counter(r['meter'] for r in rows)),
          'key':dict(collections.Counter(r['key'] for r in rows)),
          'character_counts':dict(collections.Counter(''.join(r['text'] for r in rows)))}
    source_manifest=[{'id':r['id'],'file':r['source_file'],'sha256':r['source_sha256'],'family_id':r['family_id']} for r in records]
    manifest={'version':1,'source_url':SOURCE_URL,'source_commit':head,'source_records':len(records),
      'license':'CC BY-NC-SA 4.0','creator':'Craig Stuart Sapp, digital edition (2009)',
      'use':'Educational, noncommercial training; attribution and share-alike for adapted corpus retained',
      'unit':'One distinct soprano realization; family is the independent split unit',
      'near_duplicate_threshold_percent':NEAR_THRESHOLD,'char_overlap_window':CHAR_WINDOW,'interval_overlap_window':NOTE_WINDOW,
      'exact_melody_duplicates_removed':len(duplicates),'kept_records':len(kept),
      'families':len(set(r['family_id'] for r in kept)),
      'split_assignment':assignment,'profile':profile,'source_files':source_manifest,
      'data_files':{f'{s}.jsonl':digest((data/f'{s}.jsonl').read_bytes()) for s in SPLITS},
      'vocab_sha256':digest((data/'vocab.json').read_bytes()),
      'test_policy':'No test loss, generation ranking, checkpoint selection or hyperparameter tuning before final protocol freeze',
      'representation':'Soprano, exact durations, explicit accidentals, ties retained, no title/BWV/source ID in model input; repeats not expanded'}
    write_json(data/'manifest.json',manifest)
    write_json(audit/'group_edges.json',edges);write_json(audit/'near_duplicates.json',near)
    write_json(audit/'removed_duplicates.json',duplicates)
    summary={'source_records':len(records),'duplicates_removed':len(duplicates),'kept':len(kept),'families':manifest['families'],
             'largest_family':max(collections.Counter(r['family_id'] for r in kept).values()),
             'splits':{s:{k:v for k,v in profile[s].items() if k not in ('character_counts','key')} for s in SPLITS}}
    write_json(audit/'profile_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()

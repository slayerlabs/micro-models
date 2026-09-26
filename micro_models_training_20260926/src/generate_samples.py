"""Fixed raw samples from recovered models; no test prompts and no selection of favorable outputs."""
import hashlib
import copy
import json
import math
import warnings
from pathlib import Path
import torch
from music21 import converter,midi,tempo,stream,meter
from gpt import GPT,GPTConfig
from train import NPLM
from protocol import ROOT,check_gate,sha

torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
gate=check_gate()
PROMPT='X:1\nM:4/4\nL:1/4\nK:G\n'
SEEDS=[2026092601,2026092602]
MAX_NEW=384;TEMPERATURE=.8;TOP_K=20
dest=ROOT/'samples';dest.mkdir(exist_ok=True)

@torch.no_grad()
def sample(model,vocab,seed):
    generator=torch.Generator().manual_seed(seed)
    ids=[vocab['special_tokens']['BOS']]+[vocab['stoi'][c] for c in PROMPT]
    generated=[];inverse={v:k for k,v in vocab['stoi'].items()};reason='token_limit'
    for _ in range(MAX_NEW):
        x=torch.tensor([ids[-128:]],dtype=torch.long)
        logits,_=model(x)
        scores=logits[0,-1].clone()/TEMPERATURE
        scores[vocab['special_tokens']['PAD']]=-torch.inf
        scores[vocab['special_tokens']['BOS']]=-torch.inf
        threshold=torch.topk(scores,TOP_K).values[-1]
        scores[scores<threshold]=-torch.inf
        token=int(torch.multinomial(torch.softmax(scores,dim=-1),1,generator=generator))
        ids.append(token);generated.append(token)
        if token==vocab['special_tokens']['EOS']:
            reason='EOS';break
    text=PROMPT+''.join(inverse[t] for t in generated if t in inverse)
    return text,generated,reason

report={'manifest_sha256':gate['manifest_sha256'],'test_evaluated':False,
        'protocol':{'prompt':PROMPT,'seeds':SEEDS,'max_new_tokens':MAX_NEW,
                    'temperature':TEMPERATURE,'top_k':TOP_K,'masked_tokens':['PAD','BOS'],
                    'selection':'All two predeclared seeds for every neural model; no retries, no editing, no grammar repair'},
        'samples':[]}
for run in sorted((ROOT/'runs').glob('*_seed_*')):
    if not run.is_dir():continue
    ck=torch.load(run/'checkpoint_best.pt',weights_only=True,map_location='cpu')
    assert ck['manifest_sha256']==gate['manifest_sha256']
    model=GPT(GPTConfig(**ck['model_config'])) if ck['model_type']=='gpt' else NPLM(**ck['model_config'])
    model.load_state_dict(ck['model']);model.eval()
    for index,seed in enumerate(SEEDS):
        text,ids,reason=sample(model,ck['vocab'],seed)
        if index==0:
            repeated,repeated_ids,_=sample(model,ck['vocab'],seed)
            assert text==repeated and ids==repeated_ids
        name=f'{run.name}_sample_{index+1}'
        abc_path=dest/f'{name}.abc';abc_path.write_text(text)
        item={'model_run':run.name,'checkpoint_step':ck['step'],'checkpoint_sha256':sha(run/'checkpoint_best.pt'),
              'sample_seed':seed,'abc':str(abc_path.relative_to(ROOT)),'text_sha256':sha(abc_path),
              'generated_tokens':len(ids),'stop_reason':reason,'deterministic_repeat_checked':index==0,
              'parser_accepted':False,'midi_written':False}
        try:
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter('always')
                score=converter.parseData(text,format='abc')
                notes=list(score.recurse().notes)
                if not notes:raise ValueError('No notes decoded from raw sample')
                item.update({'parser_accepted':True,'notes':len(notes)})
                measures=list(score.recurse().getElementsByClass('Measure'))
                interior=measures[1:-1]
                bad=[]
                for i,m in enumerate(interior,1):
                    actual=float(m.duration.quarterLength);expected=float(m.barDuration.quarterLength)
                    if not math.isclose(actual,expected,abs_tol=1e-8):bad.append({'index':i,'beats':actual,'expected':expected})
                item.update({'measures':len(measures),'interior_measures_checked':len(interior),
                             'incorrect_interior_measures':bad})
                notation_path=dest/f'{name}_notation.mid'
                try:
                    notation=copy.deepcopy(score)
                    notation.insert(0,tempo.MetronomeMark(number=90))
                    notation.write('midi',fp=str(notation_path))
                    item['notation_midi_export']={'succeeded':True,'file':str(notation_path.relative_to(ROOT))}
                except Exception as notation_error:
                    item['notation_midi_export']={'succeeded':False,'error_type':type(notation_error).__name__}
                    notation_path.unlink(missing_ok=True)
                # The common MIDI preview preserves the parsed note/rest sequence,
                # but not invalid ABC bar boundaries. Raw ABC is never repaired.
                linear=stream.Part()
                linear.append(tempo.MetronomeMark(number=90))
                linear.append(meter.TimeSignature('4/4'))
                for event in score.recurse().notesAndRests:
                    linear.append(copy.deepcopy(event))
                path=dest/f'{name}.mid'
                linear.write('midi',fp=str(path))
                f=midi.MidiFile();f.open(str(path));f.read();f.close()
                assert path.read_bytes().startswith(b'MThd') and f.tracks
                item.update({'midi_written':True,'midi_reopened':True,'midi':str(path.relative_to(ROOT)),
                             'midi_policy':'Sequential note/rest rendering; ABC barline positions are not retained',
                             'midi_sha256':sha(path),'warnings':[str(w.message) for w in seen]})
        except Exception as e:
            item['error']=f'{type(e).__name__}: {e}'
        report['samples'].append(item)
        print(json.dumps({k:v for k,v in item.items() if k not in ['warnings','incorrect_interior_measures']},ensure_ascii=False),flush=True)
report['summary']={'total':len(report['samples']),'parser_accepted':sum(s['parser_accepted'] for s in report['samples']),
                   'midi_reopened':sum(s.get('midi_reopened',False) for s in report['samples']),
                   'EOS':sum(s['stop_reason']=='EOS' for s in report['samples']),
                   'samples_with_incorrect_interior_measures':sum(bool(s.get('incorrect_interior_measures')) for s in report['samples']),
                   'original_notation_midi_exports':sum(s.get('notation_midi_export',{}).get('succeeded',False) for s in report['samples'])}
report['limitations']=['Six fixed diagnostic samples are not a population-level music quality evaluation.',
                      'music21 can accept imperfect ABC; parsing or valid MIDI does not prove correct meter or musicality.',
                      'Primary MIDI is a sequential rendering of parsed pitches/durations at 90 BPM; original ABC bar positions are not retained.',
                      'A _notation.mid file is included only when the original parsed notation can be exported directly.',
                      'Token-limit endings are not presented as model-chosen complete pieces.']
(dest/'generation_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report['summary']),flush=True)

"""Interpolated character n-gram baseline on the SAME frozen training records."""
import collections
import json
import math
import pickle

from protocol import ROOT,check_gate,encode,load_data


def main():
    gate=check_gate()
    train,vocab,manifest=load_data('train');val,_,_=load_data('validation')
    counts=[collections.defaultdict(collections.Counter) for _ in range(7)]
    for row in train:
        seq=encode(row,vocab)
        # Reset at every record; BOS is context, EOS is a training target.
        for j in range(1,len(seq)):
            for order in range(min(6,j)+1):
                ctx=tuple(seq[j-order:j]) if order else ()
                counts[order][ctx][seq[j]]+=1
    uni=counts[0][()];total=sum(uni.values());size=vocab['size']
    totals=[{ctx:sum(c.values()) for ctx,c in layer.items()} for layer in counts]
    def prob(history,target,order):
        out=(uni[target]+.01)/(total+.01*size);weight=1.
        for n in range(1,min(order,len(history))+1):
            ctx=tuple(history[-n:]);c=counts[n].get(ctx)
            if c:
                w=2.**n;out+=w*c[target]/totals[n][ctx];weight+=w
        return out/weight
    # Normalization and unseen-symbol checks before recording results.
    for h in [[],[1],[999],[1,2,3],encode(train[0],vocab)[4:10]]:
        for n in (1,3,6):
            assert abs(sum(prob(h,t,n) for t in range(size))-1)<1e-10
            assert all(prob(h,t,n)>0 for t in range(size))
    rows=[]
    for order in (1,3,6):
        result={'order':order,'per_record':[]};nll=0.;nchar=0;bnll=0.;bchar=0
        for r in val:
            seq=encode(r,vocab);loss=0.;body_loss=0.;header=len(r['text'])-len(r['body'])
            for j in range(1,len(seq)-1):
                term=-math.log(prob(seq[max(0,j-6):j],seq[j],order));loss+=term
                if j-1>=header:body_loss+=term
            nll+=loss;nchar+=len(r['text']);bnll+=body_loss;bchar+=len(r['body'])
            result['per_record'].append({'id':r['id'],'family_id':r['family_id'],'nll':loss,'chars':len(r['text']),
                                        'body_nll':body_loss,'body_chars':len(r['body'])})
        result.update({'nll_per_char':nll/nchar,'ppl':math.exp(nll/nchar),'bits_per_char':nll/nchar/math.log(2),
                       'body_nll_per_char':bnll/bchar,'body_ppl':math.exp(bnll/bchar),
                       'characters_scored':nchar,'body_characters_scored':bchar,
                       'contexts':sum(len(c) for c in counts[:order+1])})
        rows.append(result)
    run=ROOT/f"runs/ngram_{gate['manifest_sha256'][:8]}";run.mkdir(exist_ok=True)
    artifact={'manifest_sha256':gate['manifest_sha256'],'vocab':vocab,'orders':[1,3,6],
              'training_records':len(train),'training_characters':sum(len(r['text']) for r in train),
              'validation_records':len(val),'test_evaluated':False,'normalization_test_passed':True,'results':rows}
    (run/'validation.json').write_text(json.dumps(artifact,indent=2,ensure_ascii=False)+'\n')
    with (run/'counts.pkl').open('wb') as f:pickle.dump([dict(c) for c in counts],f,protocol=5)
    print(json.dumps({**{k:v for k,v in artifact.items() if k not in ('vocab','results')},
                      'results':[{k:v for k,v in r.items() if k!='per_record'} for r in rows]},indent=2))


if __name__=='__main__':main()

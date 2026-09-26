"""Standard-library integrity check; works before installing PyTorch or music21."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

checked=[]
for line in (ROOT/'SHA256SUMS').read_text().splitlines():
    digest,name=line.split('  ',1)
    path=(ROOT/name).resolve()
    assert path.is_relative_to(ROOT), 'Path outside package'
    assert path.is_file(), f'Missing: {name}'
    assert sha(path)==digest, f'Hash mismatch: {name}'
    checked.append(name)
assert len(checked)==len(set(checked))
manifest_hash=sha(ROOT/'data/manifest.json')
audit=json.loads((ROOT/'audit/double_check.json').read_text())
assert audit['passed'] and audit['manifest_sha256']==manifest_hash
runs=[]
for path in sorted((ROOT/'runs').glob('*/config.json')):
    config=json.loads(path.read_text())
    assert config['manifest_sha256']==manifest_hash
    for name,digest in config['source_code_sha256'].items():
        assert sha(ROOT/'src'/name)==digest, f'Training source changed: {name}'
    status=json.loads((path.parent/'status.json').read_text())
    runs.append({'name':path.parent.name,'state':status['state'],'steps':status['step'],
                 'planned_steps':status['planned_steps'],'test_evaluated':status['test_evaluated']})
print(json.dumps({'passed':True,'files_checked':len(checked),'manifest_sha256':manifest_hash,'runs':runs},indent=2))

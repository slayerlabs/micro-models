"""Execute sequentially without kernel sockets, retaining genuine IPython outputs."""
import os,json
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
P=Path(__file__).resolve().parent
os.chdir(P);os.environ['MPLBACKEND']='Agg'
path=P/'Micro_models_ewaluacja_PL.ipynb';n=nbformat.read(path,as_version=4)
shell=InteractiveShell.instance();count=0
for cell in n.cells:
    if cell.cell_type!='code':continue
    with capture_output(stdout=True,stderr=True,display=True) as cap:
        result=shell.run_cell(cell.source,store_history=True)
    result.raise_error();count+=1
    cell.execution_count=result.execution_count;cell.outputs=[]
    if cap.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=cap.stdout))
    if cap.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=cap.stderr))
    for out in cap.outputs:cell.outputs.append(nbformat.v4.new_output('display_data',data=out.data,metadata=out.metadata))
n.metadata['execution']={'engine':'IPython InteractiveShell, fresh process, sequential cells','reason':'Jupyter TCP kernel startup blocked by runtime network-socket permissions; nbclient failed before first cell.'}
nbformat.validate(n);nbformat.write(n,path)
proof={'format_valid':True,'code_cells_executed':count,'errors':0,'figure_outputs':sum('image/png' in o.get('data',{}) for c in n.cells for o in c.get('outputs',[])),'engine':n.metadata['execution']}
(P/'notebook_execution.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))

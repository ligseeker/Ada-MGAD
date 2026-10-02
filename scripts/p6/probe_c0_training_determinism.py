import json, sys
from pathlib import Path
import numpy as np
import torch
from adabelief_pytorch import AdaBelief
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from src.e2e.onset_trigger import OnsetDevelopmentState
from src.e2e.protocol import load_config
from src.e2e.system_trigger_data import build_trigger_loader
from src.e2e.system_trigger_tcn import WindowCausalTCNTrigger
from scripts.p6.run_c0_trigger import move_to_device
from util.util import seed_everything
config=json.loads((root/'configs/e2e/gaia_p6_c0_training_budget_v1.json').read_text())
old=Path(config['budget']['old_runs']['17'])
args=json.loads((old/'validation_selection.json').read_text())['model_args']
state=OnsetDevelopmentState(load_config(root/config['base_config']),Path(config['budget']['data_root']),Path(config['budget']['registry']))
dataset=state.build_dataset('fit')
torch.set_num_threads(4)
if "--deterministic" in sys.argv:
 torch.backends.cudnn.deterministic=True
 torch.use_deterministic_algorithms(True)
# Cache eight real input batches, outside either seeded training trajectory.
loader=build_trigger_loader(dataset,batch_size=32,num_workers=0)
batches=[]
for b in loader:
 batches.append(move_to_device(b,torch.device('cuda')))
 if len(batches)==8:break
graph=np.load(state.data_root/'graph.npy',allow_pickle=False)
def trajectory():
 seed_everything(17);torch.manual_seed(17)
 model=WindowCausalTCNTrigger(graph,**args).cuda().train()
 init={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 opt=AdaBelief(model.parameters(),lr=.001,weight_decay=.0005,print_change_log=False)
 criterion=torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor([31841/7027],device='cuda'))
 losses=[];outputs=[]
 for b in batches:
  logits,reg=model(b);labels=b['trigger_label'];mask=(labels!=2).float();target=(labels==1).float()
  loss=(criterion(logits,target)*mask).sum()/mask.sum()+reg
  outputs.append(logits.detach().cpu().clone());losses.append(float(loss))
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.,norm_type=2);opt.step()
 return init,outputs,losses,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
a=trajectory();b=trajectory()
err=lambda x,y:max(float((x[k]-y[k]).abs().max()) for k in x)
r={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'initial_weights_max_abs_error':err(a[0],b[0]),'step_logits_max_abs_error':[float((x-y).abs().max()) for x,y in zip(a[1],b[1])],'loss_absolute_errors':[abs(x-y) for x,y in zip(a[2],b[2])],'final_weights_max_abs_error':err(a[3],b[3])}
print(json.dumps(r,indent=2))
if "--output" in sys.argv:
 target=Path(sys.argv[sys.argv.index("--output")+1])
 if target.exists():raise FileExistsError(target)
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_text(json.dumps(r,indent=2)+"\n")
if r['final_weights_max_abs_error'] or any(r['step_logits_max_abs_error']):
 raise SystemExit(2)

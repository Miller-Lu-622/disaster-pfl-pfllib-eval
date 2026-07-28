#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

CLIENT_NAMES = {
    "0":"guatemala-volcano", "1":"hurricane-florence", "2":"hurricane-harvey",
    "3":"hurricane-matthew", "4":"hurricane-michael", "5":"mexico-earthquake",
    "6":"midwest-flooding", "7":"palu-tsunami", "8":"santa-rosa-wildfire",
}
CLIENT_TYPES = {
    "0":"volcano", "1":"hurricane", "2":"hurricane", "3":"hurricane", "4":"hurricane",
    "5":"earthquake", "6":"flood", "7":"tsunami", "8":"wildfire",
}

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

class GlobalAnchorNet(nn.Module):
    def __init__(self, num_classes:int=2):
        super().__init__()
        self.conv1=nn.Conv2d(3,32,5,padding=2); self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(32,64,5,padding=2)
        self.fc1=nn.Linear(64*16*16,512); self.fc=nn.Linear(512,num_classes)
    def features(self,x):
        x=self.pool(F.relu(self.conv1(x))); x=self.pool(F.relu(self.conv2(x)))
        x=torch.flatten(x,1); return F.relu(self.fc1(x))
    def forward(self,x): return self.fc(self.features(x))

class ResidualHead(nn.Module):
    def __init__(self, feature_dim:int=512, num_classes:int=2):
        super().__init__(); self.fc=nn.Linear(feature_dim,num_classes)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)
    def forward(self,feature): return self.fc(feature)

def load_npz(path:Path)->Tuple[np.ndarray,np.ndarray]:
    obj=np.load(path, allow_pickle=True); keys=set(obj.files)
    if 'data' in keys and len(keys)==1:
        d=obj['data']
        if isinstance(d,np.ndarray) and d.shape==(): d=d.item()
        x,y=d['x'],d['y']
    else:
        x_key='x' if 'x' in keys else ('data' if 'data' in keys else None)
        y_key='y' if 'y' in keys else ('labels' if 'labels' in keys else None)
        if x_key is None or y_key is None: raise ValueError(f'{path} bad keys {obj.files}')
        x,y=obj[x_key],obj[y_key]
    x=np.asarray(x).astype('float32'); y=np.asarray(y).astype('int64')
    if x.ndim==4 and x.shape[-1]==3: x=np.transpose(x,(0,3,1,2))
    if x.max()>5.0: x=x/255.0
    return x,y

def normalize_disaster_type(name:str, raw:str='unknown')->str:
    x=(raw or '').strip().lower(); n=(name or '').strip().lower()
    if 'hurricane' in n: return 'hurricane'
    if 'wildfire' in n or 'bushfire' in n or 'fire' in n: return 'fire'
    if 'flood' in n: return 'flooding'
    if 'volcano' in n: return 'volcano'
    if 'tsunami' in n: return 'tsunami'
    if 'earthquake' in n: return 'earthquake'
    if x in ('wind','hurricane'): return 'hurricane'
    if x in ('wildfire','fire','bushfire'): return 'fire'
    if x in ('flood','flooding'): return 'flooding'
    if x in ('volcano','tsunami','earthquake'): return x
    return x or 'unknown'

def load_client_meta(dataset_dir:Path):
    meta={}; p=dataset_dir/'client_map.csv'
    if not p.exists(): return meta
    with p.open(newline='') as f:
        for row in csv.DictReader(f):
            cid=str(row.get('client_id') or row.get('client_idx') or row.get('id') or '').strip()
            if not cid: continue
            name=(row.get('event_name') or row.get('client_name') or row.get('name') or cid).strip()
            raw=(row.get('disaster_type') or row.get('type') or row.get('client_type') or 'unknown').strip()
            meta[cid]={'name':name,'type':normalize_disaster_type(name,raw)}
    return meta

def build_loaders(dataset_dir:Path,batch_size:int,num_workers:int,shuffle_train:bool=True):
    client_meta=load_client_meta(dataset_dir)
    train_dir,test_dir=dataset_dir/'train',dataset_dir/'test'
    files=sorted(train_dir.glob('*.npz'), key=lambda p:int(p.stem) if p.stem.isdigit() else p.stem)
    if not files: raise FileNotFoundError(f'No npz files in {train_dir}')
    clients=[]
    for tr in files:
        te=test_dir/tr.name
        if not te.exists(): raise FileNotFoundError(f'Missing {te}')
        xtr,ytr=load_npz(tr); xte,yte=load_npz(te)
        train_tensor=TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr))
        meta=client_meta.get(tr.stem,{})
        clients.append({'id':tr.stem,'name':meta.get('name',CLIENT_NAMES.get(tr.stem,tr.stem)),'type':meta.get('type',CLIENT_TYPES.get(tr.stem,'unknown')),'n_train':len(ytr),'n_test':len(yte),
            'train_pos_rate':float(ytr.mean()) if len(ytr) else 0.0,'test_pos_rate':float(yte.mean()) if len(yte) else 0.0,
            'x_train':xtr,'y_train':ytr,'stream_cursor':0,'stream_start_idx':0,'stream_end_idx':0,'stream_wraps':0,
            'train_loader':DataLoader(train_tensor,batch_size=batch_size,shuffle=shuffle_train,num_workers=num_workers,pin_memory=True),
            'test_loader':DataLoader(TensorDataset(torch.from_numpy(xte),torch.from_numpy(yte)),batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)})
    return clients

def make_sequential_stream_loader(client,batch_size:int,window_batches:int,num_workers:int):
    """Return a deterministic per-round window loader and advance the client's cursor.

    Each round sees the next contiguous local window, cycling at the end. Probe and
    normal local training both use this same window, so the signature describes the
    data actually used in that round. This is intended to simulate disaster data
    arriving sequentially over time, not IID reshuffling every round.
    """
    x=client['x_train']; y=client['y_train']; n=len(y)
    if n<=0: raise ValueError(f"Client {client['id']} has no training data")
    window=max(1,int(window_batches))*int(batch_size)
    start=int(client.get('stream_cursor',0))%n
    if window>=n:
        idx=np.concatenate([np.arange(start,n),np.arange(0,start)])
        end=start
        wraps=1
    else:
        end=(start+window)%n
        if start+window<=n:
            idx=np.arange(start,start+window)
            wraps=0
        else:
            idx=np.concatenate([np.arange(start,n),np.arange(0,end)])
            wraps=1
    client['stream_start_idx']=int(start); client['stream_end_idx']=int(end); client['stream_wraps']=int(wraps)
    client['stream_cursor']=int(end)
    ds=TensorDataset(torch.from_numpy(x[idx]),torch.from_numpy(y[idx]))
    return DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)

def prepare_round_loaders(clients,args):
    if args.data_order=='sequential_stream':
        return [make_sequential_stream_loader(c,args.batch_size,args.stream_window_batches,args.num_workers) for c in clients]
    return [c['train_loader'] for c in clients]

def clone_module(m,device): return deepcopy(m).to(device)

def average_state_dicts(states,weights,device):
    total=float(sum(weights))
    if total<=0: weights=[1.0]*len(states); total=float(len(states))
    out={}
    for k in states[0].keys():
        if torch.is_floating_point(states[0][k]):
            acc=None
            for sd,w in zip(states,weights):
                val=sd[k].detach().to(device)*(float(w)/total); acc=val if acc is None else acc+val
            out[k]=acc
        else: out[k]=states[0][k].detach().to(device)
    return out

def weighted_average_modules(modules,weights,template,device):
    out=clone_module(template,device); out.load_state_dict(average_state_dicts([m.state_dict() for m in modules],weights,device)); return out

def train_global_only_probe(anchor,loader,device,lr,probe_steps,signature_mode,return_details:bool=False):
    model=clone_module(anchor,device); model.train(); opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.0,weight_decay=0.0)
    before={k:v.detach().clone() for k,v in model.state_dict().items() if torch.is_floating_point(v)}
    chunks=[]; step_losses=[]; step_update_norms=[]; step_update_dims=[]
    cycled=0
    it=iter(loader)
    for step in range(probe_steps):
        try:
            xb,yb=next(it)
        except StopIteration:
            # Some xBD clients are tiny, e.g. mexico-earthquake has only 22 train samples.
            # Cycle the loader so every client produces exactly probe_steps chunks and
            # therefore comparable fixed-length gradient-path signatures.
            cycled+=1
            it=iter(loader)
            xb,yb=next(it)
        xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True); loss=F.cross_entropy(model(xb),yb); loss.backward(); opt.step()
        after=model.state_dict()
        if signature_mode=='head': keys=['fc.weight','fc.bias']
        elif signature_mode=='last': keys=['fc1.weight','fc1.bias','fc.weight','fc.bias']
        else: keys=list(before.keys())
        chunk=torch.cat([(after[k].detach()-before[k]).flatten().float().cpu() for k in keys if k in after])
        chunks.append(chunk); step_losses.append(float(loss.item())); step_update_norms.append(float(torch.norm(chunk).item())); step_update_dims.append(int(chunk.numel()))
        for k in before: before[k]=after[k].detach().clone()
    raw=torch.cat(chunks); raw_norm=float(torch.norm(raw).item()); sig=raw/torch.norm(raw).clamp_min(1e-12)
    if not return_details:
        return sig
    detail={
        'signature_dim':int(raw.numel()),
        'raw_norm':raw_norm,
        'mean_abs':float(raw.abs().mean().item()) if raw.numel() else 0.0,
        'std':float(raw.std(unbiased=False).item()) if raw.numel() else 0.0,
        'positive_fraction':float((raw>0).float().mean().item()) if raw.numel() else 0.0,
        'negative_fraction':float((raw<0).float().mean().item()) if raw.numel() else 0.0,
        'zero_fraction':float((raw==0).float().mean().item()) if raw.numel() else 0.0,
        'probe_loss_mean':float(np.mean(step_losses)) if step_losses else math.nan,
        'probe_loss_first':step_losses[0] if step_losses else math.nan,
        'probe_loss_last':step_losses[-1] if step_losses else math.nan,
        'step_update_norm_mean':float(np.mean(step_update_norms)) if step_update_norms else 0.0,
        'step_update_norm_max':float(np.max(step_update_norms)) if step_update_norms else 0.0,
        'probe_steps_requested':int(probe_steps),
        'probe_steps_observed':len(chunks),
        'probe_loader_cycles':int(cycled),
        'signature_mode':signature_mode,
    }
    return sig,detail

def log_similarity_edges(sim,clients,rnd):
    rows=[]; n=sim.shape[0]
    for i,c in enumerate(clients):
        order=sorted([j for j in range(n) if j!=i], key=lambda j:float(sim[i,j]), reverse=True)
        for rank,j in enumerate(order, start=1):
            d=clients[j]
            rows.append({'round':rnd,'source_client':c['id'],'source_name':c['name'],'source_type':c.get('type','unknown'),
                'target_client':d['id'],'target_name':d['name'],'target_type':d.get('type','unknown'),
                'rank':rank,'similarity':float(sim[i,j]),'same_type':int(c.get('type')==d.get('type'))})
    return rows

def train_anchor_only(anchor,loader,device,lr,local_epochs):
    la=clone_module(anchor,device); la.train()
    opt=torch.optim.SGD(la.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)
    loss_sum=0.0; n=0
    for _ in range(local_epochs):
        for xb,yb in loader:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True); logits=la(xb)
            loss=F.cross_entropy(logits,yb); loss.backward(); opt.step()
            loss_sum+=float(loss.item())*len(yb); n+=len(yb)
    return la,loss_sum/max(n,1)

def train_full_client(anchor,residual,loader,device,lr,local_epochs):
    la=clone_module(anchor,device); lrh=clone_module(residual,device); la.train(); lrh.train()
    opt=torch.optim.SGD(list(la.parameters())+list(lrh.parameters()),lr=lr,momentum=0.9,weight_decay=1e-4)
    loss_sum=0.0; n=0
    for _ in range(local_epochs):
        for xb,yb in loader:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True); feat=la.features(xb); logits=la.fc(feat)+lrh(feat)
            loss=F.cross_entropy(logits,yb); loss.backward(); opt.step()
            loss_sum+=float(loss.item())*len(yb); n+=len(yb)
    return la,lrh,loss_sum/max(n,1)

def residual_delta(local,base):
    lsd,bsd=local.state_dict(),base.state_dict()
    return {k:(lsd[k].detach().cpu()-bsd[k].detach().cpu()) for k in lsd if torch.is_floating_point(lsd[k])}

def add_residual_delta(base,deltas,weights,lam,device):
    out=clone_module(base,device); sd=out.state_dict(); total=float(sum(weights))
    if total<=0 or not deltas: return out
    for k in sd:
        if torch.is_floating_point(sd[k]):
            upd=None
            for d,w in zip(deltas,weights):
                if k not in d: continue
                val=d[k].to(device)*(float(w)/total); upd=val if upd is None else upd+val
            if upd is not None: sd[k].copy_(sd[k]+lam*upd)
    out.load_state_dict(sd); return out

def cosine_matrix(signatures):
    S=F.normalize(torch.stack(signatures),dim=1,eps=1e-12); sim=(S@S.T).cpu().numpy(); np.fill_diagonal(sim,0.0); return np.maximum(sim,0.0)

def relation_global_weights(sim,eps=1e-12):
    q=sim.sum(axis=1); a=np.ones_like(q)/len(q) if float(q.sum())<=eps else q/q.sum(); return q,a

def build_neighbours(sim,top_k,tau,mode,rng,fixed_cache=None):
    n=sim.shape[0]
    if mode=='fixed' and fixed_cache is not None: return fixed_cache
    out=[]
    for i in range(n):
        cand=[j for j in range(n) if j!=i]
        if mode=='random': rng.shuffle(cand); chosen=cand[:top_k]
        else: chosen=sorted(cand,key=lambda j:sim[i,j],reverse=True)[:top_k]
        out.append([(j,float(sim[i,j])) for j in chosen if sim[i,j]>tau])
    return out


def eval_pair_loss(anchor,residual,loader,device):
    anchor.eval(); residual.eval(); loss_sum=0.0; n=0
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            feat=anchor.features(xb); logits=anchor.fc(feat)+residual(feat)
            loss=F.cross_entropy(logits,yb); loss_sum+=float(loss.item())*len(yb); n+=len(yb)
    return loss_sum/max(n,1)

def blend_residual_toward_target(local_residual,target_residual,lam,device):
    out=clone_module(local_residual,device); osd=out.state_dict(); tsd=target_residual.state_dict()
    for k in osd:
        if torch.is_floating_point(osd[k]) and k in tsd:
            osd[k].copy_(osd[k] + lam*(tsd[k].to(device)-osd[k]))
    out.load_state_dict(osd); return out

def weighted_average_residuals(residuals,weights,template,device):
    return weighted_average_modules(residuals,weights,template,device)

def collaboration_biased_weights(clients,current,base_counts,beta):
    """Sample-size FedAvg weights softly biased by incoming collaboration graph.

    base_counts keep every client non-zero. incoming_score_i sums similarity scores
    from other clients that selected i as a neighbour. beta controls deviation from
    FedAvg: weight_i ∝ n_i * (1 + beta * normalized_incoming_i).
    """
    base=np.asarray([float(x) for x in base_counts],dtype=np.float64)
    base=np.maximum(base,1e-12)
    incoming=np.zeros(len(clients),dtype=np.float64)
    for src,nb in enumerate(current):
        for j,score in nb:
            if 0 <= int(j) < len(clients):
                incoming[int(j)] += max(float(score),0.0)
    if incoming.max() > 1e-12:
        norm=incoming / incoming.max()
    else:
        norm=incoming
    beta=max(float(beta),0.0)
    weights=base * (1.0 + beta*norm)
    if weights.sum() <= 0:
        weights=base
    return weights.tolist(), incoming.tolist(), norm.tolist()

def eval_one(anchor,residual,loader,device):
    anchor.eval();
    if residual is not None: residual.eval()
    correct=total=tp=tn=fp=fn=0; loss_sum=0.0
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
            if residual is None: logits=anchor(xb)
            else:
                feat=anchor.features(xb); logits=anchor.fc(feat)+residual(feat)
            loss=F.cross_entropy(logits,yb); pred=logits.argmax(1)
            loss_sum+=float(loss.item())*len(yb); correct+=int((pred==yb).sum()); total+=len(yb)
            tp+=int(((pred==1)&(yb==1)).sum()); tn+=int(((pred==0)&(yb==0)).sum()); fp+=int(((pred==1)&(yb==0)).sum()); fn+=int(((pred==0)&(yb==1)).sum())
    acc=correct/max(total,1); dr=tp/max(tp+fn,1); nr=tn/max(tn+fp,1); pp=tp/max(tp+fp,1); pn=tn/max(tn+fn,1)
    f1p=2*pp*dr/max(pp+dr,1e-12); f1n=2*pn*nr/max(pn+nr,1e-12)
    return {'loss':loss_sum/max(total,1),'accuracy':acc,'macro_f1':0.5*(f1p+f1n),'damaged_recall':dr,'nodamage_recall':nr,'tp':tp,'tn':tn,'fp':fp,'fn':fn,'n':total}

def eval_all(anchor,residuals,clients,mode,device):
    rows=[]
    for i,c in enumerate(clients):
        a=anchor[i] if isinstance(anchor,list) else anchor; r=eval_one(a,None if residuals is None else residuals[i],c['test_loader'],device)
        r.update({'client_idx':i,'client_id':c['id'],'client_name':c['name']}); rows.append(r)
    total=sum(r['n'] for r in rows)
    w=lambda k: sum(float(r[k])*r['n'] for r in rows)/max(total,1)
    return {'avg_accuracy':w('accuracy'),'worst_client_accuracy':min(float(r['accuracy']) for r in rows),'avg_macro_f1':w('macro_f1'),'avg_damaged_recall':w('damaged_recall'),'avg_nodamage_recall':w('nodamage_recall'),'avg_loss':w('loss'),'client_acc_std':float(np.std([float(r['accuracy']) for r in rows]))}, rows

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    fields=[]; seen=set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset-dir',default='dataset/xBD_event_balanced'); ap.add_argument('--out-dir',default='outputs/ga_dnc_pfl_v05')
    ap.add_argument('--mode',choices=['fedavg','local','residual_local','ga_dnc','ga_dnc_residual_consensus'],required=True)
    ap.add_argument('--rounds',type=int,default=50); ap.add_argument('--local-epochs',type=int,default=1); ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--lr',type=float,default=0.01); ap.add_argument('--probe-lr',type=float,default=None); ap.add_argument('--probe-steps',type=int,default=5)
    ap.add_argument('--signature-mode',choices=['head','last','all'],default='head'); ap.add_argument('--global-agg',choices=['fedavg','relation'],default='relation')
    ap.add_argument('--neighbour-mode',choices=['none','dynamic','fixed','random'],default='dynamic'); ap.add_argument('--refresh-interval',type=int,default=1)
    ap.add_argument('--top-k',type=int,default=2); ap.add_argument('--tau',type=float,default=0.0); ap.add_argument('--assist-lambda',type=float,default=0.1); ap.add_argument('--gate-margin',type=float,default=0.0); ap.add_argument('--consensus-mode',choices=['endpoint','trajectory'],default='endpoint'); ap.add_argument('--global-bias-beta',type=float,default=0.0,help='Softly bias FedAvg anchor weights by incoming collaboration importance; 0.0 = pure FedAvg')
    ap.add_argument('--data-order',choices=['shuffle','sequential_stream'],default='shuffle')
    ap.add_argument('--stream-window-batches',type=int,default=5,help='For sequential_stream, number of local mini-batches consumed per client per round')
    ap.add_argument('--seed',type=int,default=20260727); ap.add_argument('--num-workers',type=int,default=2); ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args(); set_seed(args.seed); rng=random.Random(args.seed); args.probe_lr=args.lr if args.probe_lr is None else args.probe_lr
    device=torch.device(args.device if args.device=='cpu' or torch.cuda.is_available() else 'cpu')
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True); (out_dir/'relation_matrices').mkdir(exist_ok=True)
    clients=build_loaders(Path(args.dataset_dir),args.batch_size,args.num_workers,shuffle_train=(args.data_order=='shuffle')); n=len(clients)
    print(f'Device={device} mode={args.mode} clients={n} rounds={args.rounds}')
    print('Counts:',[(c['id'],c['name'],c['n_train'],c['n_test'],round(c['train_pos_rate'],3)) for c in clients])
    if args.mode=='local': anchors=[GlobalAnchorNet(2).to(device) for _ in range(n)]; global_anchor=None; residuals=None
    else: global_anchor=GlobalAnchorNet(2).to(device); anchors=None; residuals=[ResidualHead(512,2).to(device) for _ in range(n)] if args.mode in ('residual_local','ga_dnc','ga_dnc_residual_consensus') else None
    round_rows=[]; client_rows=[]; neigh_rows=[]; weight_rows=[]; signature_rows=[]; similarity_edge_rows=[]; neighbour_decision_rows=[]; fixed_neigh=None; last_sim=np.zeros((n,n),dtype=np.float32); last_q=np.zeros(n); last_a=np.ones(n)/max(n,1)
    for rnd in range(1,args.rounds+1):
        train_loaders=prepare_round_loaders(clients,args)
        losses=[]
        if args.mode=='fedavg':
            loc=[]
            for c,loader in zip(clients,train_loaders):
                la,loss=train_anchor_only(global_anchor,loader,device,args.lr,args.local_epochs); loc.append(la); losses.append(loss)
            global_anchor=weighted_average_modules(loc,[c['n_train'] for c in clients],global_anchor,device)
            summary,per=eval_all(global_anchor,None,clients,args.mode,device); no_neigh=n
        elif args.mode=='local':
            nxt=[]
            for i,(c,loader) in enumerate(zip(clients,train_loaders)):
                la,loss=train_anchor_only(anchors[i],loader,device,args.lr,args.local_epochs); nxt.append(la); losses.append(loss)
            anchors=nxt; summary,per=eval_all(anchors,None,clients,args.mode,device); no_neigh=n
        elif args.mode=='ga_dnc_residual_consensus':
            # v0.8 residual-consensus:
            # shared global_anchor gives comparable signatures and feature space;
            # each client trains its own residual r_i locally;
            # neighbours propose a consensus residual endpoint target;
            # client i moves only a small step from r_i_local toward that target, then gates by local loss.
            if rnd==1 or args.refresh_interval<=1 or rnd%args.refresh_interval==0:
                sigs=[]
                for c,loader in zip(clients,train_loaders):
                    sig,detail=train_global_only_probe(global_anchor,loader,device,args.probe_lr,args.probe_steps,args.signature_mode,return_details=True)
                    sigs.append(sig)
                    signature_rows.append({'round':rnd,'client_id':c['id'],'client_name':c['name'],'client_type':c.get('type','unknown'),
                        'stream_start_idx':c.get('stream_start_idx',0),'stream_end_idx':c.get('stream_end_idx',0),'stream_wraps':c.get('stream_wraps',0),**detail})
                last_sim=cosine_matrix(sigs)
                np.savetxt(out_dir/'relation_matrices'/f'similarity_round_{rnd:03d}.csv',last_sim,delimiter=',',fmt='%.6f')
                last_q,last_a=relation_global_weights(last_sim)
                similarity_edge_rows.extend(log_similarity_edges(last_sim,clients,rnd))

            loc_anchors=[]; loc_resids=[]; losses=[]
            for i,(c,loader) in enumerate(zip(clients,train_loaders)):
                la,lrh,loss=train_full_client(global_anchor,residuals[i],loader,device,args.lr,args.local_epochs)
                loc_anchors.append(la); loc_resids.append(lrh); losses.append(loss)

            current=build_neighbours(last_sim,args.top_k,args.tau,args.neighbour_mode,rng,None)
            # Soft collaboration-biased FedAvg for the global anchor.
            # Every client keeps non-zero sample-size contribution; clients selected by more/high-similarity neighbours get a mild boost.
            global_weights,incoming_scores,incoming_norms=collaboration_biased_weights(clients,current,[c['n_train'] for c in clients],args.global_bias_beta)
            global_anchor=weighted_average_modules(loc_anchors,global_weights,global_anchor,device)
            for i,c in enumerate(clients):
                weight_rows.append({'round':rnd,'client_idx':i,'client_id':c['id'],'client_name':c['name'],
                    'relation_importance_q':float(last_q[i]),'global_weight_a':float(last_a[i]),
                    'incoming_collab_score':float(incoming_scores[i]),'incoming_collab_norm':float(incoming_norms[i]),
                    'global_bias_beta':float(args.global_bias_beta),'aggregation_weight_used':float(global_weights[i])})

            nxt=[]; no_neigh=0; accepted=0; rejected=0
            for i,c in enumerate(clients):
                nb=[] if args.neighbour_mode=='none' else current[i]
                if not nb:
                    nxt.append(loc_resids[i]); no_neigh+=1
                    neigh_rows.append({'round':rnd,'client_idx':i,'client_id':c['id'],'client_name':c['name'],'client_type':c.get('type','unknown'),
                        'neighbours':'','neighbour_names':'','neighbour_types':'','scores':'','same_type_neighbour_count':0,
                        'no_neighbour':1,'assist_accepted':0,'local_loss':'','candidate_loss':''})
                    continue
                js=[j for j,_ in nb]; scores=[sc for _,sc in nb]
                # consensus target in shared residual space: weighted average of neighbour local residual endpoints.
                target=weighted_average_residuals([loc_resids[j] for j in js],scores,loc_resids[i],device)
                candidate=blend_residual_toward_target(loc_resids[i],target,args.assist_lambda,device)
                local_loss=eval_pair_loss(global_anchor,loc_resids[i],train_loaders[i],device)
                candidate_loss=eval_pair_loss(global_anchor,candidate,train_loaders[i],device)
                same=sum(1 for j in js if clients[j].get('type')==c.get('type'))
                if candidate_loss <= local_loss - args.gate_margin:
                    nxt.append(candidate); accepted+=1; flag=1
                else:
                    nxt.append(loc_resids[i]); rejected+=1; flag=0
                neigh_rows.append({'round':rnd,'client_idx':i,'client_id':c['id'],'client_name':c['name'],'client_type':c.get('type','unknown'),
                    'neighbours':' '.join(map(str,js)),'neighbour_names':' | '.join(clients[j]['name'] for j in js),
                    'neighbour_types':' | '.join(clients[j].get('type','unknown') for j in js),'scores':' '.join(f'{sc:.6f}' for sc in scores),
                    'same_type_neighbour_count':same,'no_neighbour':0,'assist_accepted':flag,
                    'local_loss':local_loss,'candidate_loss':candidate_loss})
            residuals=nxt
            summary,per=eval_all(global_anchor,residuals,clients,args.mode,device)
            no_neigh=int(no_neigh)
            summary['accepted_assist_clients']=accepted; summary['rejected_assist_clients']=rejected
        else:
            if args.mode=='ga_dnc' and (rnd==1 or args.refresh_interval<=1 or rnd%args.refresh_interval==0):
                sigs=[]
                for c,loader in zip(clients,train_loaders):
                    sig,detail=train_global_only_probe(global_anchor,loader,device,args.probe_lr,args.probe_steps,args.signature_mode,return_details=True)
                    sigs.append(sig)
                    signature_rows.append({'round':rnd,'client_id':c['id'],'client_name':c['name'],'client_type':c.get('type','unknown'),
                        'stream_start_idx':c.get('stream_start_idx',0),'stream_end_idx':c.get('stream_end_idx',0),'stream_wraps':c.get('stream_wraps',0),**detail})
                last_sim=cosine_matrix(sigs); np.savetxt(out_dir/'relation_matrices'/f'similarity_round_{rnd:03d}.csv',last_sim,delimiter=',',fmt='%.6f'); last_q,last_a=relation_global_weights(last_sim)
                similarity_edge_rows.extend(log_similarity_edges(last_sim,clients,rnd))
            loc_anchors=[]; loc_resids=[]; deltas=[]
            for i,(c,loader) in enumerate(zip(clients,train_loaders)):
                la,lrh,loss=train_full_client(global_anchor,residuals[i],loader,device,args.lr,args.local_epochs); loc_anchors.append(la); loc_resids.append(lrh); deltas.append(residual_delta(lrh,residuals[i])); losses.append(loss)
            weights=[float(x) for x in last_a] if args.mode=='ga_dnc' and args.global_agg=='relation' else [c['n_train'] for c in clients]
            global_anchor=weighted_average_modules(loc_anchors,weights,global_anchor,device)
            for i,c in enumerate(clients): weight_rows.append({'round':rnd,'client_idx':i,'client_id':c['id'],'client_name':c['name'],'relation_importance_q':float(last_q[i]),'global_weight_a':float(last_a[i]),'aggregation_weight_used':float(weights[i])})
            if args.mode=='residual_local' or args.neighbour_mode=='none':
                residuals=loc_resids; no_neigh=n
                for i,c in enumerate(clients): neigh_rows.append({'round':rnd,'client_idx':i,'client_id':c['id'],'client_name':c['name'],'client_type':c.get('type','unknown'),'neighbours':'','neighbour_names':'','neighbour_types':'','scores':'','same_type_neighbour_count':0,'no_neighbour':1})
            else:
                refresh=(rnd==1 or args.refresh_interval<=1 or rnd%args.refresh_interval==0)
                if refresh or fixed_neigh is None:
                    current=build_neighbours(last_sim,args.top_k,args.tau,args.neighbour_mode,rng,fixed_neigh if args.neighbour_mode=='fixed' else None)
                    if args.neighbour_mode=='fixed' and fixed_neigh is None: fixed_neigh=current
                else: current=fixed_neigh if args.neighbour_mode=='fixed' else build_neighbours(last_sim,args.top_k,args.tau,args.neighbour_mode,rng,None)
                nxt=[]; no_neigh=0
                for i,c in enumerate(clients):
                    nb=current[i]
                    if not nb:
                        nxt.append(loc_resids[i]); no_neigh+=1; neigh_rows.append({'round':rnd,'client_idx':i,'client_id':c['id'],'client_name':c['name'],'client_type':c.get('type','unknown'),'neighbours':'','neighbour_names':'','neighbour_types':'','scores':'','same_type_neighbour_count':0,'no_neighbour':1})
                    else:
                        js=[j for j,_ in nb]; scores=[s for _,s in nb]; nxt.append(add_residual_delta(loc_resids[i],[deltas[j] for j in js],scores,args.assist_lambda,device))
                        neigh_rows.append({'round':rnd,'client_idx':i,'client_id':c['id'],'client_name':c['name'],'neighbours':' '.join(map(str,js)),'scores':' '.join(f'{s:.6f}' for s in scores),'no_neighbour':0})
                residuals=nxt
            summary,per=eval_all(global_anchor,residuals,clients,args.mode,device)
        row={'round':rnd,'mode':args.mode,'data_order':args.data_order,'stream_window_batches':args.stream_window_batches,'global_agg':args.global_agg,'neighbour_mode':args.neighbour_mode,'refresh_interval':args.refresh_interval,'top_k':args.top_k,'tau':args.tau,'assist_lambda':args.assist_lambda,'train_loss_mean':float(np.mean(losses)) if losses else math.nan,'num_no_neighbour_clients':int(no_neigh),**summary}
        round_rows.append(row)
        for r in per:
            c=clients[int(r['client_idx'])]
            client_rows.append({'round':rnd,'mode':args.mode,'data_order':args.data_order,'stream_start_idx':c.get('stream_start_idx',0),'stream_end_idx':c.get('stream_end_idx',0),'stream_wraps':c.get('stream_wraps',0),'global_agg':args.global_agg,'neighbour_mode':args.neighbour_mode,'top_k':args.top_k,'tau':args.tau,**r})
        print(json.dumps(row,ensure_ascii=False))
    write_csv(out_dir/'round_metrics.csv',round_rows); write_csv(out_dir/'client_metrics.csv',client_rows); write_csv(out_dir/'neighbours.csv',neigh_rows); write_csv(out_dir/'global_weights.csv',weight_rows); write_csv(out_dir/'signature_metrics.csv',signature_rows); write_csv(out_dir/'similarity_edges.csv',similarity_edge_rows); write_csv(out_dir/'neighbour_decisions.csv',neighbour_decision_rows)
    (out_dir/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    print('FINAL',json.dumps(round_rows[-1],ensure_ascii=False))
if __name__=='__main__': main()

import torch
import time
import numpy as np
print("="*60)
print("ODIN RUGGED TEST SUITE - THOROUGH EDITION")
print("="*60)
model.eval()
results={}
print("\n[1] SPEED TEST - 100 iterations")
h=model.init_state(1,device)
u=torch.randn(1,config.input_dim,device=device)
times=[]
for i in range(100):
 if device=='cuda':torch.cuda.synchronize()
 t0=time.perf_counter()
 with torch.no_grad():h,y,_=model(h,u)
 if device=='cuda':torch.cuda.synchronize()
 times.append((time.perf_counter()-t0)*1000)
avg_ms=sum(times)/len(times)
results['speed']=avg_ms<50
print(f"  Avg: {avg_ms:.2f}ms | Min: {min(times):.2f}ms | Max: {max(times):.2f}ms")
print(f"  {'PASS' if avg_ms<50 else 'FAIL'}: target <50ms")
print("\n[2] STATE STABILITY - 500 steps")
h=model.init_state(1,device)
norms=[]
for i in range(500):
 u=torch.randn(1,config.input_dim,device=device)
 with torch.no_grad():h,_,_=model(h,u)
 norms.append(h.norm().item())
stable=max(norms)<1e6 and min(norms)>1e-6
results['stable']=stable
print(f"  Range: {min(norms):.4f} - {max(norms):.4f}")
print(f"  Trend: {norms[-1]/norms[0]:.2f}x (last/first)")
print(f"  {'PASS' if stable else 'FAIL'}: no explosion or collapse")
print("\n[3] GRADIENT FLOW - backprop test")
h=model.init_state(1,device)
u=torch.randn(1,config.input_dim,device=device,requires_grad=True)
model.train()
h2,y,_=model(h,u)
loss=y.sum()
loss.backward()
grad_ok=u.grad is not None and not torch.isnan(u.grad).any()
results['gradient']=grad_ok
model.eval()
print(f"  Gradient exists: {u.grad is not None}")
print(f"  No NaN: {not torch.isnan(u.grad).any() if u.grad is not None else False}")
print(f"  {'PASS' if grad_ok else 'FAIL'}: gradients flow properly")
print("\n[4] DETERMINISM - same input = same output")
h1=model.init_state(1,device)
u_fixed=torch.randn(1,config.input_dim,device=device)
torch.manual_seed(42)
with torch.no_grad():_,y1,_=model(h1,u_fixed)
h2=model.init_state(1,device)
torch.manual_seed(42)
with torch.no_grad():_,y2,_=model(h2,u_fixed)
determ=torch.allclose(y1,y2,atol=1e-5)
results['determinism']=determ
print(f"  Outputs match: {determ}")
print(f"  {'PASS' if determ else 'FAIL'}: reproducible outputs")
print("\n[5] ACTION DIVERSITY - distribution check")
h=model.init_state(1,device)
actions=[0]*10
for i in range(200):
 u=torch.randn(1,config.input_dim,device=device)
 with torch.no_grad():h,_,_=model(h,u)
 _,out=model.heads(h)
 a=out['action']['action_logits'].argmax().item()
 actions[a]+=1
unique=sum(1 for c in actions if c>0)
diverse=unique>=3
results['diverse']=diverse
print(f"  Distribution: {actions}")
print(f"  Unique: {unique}/10")
print(f"  {'PASS' if diverse else 'WARN'}: using multiple actions")
print("\n[6] STANCE SANITY - check distribution")
h=model.init_state(1,device)
stances=[0]*4
for i in range(200):
 u=torch.randn(1,config.input_dim,device=device)
 with torch.no_grad():h,_,_=model(h,u)
 _,out=model.heads(h)
 s=out['cognitive']['stance_logits'].argmax().item()
 stances[s]+=1
stance_ok=max(stances)<180
results['stance']=stance_ok
print(f"  EXPLORE:{stances[0]} PLAN:{stances[1]} EXECUTE:{stances[2]} REFLECT:{stances[3]}")
print(f"  {'PASS' if stance_ok else 'WARN'}: not stuck on one stance")
print("\n[7] CONFIDENCE CALIBRATION")
h=model.init_state(1,device)
confs=[]
for i in range(100):
 u=torch.randn(1,config.input_dim,device=device)
 with torch.no_grad():h,_,_=model(h,u)
 _,out=model.heads(h)
 c=out['cognitive']['confidence'].item()
 confs.append(c)
conf_spread=max(confs)-min(confs)
conf_ok=conf_spread>0.1 and 0.1<np.mean(confs)<0.9
results['confidence']=conf_ok
print(f"  Range: {min(confs):.2%} - {max(confs):.2%}")
print(f"  Spread: {conf_spread:.2%}")
print(f"  {'PASS' if conf_ok else 'WARN'}: confidence varies")
print("\n[8] TIMESCALE BLOCKS - checking each")
h=model.init_state(1,device)
u=torch.randn(1,config.input_dim,device=device)
blocks=[]
idx=0
for dim in config.timescales:
 blocks.append(h[0,idx:idx+dim].norm().item())
 idx+=dim
block_ok=all(b>0 for b in blocks)
results['blocks']=block_ok
print(f"  Block norms: {[f'{b:.4f}' for b in blocks]}")
print(f"  {'PASS' if block_ok else 'FAIL'}: all timescales active")
print("\n[9] LONG SEQUENCE - 1000 steps")
h=model.init_state(1,device)
for i in range(1000):
 u=torch.randn(1,config.input_dim,device=device)
 with torch.no_grad():h,_,_=model(h,u)
long_stable=h.norm().item()<1e6 and h.norm().item()>1e-6
results['long']=long_stable
print(f"  Final norm: {h.norm().item():.4f}")
print(f"  {'PASS' if long_stable else 'FAIL'}: stable over 1000 steps")
print("\n[10] TRAINING METRICS")
l=history['train_loss'][-1]
sa=history['stance_acc'][-1]
aa=history['action_acc'][-1]
train_ok=l<2.0 and sa>0.35 and aa>0.25
results['train']=train_ok
print(f"  Loss: {l:.4f} (target <2.0)")
print(f"  Stance: {sa:.2%} (target >35%)")
print(f"  Action: {aa:.2%} (target >25%)")
print(f"  {'PASS' if train_ok else 'WARN'}: training converged")
print("\n"+"="*60)
print("FINAL RESULTS")
print("="*60)
passed=sum(results.values())
total=len(results)
for k,v in results.items():
 print(f"  {k}: {'PASS' if v else 'FAIL'}")
print(f"\nScore: {passed}/{total}")
if passed>=8:print("MODEL READY FOR SCALING")
elif passed>=6:print("MODEL OK - minor issues")
else:print("MODEL NEEDS WORK")

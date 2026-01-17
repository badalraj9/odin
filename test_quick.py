import torch
import time
print("ODIN TEST SUITE")
model.eval()
h=model.init_state(1,device)
u=torch.randn(1,config.input_dim,device=device)
t=[]
for i in range(100):
 t0=time.perf_counter()
 with torch.no_grad():h,y,_=model(h,u)
 t.append((time.perf_counter()-t0)*1000)
print("Speed:",sum(t)/100,"ms")
h=model.init_state(1,device)
n=[]
for i in range(100):
 u=torch.randn(1,config.input_dim,device=device)
 with torch.no_grad():h,_,_=model(h,u)
 n.append(h.norm().item())
print("Norm range:",min(n),"-",max(n))
print("Loss:",history['train_loss'][-1])
print("Stance acc:",history['stance_acc'][-1])
print("Action acc:",history['action_acc'][-1])

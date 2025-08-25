import math
import torch
from torch import nn

torch.manual_seed(0)

# V1:
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])

mlp = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())
opt = torch.optim.SGD(mlp.parameters(), lr=0.1)
bce = nn.BCELoss()

with torch.no_grad():
    init_loss = bce(mlp(X), y).item()
for _ in range(200):
    opt.zero_grad()
    loss = bce(mlp(X), y)
    loss.backward()
    opt.step()
print(f"[V1] XOR BCE loss: init={init_loss:.4f}  final={loss.item():.4f}")

# V2:
def grad_norms(depth=6, width=64, act="sigmoid"):
    layers = [nn.Linear(16, width)]
    for _ in range(depth):
        layers.append(nn.Linear(width, width))
        layers.append(nn.Sigmoid() if act=="sigmoid" else nn.ReLU())
    layers.append(nn.Linear(width, 1))
    net = nn.Sequential(*layers)

    x = torch.randn(32, 16)
    y = torch.randn(32, 1)
    loss = nn.MSELoss()(net(x), y)
    loss.backward()

    norms = [m.weight.grad.norm().item()
             for m in net if isinstance(m, nn.Linear)]
    return norms

def brief(ns):
    mid = ns[len(ns)//2]
    return f"{ns[0]:.2e}, ... {mid:.2e} ..., {ns[-1]:.2e}"

sig = grad_norms(act="sigmoid")
rel = grad_norms(act="relu")
print(f"[V2] Sigmoid grad norms: {brief(sig)}")
print(f"[V2] ReLU    grad norms: {brief(rel)}")

# V3:
Q = torch.tensor([[1.,0.,1.,0.],
                  [0.,1.,0.,1.],
                  [1.,1.,0.,0.]])
K = torch.tensor([[1.,0.,1.,0.],
                  [1.,1.,0.,0.],
                  [0.,1.,1.,0.]])
V = torch.tensor([[1.,2.,0.,0.],
                  [0.,1.,3.,0.],
                  [2.,0.,0.,1.]])

scores = (Q @ K.T) / math.sqrt(Q.shape[-1])
weights = torch.softmax(scores, dim=-1)
out = weights @ V
print("[V3] Attention weights:\n", weights.round(decimals=3))
print("[V3] Attention outputs:\n", out.round(decimals=3))
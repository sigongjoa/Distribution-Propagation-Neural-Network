# 그래프 기반 분포 전파 PoC 코드 정리

아래는 DPNN의 **그래프 자체 연산**(PDE 및 ODE) PoC를 위한 핵심 코드 스켈레톤입니다. 각 노드에 Gaussian 분포 상태를 유지하고, 그래프 라플라시안 및 Neural ODE 관점에서 상태를 업데이트합니다.

---

## 1. 공통 모듈

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torchdiffeq import odeint

# GaussianDistribution 정의 (mu, var)
class GaussianDistribution:
    def __init__(self, mu: torch.Tensor, var: torch.Tensor):
        self.mu = mu      # (num_nodes, d)
        self.var = var    # (num_nodes, d)

    def sample(self):
        eps = torch.randn_like(self.mu)
        return self.mu + torch.sqrt(self.var) * eps
```

---

## 2. Graph PDE Layer (라플라시안 기반)

```python
class GraphPDELayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 학습 가능한 엣지별 가중치 (간단화)
        self.theta = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, dist: GaussianDistribution, edge_index):
        # edge_index: [2, E]
        mu, var = dist.mu, dist.var  # (N, d)
        N, d = mu.size()
        # 메시지 전달: 이웃 차이
        row, col = edge_index
        diff_mu = mu[col] - mu[row]      # (E, d)
        diff_var = var[col] - var[row]
        # 가중합
        agg_mu = torch.zeros_like(mu)
        agg_var = torch.zeros_like(var)
        agg_mu = agg_mu.index_add_(0, row, diff_mu)
        agg_var = agg_var.index_add_(0, row, diff_var)
        # PDE 업데이트
        new_mu  = mu  + agg_mu
        new_var = var + agg_var
        # 파라미터화
        return GaussianDistribution(
            mu  = self.theta(new_mu),
            var = F.softplus(self.theta(new_var))
        )
```

---

## 3. Graph Neural ODE Func

```python
class GraphODEFunc(nn.Module):
    def __init__(self, pde_layer):
        super().__init__()
        self.pde_layer = pde_layer

    def forward(self, t, z, **kwargs):
        # z: concatenated state [mu||logvar]
        edge_index = kwargs['edge_index']
        N, twod = z.size()
        d = twod // 2
        mu, logvar = z[:, :d], z[:, d:]
        var = torch.exp(logvar)
        dist = GaussianDistribution(mu, var)
        # 한 스텝 PDE 연산
        new_dist = self.pde_layer(dist, edge_index)
        # concatenation for ODE solver
        return torch.cat([new_dist.mu, torch.log(new_dist.var)], dim=-1)
```

---

## 4. PoC 실행 스크립트

```python
# 1) 그래프 정의 (예: 4 노드 완전 그래프)
edge_index = torch.tensor([
    [0,0,0,1,1,1,2,2,2,3,3,3],
    [1,2,3,0,2,3,0,1,3,0,1,2]
], dtype=torch.long)
N, d = 4, 16
# 초기 분포: 무작위 mu, 단위 var
init_mu  = torch.randn(N, d)
init_var = torch.ones(N, d)
dist = GaussianDistribution(init_mu, init_var)

# 2) Graph PDE Layer 생성
pde_layer = GraphPDELayer(d, d)

def run_pde(dist, steps=10):
    for _ in range(steps):
        dist = pde_layer(dist, edge_index)
    return dist

# 순전파 예시
final_dist_pde = run_pde(dist)
print("PDE PoC result mu:", final_dist_pde.mu)

# 3) Graph Neural ODE 실행
odefunc = GraphODEFunc(pde_layer)
# zT: concatenated state
zT = torch.cat([dist.mu, torch.log(dist.var)], dim=-1)  # (N, 2d)
# 연속시간 통합
t_space = torch.linspace(0, 1, steps=20)
zs = odeint(odefunc, zT, t_space, rtol=1e-3, atol=1e-3, method='dopri5', options=dict(edge_index=edge_index))
# 최종
z0 = zs[-1]
mu0, logvar0 = z0[:, :d], z0[:, d:]
final_dist_ode = GaussianDistribution(mu0, torch.exp(logvar0))
print("ODE PoC result mu:", final_dist_ode.mu)

# === PDE 방식 학습 루프 (순전파/역전파) ===
optimizer = torch.optim.Adam(pde_layer.parameters(), lr=1e-3)
target_mu  = torch.ones(N, d)
target_var = 0.5 * torch.ones(N, d)
for epoch in range(1, 11):
    optimizer.zero_grad()
    pred_dist = run_pde(dist)
    loss_mu  = F.mse_loss(pred_dist.mu, target_mu)
    loss_var = F.mse_loss(pred_dist.var, target_var)
    loss = loss_mu + loss_var
    loss.backward()
    optimizer.step()
    print(f"PDE Training Epoch {epoch}, Loss: {loss.item():.4f}")

# === Neural ODE 방식 학습 루프 (순전파/역전파) ===
optimizer_ode = torch.optim.Adam(odefunc.parameters(), lr=1e-3)
for epoch in range(1, 11):
    optimizer_ode.zero_grad()
    zs = odeint(odefunc, zT, t_space, rtol=1e-3, atol=1e-3, method='dopri5', options=dict(edge_index=edge_index))
    z0 = zs[-1]
    mu0, logvar0 = z0[:, :d], z0[:, d:]
    pred_mu  = mu0
    pred_var = torch.exp(logvar0)
    loss_mu  = F.mse_loss(pred_mu, target_mu)
    loss_var = F.mse_loss(pred_var, target_var)
    loss = loss_mu + loss_var
    loss.backward()
    optimizer_ode.step()
    print(f"ODE Training Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

위 코드를 통해 **그래프 라플라시안 기반 분포 PDE**와 **Neural ODE** 방식의 **그래프 분포 전파 PoC**를 실행·검증할 수 있습니다.

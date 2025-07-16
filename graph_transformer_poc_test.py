
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import dense_to_sparse, scatter

# GaussianDistribution (μ, var)
class GaussianDistribution:
    def __init__(self, mu: torch.Tensor, var: torch.Tensor):
        self.mu  = mu    # (N, d)
        self.var = var  # (N, d)

    def sample(self):
        eps = torch.randn_like(self.mu)
        return self.mu + torch.sqrt(self.var) * eps

# Graph Attention Layer
class GraphDistAttention(MessagePassing):
    def __init__(self, d_model, heads=4):
        super().__init__(aggr='add')  # sum aggregation
        self.heads   = heads
        self.d_k     = d_model // heads
        # Q/K/V projections for μ and var
        self.q_mu  = nn.Linear(d_model, d_model)
        self.q_var = nn.Linear(d_model, d_model)
        self.k_mu  = nn.Linear(d_model, d_model)
        self.k_var = nn.Linear(d_model, d_model)
        self.v_mu  = nn.Linear(d_model, d_model)
        self.v_var = nn.Linear(d_model, d_model)
        self.out_mu  = nn.Linear(d_model, d_model)
        self.out_var = nn.Linear(d_model, d_model)

    def forward(self, dist: GaussianDistribution, edge_index):
        # dist.mu/var: (N, d)
        N, d = dist.mu.size()
        # project and flatten heads
        Qm = self.q_mu(dist.mu).view(N, self.heads * self.d_k)
        Qv = F.softplus(self.q_var(dist.var)).view(N, self.heads * self.d_k)
        Km = self.k_mu(dist.mu).view(N, self.heads * self.d_k)
        Kv = F.softplus(self.k_var(dist.var)).view(N, self.heads * self.d_k)
        Vm = self.v_mu(dist.mu).view(N, self.heads * self.d_k)
        Vv = F.softplus(self.v_var(dist.var)).view(N, self.heads * self.d_k)
        # propagate
        mu_out_flat, var_out_flat = self.propagate(edge_index, size=(N, N), Qm=Qm, Qv=Qv, Km=Km, Kv=Kv, Vm=Vm, Vv=Vv)
        # combine heads
        mu_out  = mu_out_flat.view(N, d)
        var_out = var_out_flat.view(N, d)
        return GaussianDistribution(mu=self.out_mu(mu_out), var=F.softplus(self.out_var(var_out)))

    def message(self, Qm_i, Qv_i, Km_j, Kv_j, Vm_j, Vv_j):
        # Qm_i, Qv_i, Km_j, Kv_j, Vm_j, Vv_j are all (num_edges, heads * d_k)
        num_edges = Qm_i.size(0)
        # Reshape to (num_edges, heads, d_k)
        Qm_i_reshaped = Qm_i.view(num_edges, self.heads, self.d_k)
        Qv_i_reshaped = Qv_i.view(num_edges, self.heads, self.d_k)
        Km_j_reshaped = Km_j.view(num_edges, self.heads, self.d_k)
        Kv_j_reshaped = Kv_j.view(num_edges, self.heads, self.d_k)
        Vm_j_reshaped = Vm_j.view(num_edges, self.heads, self.d_k)
        Vv_j_reshaped = Vv_j.view(num_edges, self.heads, self.d_k)

        # Compute KL approx between Q and K for each head
        Em = (Qm_i_reshaped - Km_j_reshaped).pow(2) / (2 * Kv_j_reshaped) + 0.5*(Qv_i_reshaped/Kv_j_reshaped - (Qv_i_reshaped/Kv_j_reshaped).log() - 1)
        # attention weights: (num_edges, heads)
        alpha = F.softmax(-Em.sum(dim=-1) / (self.d_k**0.5), dim=0)
        # weighted sum for μ and var: (num_edges, heads, d_k)
        mu_msg_reshaped  = Vm_j_reshaped * alpha.unsqueeze(-1)
        var_msg_reshaped = Vv_j_reshaped * (alpha.unsqueeze(-1).pow(2))
        
        # Flatten back to (num_edges, heads * d_k)
        mu_msg_flat = mu_msg_reshaped.view(num_edges, self.heads * self.d_k)
        var_msg_flat = var_msg_reshaped.view(num_edges, self.heads * self.d_k)
        return mu_msg_flat, var_msg_flat

    def aggregate(self, inputs, index, dim_size):
        mu_msgs_flat, var_msgs_flat = inputs # (num_edges, heads * d_k)

        # Perform summation using scatter
        mu_out_flat = scatter(mu_msgs_flat, index, dim=0, dim_size=dim_size, reduce='sum')
        var_out_flat = scatter(var_msgs_flat, index, dim=0, dim_size=dim_size, reduce='sum')

        return mu_out_flat, var_out_flat

# Graph Transformer Block
class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff):
        super().__init__()
        # attention
        self.gattn = GraphDistAttention(d_model, heads)
        # feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, dist: GaussianDistribution, edge_index):
        # 1) graph attention
        attn_out = self.gattn(dist, edge_index)
        # 2) residual + norm
        mu1  = self.norm1(dist.mu + attn_out.mu)
        var1 = self.norm1(dist.var + attn_out.var)
        # 3) feed-forward on μ; var updated via simple combine
        ff_mu  = self.ffn(mu1)
        ff_var = var1  # or learnable variance transform
        mu2 = self.norm2(mu1 + ff_mu)
        var2 = self.norm2(var1 + ff_var)
        return GaussianDistribution(mu=mu2, var=var2)

# PoC 실행 예시
# 1) 입력 토큰 분포 임베딩
# Dummy embedding layers for demonstration
class DummyEmbedMu(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, tokens):
        return self.embedding(tokens)

class DummyEmbedVar(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    def forward(self, tokens):
        return self.embedding(tokens)

class DummyLMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)
    def forward(self, mu):
        return self.linear(mu)

vocab_size = 100
N = 10 # Number of nodes (tokens)
d_model = 64
heads = 4
d_ff = 128

embed_mu = DummyEmbedMu(vocab_size, d_model)
embed_var = DummyEmbedVar(vocab_size, d_model)
lm_head = DummyLMHead(d_model, vocab_size)

tokens = torch.randint(0, vocab_size, (N,))  # N nodes
token_mu = embed_mu(tokens)
token_var = F.softplus(embed_var(tokens)) # Ensure variance is positive
dist = GaussianDistribution(token_mu, token_var)

# 2) 그라프 정의 (완전 그래프)
adj_matrix = torch.ones(N, N)
edge_index = dense_to_sparse(adj_matrix)[0]

# 3) 블록 스택
blocks = nn.ModuleList([GraphTransformerBlock(d_model, heads=heads, d_ff=d_ff) for _ in range(3)])

# 4) 순전파
print("Initial mu shape:", dist.mu.shape)
print("Initial var shape:", dist.var.shape)

for i, block in enumerate(blocks):
    dist = block(dist, edge_index)
    print(f"After block {i+1}, mu shape: {dist.mu.shape}, var shape: {dist.var.shape}")

# 5) LM 헤드
logits = lm_head(dist.mu)  # (N, vocab_size)
print("Final logits shape:", logits.shape)

print("Graph-based Transformer PoC executed successfully!")

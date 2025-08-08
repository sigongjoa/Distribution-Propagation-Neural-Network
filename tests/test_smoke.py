import dpnn, torch
from dpnn.dists.gaussian import GaussianDiag
from dpnn.layers.linear import DistLinear
from dpnn.layers.blocks import DistTransformerBlock
from dpnn.core.config import preset_config, Preset

def test_forward():
    x = torch.randn(2, 4, 16)                     # (B,L,D) - Sequence length added
    x = GaussianDiag.from_tensor(x, 0.1)
    lin = DistLinear(16, 16)
    y = lin(x)                                 # 분포 유지
    cfg = preset_config(Preset.FAST)
    block = DistTransformerBlock(d_model=16, n_heads=1, d_ff=32, resample_every=2)
    z = block(y, cfg)                          # 한 번은 돌아가야 함
    assert z.mean().shape[-1] == 16
    assert z.mean().shape[-2] == 4 # Check sequence length
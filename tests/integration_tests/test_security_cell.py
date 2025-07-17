import torch
import pytest
from torch import nn
from dpnn_lib.security.security_cell import SecurityCell
from dpnn_lib.security.watermark import WatermarkManager
from dpnn_lib.security.crypto import HEManager

class DummyCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def test_security_cell_forward_valid_input():
    base_cell = DummyCell()
    sec_cell = SecurityCell(base_cell, watermark_id="model-v1")
    
    wm = WatermarkManager("model-v1")
    he = HEManager()
    
    x_plain = torch.randn(8, 16, dtype=torch.float32)
    x_enc = he.encrypt(x_plain)
    x_marked = wm.embed(x_enc)
    
    y_marked = sec_cell(x_marked)
    
    assert wm.verify(y_marked), "Output of SecurityCell does not have a valid watermark!"
    
    # For PoC, check if decryption works and shape is preserved
    y_dec = he.decrypt(y_marked)
    assert y_dec.shape == x_plain.shape, "Output shape mismatch after security cell processing!"

def test_security_cell_invalid_watermark_input():
    base_cell = DummyCell()
    sec_cell = SecurityCell(base_cell, watermark_id="model-v1")
    
    wm_invalid = WatermarkManager("invalid-id")
    he = HEManager()
    
    x_plain = torch.randn(8, 16, dtype=torch.float32)
    x_enc = he.encrypt(x_plain)
    x_invalid_marked = wm_invalid.embed(x_enc) # Embed with a different watermark ID
    
    with pytest.raises(AssertionError, match="Invalid watermark!"):
        sec_cell(x_invalid_marked)

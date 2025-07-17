import torch
import pytest
from torch import nn
from dpnn_lib.security.security_cell import SecurityCell
from dpnn_lib.security.watermark import WatermarkManager
from dpnn_lib.security.crypto import HEManager
import tenseal as ts

class DummyCell(nn.Module):
    def __init__(self, input_dim=16, output_dim=16):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

def test_security_cell_forward_valid_input():
    input_shape = (8, 16)
    base_cell = DummyCell(input_dim=input_shape[-1], output_dim=input_shape[-1])
    watermark_id = "model-v1"
    secret_key = "test_secret"
    sec_cell = SecurityCell(base_cell, watermark_id=watermark_id, input_shape=input_shape, secret_key=secret_key)
    
    he = HEManager()
    wm_verifier = WatermarkManager(watermark_id, secret_key=secret_key)
    
    x_plain = torch.randn(input_shape, dtype=torch.float32)
    
    y_encrypted = sec_cell(x_plain)
    
    # Decrypt and verify outside the SecurityCell for testing
    y_decrypted_flat = he.decrypt(y_encrypted)
    y_decrypted_reshaped = y_decrypted_flat.reshape(input_shape)
    
    assert wm_verifier.verify(y_decrypted_reshaped, input_shape, watermark_id), "Output of SecurityCell does not have a valid watermark!"
    
    # Check if decryption works and shape is preserved
    # Use allclose for TenSEAL approximate decryption
    # We can't directly compare y_decrypted_reshaped with x_plain because base_cell transforms it.
    # Instead, we check if the output shape is correct.
    assert y_decrypted_reshaped.shape == input_shape, "Output shape mismatch after security cell processing!"

def test_security_cell_invalid_watermark_input():
    input_shape = (8, 16)
    base_cell = DummyCell(input_dim=input_shape[-1], output_dim=input_shape[-1])
    correct_watermark_id = "model-v1"
    secret_key = "test_secret"
    
    # Initialize SecurityCell with the correct watermark ID
    sec_cell = SecurityCell(base_cell, watermark_id=correct_watermark_id, input_shape=input_shape, secret_key=secret_key)
    
    # Create an input that will be processed by the SecurityCell
    x_plain = torch.randn(input_shape, dtype=torch.float32)
    
    # Process the input through the SecurityCell (it will embed its own watermark)
    y_encrypted = sec_cell(x_plain)
    
    # Decrypt the output
    he = HEManager()
    y_decrypted_flat = he.decrypt(y_encrypted)
    y_decrypted_reshaped = y_decrypted_flat.reshape(input_shape)
    
    # Now, try to verify the output using a WatermarkManager with an *incorrect* ID
    wm_incorrect_verifier = WatermarkManager("invalid-id", secret_key=secret_key)
    
    # This assertion should pass, as the verification with the incorrect ID should fail.
    assert not wm_incorrect_verifier.verify(y_decrypted_reshaped, input_shape, "invalid-id"), \
        "Verification with incorrect ID passed unexpectedly!"

import torch
import pytest
from dpnn_lib.security.crypto import HEManager

def test_he_encrypt_decrypt_consistency():
    he = HEManager()
    original_tensor = torch.randn(10, 10, dtype=torch.float32)
    
    encrypted_tensor = he.encrypt(original_tensor)
    decrypted_tensor = he.decrypt(encrypted_tensor)
    
    # For TenSEAL, decryption is approximate, so use torch.allclose
    assert torch.allclose(original_tensor.flatten(), decrypted_tensor, atol=1e-2), "Encryption/decryption should be consistent within tolerance!"
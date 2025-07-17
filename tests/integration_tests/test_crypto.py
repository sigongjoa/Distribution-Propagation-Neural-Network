import torch
import pytest
from dpnn_lib.security.crypto import HEManager

def test_he_encrypt_decrypt_consistency():
    he = HEManager()
    original_tensor = torch.randn(10, 10, dtype=torch.float32)
    
    encrypted_tensor = he.encrypt(original_tensor)
    decrypted_tensor = he.decrypt(encrypted_tensor)
    
    # For the PoC, encrypt/decrypt are no-ops, so tensors should be equal
    assert torch.equal(original_tensor, decrypted_tensor), "Encryption/decryption should be consistent for PoC!"

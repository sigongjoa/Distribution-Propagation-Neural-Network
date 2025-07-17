import torch
import pytest
from dpnn_lib.security.watermark import WatermarkManager

def test_watermark_embed_verify():
    wm = WatermarkManager("test-id-1")
    original_tensor = torch.randn(100, 100, dtype=torch.float32)
    
    marked_tensor = wm.embed(original_tensor)
    
    assert wm.verify(marked_tensor), "Watermark verification failed after embedding!"

def test_watermark_tampering_detection():
    wm = WatermarkManager("test-id-2")
    original_tensor = torch.randn(100, 100, dtype=torch.float32)
    marked_tensor = wm.embed(original_tensor)
    
    # Tamper with the marked tensor (change a value in the data part)
    tampered_tensor = marked_tensor.clone()
    tampered_tensor[wm.hash_length + 5] += 0.1 # Change a value outside the watermark part
    
    assert not wm.verify(tampered_tensor), "Watermark verification passed on tampered tensor!"

def test_watermark_different_id_detection():
    wm1 = WatermarkManager("test-id-3")
    wm2 = WatermarkManager("different-id")
    
    original_tensor = torch.randn(100, 100, dtype=torch.float32)
    marked_tensor = wm1.embed(original_tensor)
    
    assert not wm2.verify(marked_tensor), "Watermark verification passed with wrong ID!"

def test_watermark_small_tensor_error():
    wm = WatermarkManager("test-id-4")
    small_tensor = torch.randn(10, dtype=torch.float32) # Smaller than hash_length
    
    with pytest.raises(ValueError, match=f"Tensor too small to embed watermark. Requires at least {wm.hash_length} elements."):
        wm.embed(small_tensor)

def test_watermark_small_tensor_verify_fail():
    wm = WatermarkManager("test-id-5")
    small_tensor = torch.randn(10, dtype=torch.float32) # Smaller than hash_length
    
    assert not wm.verify(small_tensor), "Verification passed on a tensor too small to contain a watermark!"

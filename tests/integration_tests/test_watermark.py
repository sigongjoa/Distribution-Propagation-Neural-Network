import torch
import pytest
from dpnn_lib.security.watermark import WatermarkManager

def test_watermark_embed_verify():
    wm = WatermarkManager("test-id-1")
    original_tensor = torch.randn(100, 100, dtype=torch.float32)
    
    marked_tensor = wm.embed(original_tensor)
    
    assert wm.verify(marked_tensor, original_tensor.shape, "test-id-1"), "Watermark verification failed after embedding!"

def test_watermark_tampering_detection():
    wm = WatermarkManager("test-id-2")
    original_tensor = torch.randn(100, 100, dtype=torch.float32)
    marked_tensor = wm.embed(original_tensor)
    
    # Tamper with the marked tensor *within* the watermark pattern area
    tampered_tensor = marked_tensor.clone()
    tampered_tensor.flatten()[0] += 100.0 # Change a value within the watermark pattern
    
    assert not wm.verify(tampered_tensor, original_tensor.shape, "test-id-2"), "Watermark verification passed on tampered tensor!"

def test_watermark_different_id_detection():
    wm1 = WatermarkManager("test-id-3")
    wm2 = WatermarkManager("different-id")
    
    original_tensor = torch.randn(100, 100, dtype=torch.float32)
    marked_tensor = wm1.embed(original_tensor)
    
    assert not wm2.verify(marked_tensor, original_tensor.shape, "different-id"), "Watermark verification passed with wrong ID!"

def test_watermark_small_tensor_error():
    wm = WatermarkManager("test-id-4")
    small_tensor = torch.randn(5, dtype=torch.float32) # Smaller than watermark_pattern_size
    
    with pytest.raises(ValueError, match=f"Tensor too small to embed watermark. Requires at least {wm.watermark_pattern_size} elements."):
        wm.embed(small_tensor)

def test_watermark_small_tensor_verify_fail():
    wm = WatermarkManager("test-id-5")
    # Make the tensor significantly smaller than watermark_pattern_size to trigger early exit in verify
    small_tensor = torch.randn(5, dtype=torch.float32) 
    
    assert not wm.verify(small_tensor, small_tensor.shape, "test-id-5"), "Verification passed on a tensor too small to contain a watermark!"
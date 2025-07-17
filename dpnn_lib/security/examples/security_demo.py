import torch
from dpnn_lib.security.security_cell import SecurityCell
from dpnn_lib.security.watermark import WatermarkManager
from dpnn_lib.security.crypto import HEManager

# 1) 기본 셀을 정의
base = torch.nn.Linear(16,16)

# 2) 보안 셀로 래핑 (ID = "model-v1")
sec_cell = SecurityCell(base, watermark_id="model-v1")

# 3) 초기 워터마크·암호화된 입력 생성
wm = WatermarkManager("model-v1")
he = HEManager()
x_plain = torch.randn(8,16)

# Encrypt the plain tensor first
x_enc = he.encrypt(x_plain)

# Embed the watermark into the encrypted tensor
x_marked = wm.embed(x_enc)

# 4) forward 테스트
y_marked = sec_cell(x_marked)

# Verify the watermark on the output
assert wm.verify(y_marked), "Output watermark verification failed!"

# Decrypt the output
y_dec = he.decrypt(y_marked)

print("SecurityCell PoC OK, output shape:", y_dec.shape)

# Test with invalid watermark (should raise AssertionError)
print("\nTesting with invalid watermark (expected AssertionError)...")
try:
    invalid_wm = WatermarkManager("invalid-id")
    x_invalid_marked = invalid_wm.embed(x_enc) # Embed with a different watermark ID
    sec_cell(x_invalid_marked)
except AssertionError as e:
    print(f"Caught expected error: {e}")
except Exception as e:
    print(f"Caught unexpected error: {e}")

from torch import nn, Tensor
import torch
from .watermark import WatermarkManager
from .crypto import HEManager
import tenseal as ts

class SecurityCell(nn.Module):
    """
    보안 특화 셀: 내부 연산 전에 입력 암호화·워터마크 검증,
    연산 후 복호화·워터마크 재주입을 수행합니다.
    """
    def __init__(self, base_cell: nn.Module, watermark_id: str, input_shape: tuple, secret_key: str = "default_secret_key"):
        super().__init__()
        self.base_cell = base_cell
        self.wm = WatermarkManager(watermark_id, secret_key=secret_key)
        self.he = HEManager()
        self.input_shape = input_shape
        self.watermark_id = watermark_id # Store watermark_id for verification

    def forward(self, x_plain_input: Tensor) -> ts.CKKSVector:
        # 1) 입력 암호화
        # No watermark embedding on input for output-based watermarking
        x_encrypted = self.he.encrypt(x_plain_input)
        
        # 2) 기본 셀 연산 (conceptual: base_cell would operate on encrypted data in a full HE system)
        # For this PoC, we decrypt, run base_cell, then re-encrypt.
        x_decrypted_flat = self.he.decrypt(x_encrypted)
        x_decrypted = x_decrypted_flat.reshape(self.input_shape)
        
        y_plain = self.base_cell(x_decrypted)
        
        # 3) 출력 워터마크 주입 (on plaintext output)
        y_marked_plain = self.wm.embed(y_plain)
        
        # 4) 출력 암호화
        y_plain_flat = y_marked_plain.flatten()
        y_encrypted = self.he.encrypt(y_plain_flat)
        
        # 5) 출력 워터마크 검증 (on decrypted output for PoC)
        # For a real system, this would be more complex or done on encrypted data.
        y_decrypted_flat_for_wm = self.he.decrypt(y_encrypted)
        y_decrypted_for_wm = y_decrypted_flat_for_wm.reshape(self.input_shape) # Assuming output shape is same as input for simplicity
        
        assert self.wm.verify(y_decrypted_for_wm, self.input_shape, self.watermark_id), "Invalid watermark detected in output!"
        
        return y_encrypted

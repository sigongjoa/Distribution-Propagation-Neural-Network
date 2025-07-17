from torch import nn, Tensor
from .watermark import WatermarkManager
from .crypto import HEManager

class SecurityCell(nn.Module):
    """
    보안 특화 셀: 내부 연산 전에 입력 암호화·워터마크 검증,
    연산 후 복호화·워터마크 재주입을 수행합니다.
    """
    def __init__(self, base_cell: nn.Module, watermark_id: str):
        super().__init__()
        self.base_cell = base_cell
        self.wm = WatermarkManager(watermark_id)
        self.he = HEManager()

    def forward(self, x: Tensor) -> Tensor:
        # 1) 입력 워터마크 검증
        assert self.wm.verify(x), "Invalid watermark!"
        
        # 2) 입력 암호 해독
        # The watermark is now embedded by overwriting, so x is the full tensor.
        # The decryption should happen on the entire tensor.
        x_plain = self.he.decrypt(x) # This will be a placeholder decryption
        
        # 3) 기본 셀 연산
        # x_plain should now have the correct shape for base_cell
        y_plain = self.base_cell(x_plain)
        
        # 4) 출력 암호화
        y_enc = self.he.encrypt(y_plain)
        
        # 5) 워터마크 주입
        y_marked = self.wm.embed(y_enc)
        return y_marked

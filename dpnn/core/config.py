from enum import Enum
from dataclasses import dataclass

class DistMode(Enum):
    MOMENT = "moment"
    DELTA  = "delta"
    MC     = "mc"
    UKF    = "ukf"
    AUTO   = "auto"

@dataclass
class DistConfig:
    mode: DistMode = DistMode.AUTO
    mc_k_min: int = 0         # AUTO에서 사용할 최소 샘플 수
    mc_k_max: int = 4
    use_sigma_points: bool = True
    entropy_threshold: float = 0.8  # 엔트로피↑ 구간에서 정확도↑ 모드로
    curvature_threshold: float = 0.5 # 곡률↑ (델타 vs moment 결정)
    resample_every: int = 2    # 블록마다 주기적 재샘플링
    tau_init: float = 1.0      # softmax 온도 초기값
    clamp_eps: float = 1e-6
    variance_reg: float = 0.0  # 분산 규제 계수

class Preset(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    ACCURATE = "accurate"

def preset_config(p: Preset) -> DistConfig:
    if p is Preset.FAST:
        return DistConfig(mode=DistMode.MOMENT, mc_k_min=0, mc_k_max=0, use_sigma_points=False,
                          resample_every=4)
    if p is Preset.BALANCED:
        return DistConfig(mode=DistMode.AUTO, mc_k_min=0, mc_k_max=2, use_sigma_points=True,
                          resample_every=2)
    return DistConfig(mode=DistMode.AUTO, mc_k_min=2, mc_k_max=6, use_sigma_points=True,
                      resample_every=1, curvature_threshold=0.3)

# Lie Group 기반 DPNN 정보 단위 프레임워크

## 1. 개요

- **Lie Group**: 연속적인 군 구조(예: SO(3), SE(3), U(1))
- DPNN의 핵심: 모든 정보 단위를 대수구조 원소로 보고, 셀(Cell)을 통해 해당 구조 위 연산을 수행
- Lie Group도 DistributionCell, GraphCell처럼 `LieGroupCell`로 구현 가능

---

## 2. LieGroupCell 설계

```python
class LieGroupCell(AlgebraicCell):
    def __init__(self, algebra_op: Callable):
        super().__init__()
        self.algebra_op = algebra_op  # 리 대수 수준 연산 함수

    def forward(self, g: LieGroupElement) -> LieGroupElement:
        # 1) log_map: 군(G) → 리 대수(g)
        u = log_map(g)                     
        # 2) 대수 연산: 리 대수 수준에서 선형/비선형 변환
        v = self.algebra_op(u)            
        # 3) exp_map: 리 대수(g) → 군(G)
        h = exp_map(v)                    
        return h  # 최종 군원소
```

- `log_map`과 `exp_map`으로 G ↔ g 사이 변환
- `algebra_op`은 선형 변환, attention, normalization 등 자유롭게 정의

---

## 3. DPNN 통합 구조

```
dpnn_lib/
├── lie_group/                  # Lie Group 대수 연산 모듈
│   ├── __init__.py
│   ├── maps.py                # log_map, exp_map 구현
│   └── ops.py                 # 리 대수 내 연산 (e.g., 선형, 편미분)
├── distributions/              # 기존 분포 모듈
├── graph_ops/                  # 기존 그래프 모듈
├── models/                     # 모델 레이어 정의
│   ├── transformer/
│   └── diffusion/
└── cells/                      # 셀 인터페이스 모음
    ├── AlgebraicCell.py        # 추상 클래스
    ├── DistributionCell.py
    ├── GraphCell.py
    └── LieGroupCell.py         # 새로 추가된 Lie Group 셀
```

- `cells/LieGroupCell.py`에 `LieGroupCell` 정의
- `lie_group/` 모듈에서 지도 함수를 제공하여 셀에서 import

---

## 4. 활용 예시

- **로봇 자세 추정**: SE(3) 군원소 형태로 카메라 포즈 입력 → `LieGroupCell`로 동작 예측
- **3D 물체 추적**: SO(3)회전 행렬 흐름 모델링 → TimeSeriesCell 대신 `LieGroupCell` 사용
- **자율 주행**: 차량 위치·방향 제어를 SE(2) 지배 식으로 모델링

---

## 5. 장점

1. **기하학적 제약 준수**: 출력이 항상 군원소로 유지되어 물리 법칙 보장
2. **통합 인터페이스**: Transformer, Diffusion 어디에나 `LieGroupCell` 삽입 가능
3. **추상화 레벨**: 복잡한 다양체 최적화도 DPNN 프레임워크 내에서 일관된 방식으로 다룸
4. **확장성**: 다른 군(유니터리, 심플렉틱 등)도 동일 패턴 적용


# DPNN 라이브러리 설계서

## 1. 개요

DPNN(Distribution Propagation Neural Network) 라이브러리는 분포 기반 셀 상태 전파 아키텍처를 손쉽게 구현하기 위한 파이썬 프레임워크입니다. PyTorch 스타일 API를 참조하되, 셀 상태를 분포로 정의하고 전파 연산을 수행하는 데 중점을 둡니다.

## 2. 주요 구성 요소

### 2.1 Distributions 모듈

- `BaseDistribution` (추상 클래스)

  - `params`: 분포 파라미터(dict)
  - `sample(self)`: 샘플링 반환
  - `log_prob(self, x)`: 로그 우도 계산
  - `combine(self, other)`: 분포 합성 연산

- `GaussianDistribution(BaseDistribution)`

  - 평균(mu), 분산(var)
  - `combine`: 평균 & 분산 덧셈

- `PoissonDistribution(BaseDistribution)`

  - 람다(lambda)
  - `combine`: λ 평균화 방식

- `DirichletDistribution(BaseDistribution)`

  - alpha 벡터
  - `combine`: 파라미터 덧셈

### 2.2 Cells 모듈

- `DistributionCell`
  - 속성: `distribution: BaseDistribution`
  - 메서드:
    - `propagate(neighbors: List[DistributionCell])`: 분포 전파
    - `reconstruct()`: 활성화—분포 재구성

### 2.3 Layers 모듈

- `DistributionLayer`

  - 속성: `cells: List[DistributionCell]`
  - 메서드:
    - `forward()`: 모든 셀에 propagate 호출

- `DistributionNetwork`

  - 속성: `layers: List[DistributionLayer]`
  - 메서드:
    - `forward(steps: int)`: step 단위 반복 전파
    - `get_output()`: 최종 셀 분포 집합 반환

### 2.4 Optimization 모듈

- `DistributionOptimizer`
  - `loss_fn(output_distributions, target_distributions)`: KL 발산 등
  - `step()`: 파라미터 업데이트

### 2.5 Utilities

- `Sampler`
  - 분포별 샘플링 도우미
- `Metrics`
  - 분포 간 거리 (KL, JS)
  - 분포 시각화 함수

## 3. 디렉토리 구조

```
dpnn_lib/
├── distributions/
│   ├── __init__.py
│   ├── base.py
│   ├── gaussian.py
│   ├── poisson.py
│   └── dirichlet.py
├── cells/
│   ├── __init__.py
│   └── cell.py
├── layers/
│   ├── __init__.py
│   ├── layer.py
│   └── network.py
├── optim/
│   ├── __init__.py
│   └── optimizer.py
├── utils/
│   ├── __init__.py
│   └── metrics.py
└── examples/
    └── poisson_demo.py
```

## 4. 사용 예시

```python
from dpnn_lib import GaussianDistribution, DistributionCell, DistributionLayer, DistributionNetwork, DistributionOptimizer

# 1) 분포 생성
init_dist = GaussianDistribution(mu=0.0, var=1.0)
cell = DistributionCell(distribution=init_dist)

# 2) 레이어/네트워크 구성
layer = DistributionLayer(cells=[cell for _ in range(10)])
net = DistributionNetwork(layers=[layer for _ in range(5)])

# 3) 전파
net.forward(steps=100)
output = net.get_output()

# 4) 최적화
optimizer = DistributionOptimizer(net)
loss = optimizer.loss_fn(output, target)
optimizer.step()
```

## 5. 향후 확장

- CUDA 지원 가속
- 커스텀 분포 확장 가이드
- PyTorch 통합 모듈 (torch.autograd 호환)
- 분포 디퓨전 네트워크 예시 추가

---

*이 설계서는 DPNN 라이브러리 개발의 청사진입니다.*


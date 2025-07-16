
from math import sqrt
from dpnn_lib.distributions.gaussian import GaussianDistribution

def forward_step(dist_prev: GaussianDistribution, beta_t: float):
    alpha_t = 1 - beta_t
    # 1) 분포 스케일링: μ_t = sqrt(alpha)*μ_{t-1}
    mu_t = sqrt(alpha_t) * dist_prev.mu
    # 2) 분산 업데이트: var_t = alpha*var_{t-1} + beta
    var_t = alpha_t * dist_prev.var + beta_t
    return GaussianDistribution(mu=mu_t, var=var_t)

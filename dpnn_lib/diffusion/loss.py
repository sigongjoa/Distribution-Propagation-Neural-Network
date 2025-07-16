
import torch
import torch.nn.functional as F

def loss_fn(pred_dist, true_dist, var_weight=1.0):
    loss_mu  = F.mse_loss(pred_dist.mu, true_dist.mu)
    loss_var = F.mse_loss(pred_dist.var, true_dist.var)
    return loss_mu + loss_var * var_weight

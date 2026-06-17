import torch
import torch.nn.functional as F


def margin_mse(s_pos, s_neg, t_pos, t_neg):
    return F.mse_loss(s_pos - s_neg, t_pos - t_neg)


def kl_distill(s_pos, s_neg, t_pos, t_neg):
    student = torch.stack([s_pos, s_neg], dim=-1)
    teacher = torch.stack([t_pos, t_neg], dim=-1)
    log_p = F.log_softmax(student, dim=-1)
    q = F.softmax(teacher, dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean")


def distill_loss(s_pos, s_neg, t_pos, t_neg, kl_weight=0.0):
    loss = margin_mse(s_pos, s_neg, t_pos, t_neg)
    if kl_weight > 0:
        loss = loss + kl_weight * kl_distill(s_pos, s_neg, t_pos, t_neg)
    return loss

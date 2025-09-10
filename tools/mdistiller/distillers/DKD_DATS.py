import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask_gt, mask_other):
    t1 = (t * mask_gt).sum(dim=1, keepdims=True)
    t2 = (t * mask_other).sum(dim=1, keepdims=True)
    return torch.cat([t1, t2], dim=1)


def dkd_nckd_dats_loss(
    logits_student,
    logits_teacher,
    target,
    alpha,             # TCKD loss 的系数
    beta,              # NCKD loss 的系数
    T_fixed,           # 用于 TCKD 的固定温度
    T_base,            # 给 NCKD 做自适应温度时的基础温度
    beta_temp,         # NCKD 温度调节强度
    eps=1e-6
):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    pred_student_t = F.softmax(logits_student / T_fixed, dim=1)
    pred_teacher_t = F.softmax(logits_teacher / T_fixed, dim=1)

    pred_student_t = cat_mask(pred_student_t, gt_mask, other_mask)
    pred_teacher_t = cat_mask(pred_teacher_t, gt_mask, other_mask)
    log_pred_student_t = torch.log(pred_student_t + eps)

    kl_tckd = F.kl_div(log_pred_student_t, pred_teacher_t, reduction="sum")
    kl_tckd *= (T_fixed ** 2)   
    tckd_loss = kl_tckd / target.size(0)
    logits_teacher_nckd = logits_teacher - 1000.0 * gt_mask
    teacher_probs_nckd = F.softmax(logits_teacher_nckd, dim=1) 

    teacher_log_probs_nckd = torch.log(teacher_probs_nckd + eps)
    teacher_entropy_nckd = -torch.sum(teacher_probs_nckd * teacher_log_probs_nckd, dim=1)  # (B,)

    H_min = teacher_entropy_nckd.min()
    H_max = teacher_entropy_nckd.max()
    if H_max - H_min < eps:
        normalized_entropy = torch.zeros_like(teacher_entropy_nckd)
    else:
        normalized_entropy = (teacher_entropy_nckd - H_min) / (H_max - H_min + eps)

    T_n = T_base * (1.0 + beta_temp * normalized_entropy)  # (B,)
    T_n_expanded = T_n.unsqueeze(1)

    logits_student_nckd = logits_student - 1000.0 * gt_mask

    log_pred_student_nckd = F.log_softmax(logits_student_nckd / T_n_expanded, dim=1)
    pred_teacher_nckd = F.softmax(logits_teacher_nckd / T_n_expanded, dim=1)

    kl_nckd = F.kl_div(log_pred_student_nckd, pred_teacher_nckd, reduction="none").sum(dim=1)
    kl_nckd = kl_nckd * (T_n ** 2)
    nckd_loss = kl_nckd.mean()

    total_loss = alpha * tckd_loss + beta * nckd_loss
    return total_loss


class DKD_DATS(Distiller):
    def __init__(self, student, teacher, cfg):
        super(DKD_DATS, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD_DATS.CE_WEIGHT
        self.alpha = cfg.DKD_DATS.ALPHA
        self.beta = cfg.DKD_DATS.BETA
        self.T_fixed = cfg.DKD_DATS.T
        self.T_base = cfg.DKD_DATS.T_BASE
        self.beta_temp = cfg.DKD_DATS.BETA_TEMP
        self.warmup = cfg.DKD_DATS.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        current_epoch = kwargs.get("epoch", 1)
        factor = min(float(current_epoch) / self.warmup, 1.0)

        loss_dkd = factor * dkd_nckd_dats_loss(
            logits_student,
            logits_teacher,
            target,
            alpha=self.alpha,
            beta=self.beta,
            T_fixed=self.T_fixed,
            T_base=self.T_base,
            beta_temp=self.beta_temp,
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict

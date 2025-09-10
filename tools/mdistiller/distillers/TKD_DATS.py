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


def cat_mask(t, mask_gt, mask_top, mask_other):
    t1 = (t * mask_gt).sum(dim=1, keepdims=True)
    t2 = (t * mask_top).sum(dim=1, keepdims=True)
    t3 = (t * mask_other).sum(dim=1, keepdims=True)
    return torch.cat([t1, t2, t3], dim=1)


def _get_mask(pred, conf):
    K_values = []
    processed_top_K_logits = []
    processed_except_K_logits = []

    for row in pred:
        top_values, top_indices = torch.topk(row, k=len(row), sorted=True)
        cumsum_logit = torch.cumsum(top_values, dim=0)
        K = (cumsum_logit >= conf).nonzero(as_tuple=True)[0].min().item() + 1
        K_values.append(K)

        mask_top_k = torch.zeros_like(row, dtype=torch.bool)
        mask_top_k[top_indices[:K]] = True
        mask_other = ~mask_top_k

        top_k_logit = row * mask_top_k
        other_logit = row * mask_other

        processed_top_K_logits.append(top_k_logit)
        processed_except_K_logits.append(other_logit)

    processed_top_K_logits = torch.stack(processed_top_K_logits)
    processed_except_K_logits = torch.stack(processed_except_K_logits)
    return processed_top_K_logits, processed_except_K_logits, K_values


def tkd_sckd_dckd_dats_loss(
    logits_student,
    logits_teacher,
    target,
    conf,
    T_sckd_base,
    beta_sckd,
    T_dckd_base,
    beta_dckd,
    a, b, c,
    eps=1e-6
):
    B = logits_teacher.size(0)

    gt_mask = _get_gt_mask(logits_student, target)      # (B,K)
    other_mask = _get_other_mask(logits_student, target)
    with torch.no_grad():
        pred_student_for_topk = F.softmax(logits_student, dim=1)  # (B,K)
    pred_top_k_student, pred_other_student, _ = _get_mask(pred_student_for_topk, conf)
    logits_teacher_top = logits_teacher.clone()
    logits_teacher_other = logits_teacher.clone()
    logits_teacher_top[gt_mask] = -1e9
    logits_teacher_other[gt_mask] = -1e9
    logits_teacher_top[pred_top_k_student == 0] = -1e9
    logits_teacher_other[pred_other_student == 0] = -1e9
    
    mask_nonzero_top = (logits_teacher_top != -1e9)
    mask_top = ~mask_nonzero_top
    mask_nonzero_other = (logits_teacher_other != -1e9)
    mask_other = ~mask_nonzero_other

    teacher_probs_full = F.softmax(logits_teacher, dim=1)
    teacher_pred_sckd = cat_mask(teacher_probs_full, gt_mask, mask_top, mask_other)  # (B,3)

    teacher_log_sckd = torch.log(teacher_pred_sckd + eps)
    #H_sckd = -torch.sum(teacher_pred_sckd * teacher_log_sckd, dim=1)  # (B,)
    # 修改 熵的公式为不确定度
    # 计算均值 mu
    mu = teacher_pred_sckd.mean(dim=1, keepdim=True)  # 形状 (B, 1)
    # 计算标准差 H_sckd
    H_sckd = torch.sqrt(
        torch.mean((teacher_pred_sckd - mu) ** 2, dim=1)  # 形状 (B,)
    )

    H_sckd_min, H_sckd_max = H_sckd.min(), H_sckd.max()
    if H_sckd_max - H_sckd_min < eps:
        alpha_sckd = torch.zeros_like(H_sckd)
    else:
        alpha_sckd = (H_sckd - H_sckd_min) / (H_sckd_max - H_sckd_min + eps)

    T_n_sckd = T_sckd_base * (1.0 + beta_sckd * alpha_sckd)  # (B,)
    T_n_sckd_exp = T_n_sckd.unsqueeze(1)

    pred_student_all_sckd = F.softmax(logits_student / T_n_sckd_exp, dim=1)  # (B, K)
    pred_teacher_all_sckd = F.softmax(logits_teacher / T_n_sckd_exp, dim=1)
    pred_student_ternary = cat_mask(pred_student_all_sckd, gt_mask, mask_top, mask_other)
    pred_teacher_ternary = cat_mask(pred_teacher_all_sckd, gt_mask, mask_top, mask_other)

    log_pred_student_ternary = torch.log(pred_student_ternary + eps)
    kl_ternary = F.kl_div(log_pred_student_ternary, pred_teacher_ternary, reduction='none').sum(dim=1)
    kl_ternary *= T_n_sckd**2
    ternary_loss = kl_ternary.mean()

    logits_teacher_dckd = logits_teacher.clone()
    logits_teacher_dckd[gt_mask] = -1e9
    mask_topk_or_other = (pred_top_k_student != 0) | (pred_other_student != 0)
    logits_teacher_dckd[~mask_topk_or_other] = -1e9

    teacher_probs_dckd = F.softmax(logits_teacher_dckd, dim=1)       # (B,K)
    teacher_log_probs_dckd = torch.log(teacher_probs_dckd + eps)
    #H_dckd = -torch.sum(teacher_probs_dckd * teacher_log_probs_dckd, dim=1)  # (B,)
    # 计算均值 mu
    mu = teacher_probs_dckd.mean(dim=1, keepdim=True)  # 形状 (B, 1)
    # 计算标准差 H_sckd
    H_dckd = torch.sqrt(
        torch.mean((teacher_probs_dckd - mu) ** 2, dim=1)  # 形状 (B,)
    )

    H_dckd_min, H_dckd_max = H_dckd.min(), H_dckd.max()
    if H_dckd_max - H_dckd_min < eps:
        alpha_dckd = torch.zeros_like(H_dckd)
    else:
        alpha_dckd = (H_dckd - H_dckd_min) / (H_dckd_max - H_dckd_min + eps)

    T_n_dckd = T_dckd_base * (1.0 + beta_dckd * alpha_dckd)  # (B,)
    T_n_dckd_exp = T_n_dckd.unsqueeze(1)

    logits_student_top = logits_student.clone()
    logits_student_top[gt_mask] = -1e9
    logits_student_top[pred_top_k_student == 0] = -1e9

    logits_teacher_top = logits_teacher.clone()
    logits_teacher_top[gt_mask] = -1e9
    logits_teacher_top[pred_top_k_student == 0] = -1e9

    pred_student_top = F.softmax(logits_student_top / T_n_dckd_exp, dim=1)
    pred_teacher_top = F.softmax(logits_teacher_top / T_n_dckd_exp, dim=1)
    log_pred_student_top = torch.log(pred_student_top + eps)

    kl_topk = F.kl_div(log_pred_student_top, pred_teacher_top, reduction='none').sum(dim=1)
    kl_topk *= T_n_dckd**2
    topk_loss = kl_topk.mean()
    
    logits_student_other = logits_student.clone()
    logits_student_other[gt_mask] = -1e9
    logits_student_other[pred_other_student == 0] = -1e9

    logits_teacher_other = logits_teacher.clone()
    logits_teacher_other[gt_mask] = -1e9
    logits_teacher_other[pred_other_student == 0] = -1e9

    pred_student_other = F.softmax(logits_student_other / T_n_dckd_exp, dim=1)
    pred_teacher_other = F.softmax(logits_teacher_other / T_n_dckd_exp, dim=1)
    log_pred_student_other = torch.log(pred_student_other + eps)

    kl_other = F.kl_div(log_pred_student_other, pred_teacher_other, reduction='none').sum(dim=1)
    kl_other *= T_n_dckd**2
    other_loss = kl_other.mean()

    return ternary_loss, topk_loss, other_loss


class TKD_DATS(Distiller):

    def __init__(self, student, teacher, cfg):
        super(TKD_DATS, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.TKD_DATS.CE_WEIGHT

        self.warmup = cfg.TKD_DATS.WARMUP

        self.conf = cfg.TKD_DATS.CONF

        self.T_sckd_base = cfg.TKD_DATS.T_SCKD_BASE
        self.beta_sckd = cfg.TKD_DATS.BETA_SCKD

        self.T_dckd_base = cfg.TKD_DATS.T_DCKD_BASE
        self.beta_dckd = cfg.TKD_DATS.BETA_DCKD

        self.a = cfg.TKD_DATS.A
        self.b = cfg.TKD_DATS.B
        self.c = cfg.TKD_DATS.C

        self.n = cfg.TKD_DATS.N
        self.min_ = cfg.TKD_DATS.MIN
        self.epochs = cfg.SOLVER.EPOCHS

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        ternary_loss, topk_loss, other_loss = tkd_sckd_dckd_dats_loss(
            logits_student,
            logits_teacher,
            target,
            conf=self.conf,
            T_sckd_base=self.T_sckd_base,
            beta_sckd=self.beta_sckd,
            T_dckd_base=self.T_dckd_base,
            beta_dckd=self.beta_dckd,
            a=self.a,
            b=self.b,
            c=self.c,
        )

        current_epoch = kwargs.get("epoch", 1)
        factor = min(float(current_epoch) / self.warmup, 1.0)

        scale_topk = min((self.epochs - current_epoch) / (self.epochs - self.n) + self.min_, 1.0)
        scale_other = min(current_epoch / self.n + self.min_, 1.0)

        loss_tkd = factor * (
            self.a * ternary_loss
            + scale_topk * self.b * topk_loss
            + scale_other * self.c * other_loss
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_tkd,
        }
        return logits_student, losses_dict

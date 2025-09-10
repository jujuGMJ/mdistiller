import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import numpy as np

from ._base import Distiller

# 研究点1
# 研究点2 通过不确定性指标来判断学生模型学习的困难样本比例
def tkd_loss(logits_student, logits_teacher, target, t, T, conf, T_a, T_b):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / t, dim=1)
    pred_top_k_student, pred_other_student, K_student = _get_mask(pred_student, conf)

    logits_student_top = logits_student.clone()
    logits_student_top[pred_top_k_student == 0] = 0
    logits_student_top[other_mask == 0] = 0
    logits_teacher_top = logits_teacher.clone()
    logits_teacher_top[pred_top_k_student == 0] = 0
    logits_teacher_top[other_mask == 0] = 0

    logits_student_other = logits_student.clone()
    logits_student_other[pred_other_student == 0] = 0
    logits_student_other[other_mask == 0] = 0
    logits_teacher_other = logits_teacher.clone()
    logits_teacher_other[pred_other_student == 0] = 0
    logits_teacher_other[other_mask == 0] = 0

    # ternary_loss
    mask_nonzero = torch.zeros_like(logits_teacher_top, dtype=torch.bool)
    mask_nonzero[logits_teacher_top != 0] = True
    mask_top = ~mask_nonzero
    mask_nonzero = torch.zeros_like(logits_teacher_other, dtype=torch.bool)
    mask_nonzero[logits_teacher_other != 0] = True
    mask_other = ~mask_nonzero
    pred_student = F.softmax(logits_student / T, dim=1)
    pred_teacher = F.softmax(logits_teacher / T, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, mask_top, mask_other)
    pred_teacher = cat_mask(pred_teacher, gt_mask, mask_top, mask_other)
    log_pred_student = torch.log(pred_student)
    ternary_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='sum')
            * (T ** 2)
            / target.shape[0]
    )

    # topk_loss
    pred_student = F.softmax(logits_student_top / T_a, dim=1)
    pred_teacher = F.softmax(logits_teacher_top / T_a, dim=1)
    log_pred_student = torch.log(pred_student)
    top_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='sum')
            * (T_a ** 2)
            / target.shape[0]
    )

    # other_loss
    pred_student = F.softmax(logits_student_other / T_b, dim=1)
    pred_teacher = F.softmax(logits_teacher_other / T_b, dim=1)
    log_pred_student = torch.log(pred_student)
    other_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='sum')
            * (T_b ** 2)
            / target.shape[0]
    )

    return ternary_loss, top_loss, other_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2, mask3):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    t3 = (t * mask3).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2, t3], dim=1)
    return rt


def _get_mask(pred, conf):
    # 用于存储每一行的K值
    K_values = []

    # 用于存储处理后的张量
    processed_top_K_logits = []
    processed_except_K_logits = []

    for i, logit in enumerate(pred):
        # 使用 topk 替代排序
        top_values, top_indices = torch.topk(logit, k=len(logit), sorted=True)
        cumsum_logit = torch.cumsum(top_values, dim=0)
        K = (cumsum_logit >= conf).nonzero(as_tuple=True)[0].min().item() + 1
        K_values.append(K)

        # 创建掩码
        mask_top_k = torch.zeros_like(logit, dtype=torch.bool)
        mask_top_k[top_indices[:K]] = True
        mask_other = ~mask_top_k

        # 应用掩码
        top_k_logit = logit * mask_top_k
        other_logit = logit * mask_other

        processed_top_K_logits.append(top_k_logit)
        processed_except_K_logits.append(other_logit)

    # 处理后的 logits
    processed_top_K_logits = torch.stack(processed_top_K_logits)
    processed_except_K_logits = torch.stack(processed_except_K_logits)

    return processed_top_K_logits, processed_except_K_logits, K_values


class TKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(TKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.TKD.CE_WEIGHT
        self.warmup = cfg.TKD.WARMUP
        self.T_s = cfg.TKD.T_s
        self.T = cfg.TKD.T
        self.T_a = cfg.TKD.T_A
        self.T_b = cfg.TKD.T_B
        self.conf = cfg.TKD.CONF
        self.a = cfg.TKD.A
        self.b = cfg.TKD.B
        self.c = cfg.TKD.C
        self.n = cfg.TKD.N
        self.min = cfg.TKD.MIN
        self.epochs = cfg.SOLVER.EPOCHS

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        ternary_loss, topk_loss, other_loss = tkd_loss(logits_student, logits_teacher, target, self.T_s, self.T, self.conf, self.T_a, self.T_b)

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)

        loss_tkd = min(kwargs["epoch"] / self.warmup, 1.0) * (self.a * ternary_loss
                                                    + min((self.epochs - kwargs["epoch"]) / (self.epochs - self.n) + self.min,
                                                          1.0) * self.b * topk_loss
                                                    + min(kwargs["epoch"] / self.n + self.min, 1.0) * self.c * other_loss)

        #loss_tkd = min(kwargs["epoch"] / self.warmup, 1.0) * (self.a * ternary_loss
                                                              #+ self.b * topk_loss
                                                              #+ self.c * other_loss)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_tkd,
        }
        return logits_student, losses_dict


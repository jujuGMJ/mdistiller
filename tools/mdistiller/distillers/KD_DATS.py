import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def dats_kd_loss(logits_student, logits_teacher, t_base, beta, lambda_coef, eps=1e-6):
    """
    基于教师输出分布熵和最大-次大差值共同计算自适应温度，
    并利用自适应温度计算 KD 损失。

    参数:
        logits_student: 学生模型输出，形状 (B, K)
        logits_teacher: 教师模型输出，形状 (B, K)
        t_base: 全局基础温度 (标量)
        beta: 温度调节强度 (标量)
        lambda_coef: 用于平衡熵和差值的权重，取值 [0,1]
        eps: 防止除 0 的小常数
    返回:
        loss_kd: 标量，KD 损失
    """
    batch_size = logits_teacher.size(0)

    # 1. 计算教师的 softmax 概率和熵
    teacher_probs = F.softmax(logits_teacher, dim=1)
    teacher_log_probs = torch.log(teacher_probs + eps)
    teacher_entropy = -torch.sum(teacher_probs * teacher_log_probs, dim=1)  # 形状 (B,)
    print("熵的值为：", teacher_entropy)

    # 对熵归一化: E_n = (H - H_min) / (H_max - H_min)
    H_min = teacher_entropy.min()
    H_max = teacher_entropy.max()
    E_n = (teacher_entropy - H_min) / (H_max - H_min + eps)

    # 2. 计算 logits 最大-次大差值： m_n = top1 - top2
    top2_values, _ = torch.topk(logits_teacher, k=2, dim=1)
    m = top2_values[:, 0] - top2_values[:, 1]  # 形状 (B,)
    # 归一化处理: 当差值越小，则表示教师信心越不足，取 1 - normalized(m)
    m_min = m.min()
    m_max = m.max()
    M_n = 1 - ((m - m_min) / (m_max - m_min + eps))

    # 3. 计算综合难度系数:  alpha_n = lambda * E_n + (1-lambda) * M_n
    alpha = lambda_coef * E_n + (1 - lambda_coef) * M_n
    # 保证 alpha 在 [0,1]
    alpha = alpha.clamp(0, 1)

    # 4. 根据 alpha 计算每个样本的自适应温度 T_n
    # T_n = t_base * (1 + beta * alpha)
    T_n = t_base * (1 + beta * alpha)  # 形状 (B,)

    # 5. 对学生和教师 logits 分别用对应的温度计算 softmax
    # 注意需要将 T_n 展开为 (B,1) 后再广播
    T_n_expanded = T_n.view(-1, 1)
    log_pred_student = F.log_softmax(logits_student / T_n_expanded, dim=1)
    pred_teacher = F.softmax(logits_teacher / T_n_expanded, dim=1)

    # 6. 计算 KL 散度（这里 per-sample KL 损失相加后取均值）
    loss_per_sample = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1)
    # 乘上温度平方，注意 T_n 是每个样本的温度
    loss_per_sample = loss_per_sample * (T_n ** 2)
    loss_kd = loss_per_sample.mean()

    return loss_kd


class KD_DATS(Distiller):
    """使用自适应温度进行知识蒸馏，
    其中温度由教师输出分布的熵和 logits 最大-次大差值共同决定。"""

    def __init__(self, student, teacher, cfg):
        super(KD_DATS, self).__init__(student, teacher)
        # 基础温度、温度调节强度 beta 和平衡熵与差值的 lambda_coef 从配置中读取
        self.t_base = cfg.KD.T_BASE  # 例如 1 或 2
        self.beta = cfg.KD.BETA  # 温度调节强度，比如 1.0 或其他
        self.lambda_coef = cfg.KD.LAMBDA  # 平衡系数，取值范围 [0,1]
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        # 获取学生与教师 logits
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # 计算 CE 损失
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        # 计算自适应 KD 损失
        loss_kd = self.kd_loss_weight * dats_kd_loss(
            logits_student, logits_teacher, self.t_base, self.beta, self.lambda_coef
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict

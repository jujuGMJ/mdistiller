import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lingam import DirectLiNGAM
from sklearn.preprocessing import StandardScaler
from kornia.filters import canny
from scipy.stats import entropy

from ._base import Distiller
from .DKD import dkd_loss as dkd_loss_origin




def center_loss(features, labels, centers, num_classes, feat_dim, use_gpu=True):
    """
        函数式实现的中心损失

        参数:
            features: 特征矩阵，形状为 (batch_size, feat_dim)
            labels: 真实标签，形状为 (batch_size)
            centers: 中心点矩阵，形状为 (num_classes, feat_dim)
            num_classes: 类别数量
            feat_dim: 特征维度
            use_gpu: 是否使用GPU
        """
    batch_size = features.size(0)
    device = features.device  # 获取特征所在的设备
    # 确保所有张量在相同设备上
    labels = labels.to(device)
    centers = centers.to(device)

    # 计算距离矩阵
    distmat = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(batch_size, num_classes) + \
              torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(num_classes, batch_size).t()
    distmat.addmm_(1, -2, features, centers.t())

    # 创建类别掩码
    classes = torch.arange(num_classes, device=device).long()  # 直接生成在目标设备上
    labels = labels.unsqueeze(1).expand(batch_size, num_classes)
    mask = labels.eq(classes.expand(batch_size, num_classes))

    # 计算损失
    dist = distmat * mask.float()
    loss_center = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

    return loss_center

# 生成因果掩码
def compute_causal_mask(teacher_feats, target):
    # 转换为numpy数组
    if isinstance(teacher_feats, torch.Tensor):
        teacher_feats = teacher_feats.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # 标准化特征（LiNGAM对数据尺度敏感）
    scaler = StandardScaler()
    normalized_feats = scaler.fit_transform(teacher_feats)  # [N, D]

    # 合并特征和标签（最后一列为target）
    data = np.column_stack([normalized_feats, target])  # [N, D+1]

    # 使用LiNGAM进行因果发现
    model = DirectLiNGAM()
    model.fit(data)

    # 获取因果图邻接矩阵 [D+1, D+1]
    adj_matrix = model.adjacency_matrix_

    # 提取target（最后一列）对各特征的因果边
    causal_mask = (adj_matrix[-1, :-1] != 0).astype(np.float32)  # [D]

    print("Entering compute_causal_mask")  # 调试用

    # 转换为torch张量
    return torch.from_numpy(causal_mask)

# 因果损失
def loss_causal(teacher_feats, student_feats, causal_mask, temperature=1.0):
    # 确保mask在相同设备上
    causal_mask = causal_mask.to(teacher_feats.device)

    # 归一化特征（可选）
    teacher_feats = torch.nn.functional.normalize(teacher_feats, p=2, dim=1)
    student_feats = torch.nn.functional.normalize(student_feats, p=2, dim=1)

    # 应用因果掩码（只对齐因果相关维度）
    masked_teacher = teacher_feats * causal_mask.unsqueeze(0)  # [B, D]
    masked_student = student_feats * causal_mask.unsqueeze(0)  # [B, D]

    # 计算温度调整后的相似度损失
    teacher_proj = masked_teacher / temperature  # [B, D]
    student_proj = masked_student / temperature  # [B, D]

    # 使用余弦相似度或MSE损失
    causal_loss = 1 - torch.cosine_similarity(teacher_proj, student_proj, dim=1).mean()
    # 或者：loss = torch.nn.MSELoss()(teacher_proj, student_proj)

    print("Entering compute_causal_mask")  # 调试用

    return causal_loss



# 融合性蒸馏
# 拿到边缘掩码
@torch.no_grad()
def get_edge_mask(im, thr_low=0.1, thr_high=0.3):
    """
    im: [B,3,H,W] 0~1
    return: [B,1,H,W] 边缘掩码 0/1
    """
    gray = im.mean(1, keepdim=True)                # B,1,H,W
    edges = canny(gray, low_threshold=thr_low, high_threshold=thr_high)[1]  # B,1,H,W
    return edges

# 将掩码映射到patch
def mask_to_patch(mask, patch_size=8):
    """
    mask: [B,1,H,W]  0/1
    return: [B, N]  每个 patch 是否属于边缘，1/0
    """
    B, _, H, W = mask.shape
    pooled = F.avg_pool2d(mask, kernel_size=patch_size, stride=patch_size)  # B,1,h,w
    return (pooled > 0.5).flatten(1)          # B, N

# 解耦目标类和非目标类的知识蒸馏
def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='none')
            * (temperature ** 2)
            / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='none')
            * (temperature ** 2)
            / target.shape[0]
    )

    tckd_loss = torch.sum(tckd_loss, dim=1)
    nckd_loss = torch.sum(nckd_loss, dim=1)
    return alpha * tckd_loss + beta * nckd_loss

def multi_dkd(out_s_multi, out_t_multi, target, alpha, beta, temperature, img, patch_size, edge_weight):
    ###############################shape convert######################
    #  from B X C X N to N*B X C. Here N is the number of decoupled region
    #####################

    out_s_multi_t = out_s_multi.permute(2, 0, 1)  # [N, B, C]
    out_t_multi_t = out_t_multi.permute(2, 0, 1)  # [N, B, C]

    out_t = torch.reshape(out_t_multi_t, (out_t_multi_t.shape[0] * out_t_multi_t.shape[1], out_t_multi_t.shape[2]))  # [N*B, C]
    out_s = torch.reshape(out_s_multi_t, (out_s_multi_t.shape[0] * out_s_multi_t.shape[1], out_s_multi_t.shape[2]))  # [N*B, C]
    
    target_r = target.repeat(out_t_multi.shape[2])  # [N*B]

    ####################### 计算蒸馏损失 ##########################
    loss = dkd_loss(out_s, out_t, target_r, alpha, beta, temperature)  # [N*B]

    ###################### 实现的是互补性和一致性局部蒸馏损失权重调整机制 #############################
    
    out_t_predict = torch.argmax(out_t, dim=1)  # [N*B]

    mask_true = out_t_predict == target_r  # [N*B]
    mask_false = out_t_predict != target_r  # [N*B]

    # 获取全局预测（每个batch的第一个token）
    global_prediction = out_t_predict[0:len(target)]  # [B]
    global_prediction_true_mask = global_prediction == target  # [B]
    global_prediction_false_mask = global_prediction != target  # [B]

    num_tokens = out_t_multi.shape[2]  # N
    
    # 修复：只在第一个维度上repeat，而不是所有维度
    global_prediction_true_mask_repeat = global_prediction_true_mask.repeat(num_tokens)  # [N*B]
    global_prediction_false_mask_repeat = global_prediction_false_mask.repeat(num_tokens)  # [N*B]

    # 全局正确，局部错误
    mask_false_copy = mask_false.clone()  # 创建副本避免原地修改
    mask_false_copy[global_prediction_false_mask_repeat] = False
    mask_false_copy[0:len(target)] = False  # 排除全局token
    gt_lw = mask_false_copy

    # 全局错误，局部正确
    mask_true_copy = mask_true.clone()  # 创建副本避免原地修改
    mask_true_copy[global_prediction_true_mask_repeat] = False
    mask_true_copy[0:len(target)] = False  # 排除全局token
    gw_lt = mask_true_copy

    # 重新创建mask，因为之前的操作修改了原始mask
    mask_false_new = out_t_predict != target_r  # [N*B]
    mask_true_new = out_t_predict == target_r  # [N*B]
    
    index = torch.zeros_like(loss).float()  # [N*B]

    # 全局错误，局部错误
    mask_false_new[global_prediction_true_mask_repeat] = False
    gw_lw = mask_false_new
    
    # 全局正确，局部正确
    mask_true_new[global_prediction_false_mask_repeat] = False
    gt_lt = mask_true_new

    # 验证分类是否正确
    assert torch.sum(gt_lt) + torch.sum(gw_lw) + torch.sum(gt_lw) + torch.sum(gw_lt) == target_r.shape[0]

    # 权重分配
    index[gw_lw] = 1.0
    index[gt_lt] = 1.0
    index[gw_lt] = 2.0
    index[gt_lw] = 2.0

    loss = torch.sum(loss * index)  # 标量

    if torch.isnan(loss) or torch.isinf(loss):
        print("inf")
        loss = torch.zeros(1, device=loss.device, dtype=loss.dtype)

    # 边缘感知损失部分
    edge_mask = get_edge_mask(img)  # B,1,H,W
    patch_mask = mask_to_patch(edge_mask, patch_size)  # B,N
    patch_mask = patch_mask.unsqueeze(1).expand(-1, out_s_multi.shape[1], -1)  # B,C,N
    logit_s_edge = out_s_multi * patch_mask
    logit_t_edge = out_t_multi * patch_mask
    edge_loss = F.mse_loss(logit_s_edge, logit_t_edge, reduction='sum') / (patch_mask.sum() + 1e-6)

    # 合并损失
    loss_dukd = loss + edge_weight * edge_loss

    return loss_dukd

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

# 用于将输入的两个矩阵结合起来融合成另外一个矩阵
def merge_matrix(matrix1, matrix2):
    # matrix1代表sorted_indices_t
    # matrix2代表bool_list
    if matrix1.shape != matrix2.shape:
        return
    matrix = torch.zeros_like(matrix1)
    for i in range(matrix1.shape[0]):
        for j in range(matrix1.shape[1]):
            a = matrix1[i, j].item()
            b = matrix2[i, j].item()
            matrix[i, a] = b
    return matrix

# 多步骤蒸馏
def step_loss(teacher_logits, student_logits, T_right, T_error):
    # 将教师模型和学生模型的logit排序
    sorted_teacher_logits, sorted_indices_t = torch.sort(teacher_logits, descending=True, dim=1)
    sorted_student_logits, sorted_indices_s = torch.sort(student_logits, descending=True, dim=1)
    num_classes = sorted_teacher_logits.shape[1]
    batch_size = sorted_teacher_logits.shape[0]

    # 创建一个列表用于存储布尔张量 最后的维度是NxC
    bool_list = []
    for i in range(num_classes):
        matches = sorted_indices_t[:, i] == sorted_indices_s[:, i]
        bool_list.append(matches)

    bool_matrix = torch.stack(bool_list, dim=1)
    
    # 生成在原logits基础上的bool值
    bool_mask = merge_matrix(bool_matrix, sorted_indices_t)

    # 构造蒸馏函数，错误类别的赋予更高的损失值，反向传播的时候给予更大的惩罚
    # 做乘法得到正确分类的对齐损失
    right_logits_t = bool_mask.float() * teacher_logits
    right_logits_s = bool_mask.float() * student_logits
    log_pred_student_r = F.log_softmax(right_logits_s / T_right, dim=1)
    pred_teacher_r = F.softmax(right_logits_t / T_right, dim=1)
    right_loss = F.kl_div(log_pred_student_r, pred_teacher_r, reduction="none").sum(1).mean()

    # 对bool_mask取反，用于获得错误分类的对齐损失
    
    inverted_mask = ~bool_mask
    error_logits_t = inverted_mask.float() * teacher_logits
    error_logits_s = inverted_mask.float() * student_logits
    log_pred_student_e = F.log_softmax(error_logits_s / T_error, dim=1)
    pred_teacher_e = F.softmax(error_logits_t / T_error, dim=1)
    error_loss = F.kl_div(log_pred_student_e, pred_teacher_e, reduction="none").sum(1).mean()


    return right_loss

def pearson_corr(student_logits, teacher_logits, dim=-1, eps=1e-8):
    """
    计算 x 与 y 沿 dim 维度的皮尔逊相关系数
    返回形状为 [...]（少了 dim 维度）
    """
    x = student_logits - student_logits.mean(dim=dim, keepdim=True)
    y = teacher_logits - teacher_logits.mean(dim=dim, keepdim=True)
    cov = (x * y).sum(dim=dim)
    var_x = (x * x).sum(dim=dim).clamp(min=eps)
    var_y = (y * y).sum(dim=dim).clamp(min=eps)
    return cov / (var_x.sqrt() * var_y.sqrt())


def cos_loss(logits_student, logits_teacher):
    # 1. 温度缩放
    z_s = logits_student
    z_t = logits_teacher

    # 2. L2 归一化（等价于 softmax 后归一化再开方，但更稳）
    z_s = F.normalize(z_s, p=2, dim=1)
    z_t = F.normalize(z_t, p=2, dim=1)

    # 3. 余弦相似度 + 1-cos 作损失
    cos_sim = (z_s * z_t).sum(dim=1)          # (B,)
    loss_cos = (1.0 - cos_sim).mean()

    return loss_cos







    
'''
蒸馏方法
对于单张图像
先自动生成一个高斯分布
由该高斯分布分别生成教师模型和学生模型输出的变化概率图
计算概率图前期、中期、后期的两两差异，形成新的logit分布，再进行kl散度的计算，形成损失
'''

def generate_gaussian_logits(batch_size, num_classes, logits_student, temperature=1.0, device=None):
    """
    使用PyTorch生成高斯分布的logits
    """
    if device is None:
        device = logits_student.device
    
    # 使用PyTorch生成标准高斯分布
    gaussian_values = torch.randn(batch_size, num_classes, device=device)
    
    # 应用温度参数
    exp_values = torch.exp(gaussian_values / temperature)
    
    # 转换为概率分布（虽然叫logits，但这里实际上是概率的中间值）
    probabilities = exp_values / torch.sum(exp_values)
    
    # 转换回logits空间（可选，取决于你的需求）
    # 如果后续需要与logits混合，可能需要保持在这个空间
    gaussian_logits = torch.log(probabilities + 1e-10)
    
    return gaussian_logits

def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd
    
# 计算不同阶段的kl损失值
def kl_loss(logits_gauss, teacher_logits, student_logits, t1, t2, t, batch_size, num_classes, T_t, T_s):
    device = student_logits.device
    # 插值
    t1_tensor = torch.full((batch_size, num_classes), t1, device=device)
    t2_tensor = torch.full((batch_size, num_classes), t2, device=device)

    # 计算kl损失
    # 教师
    prob_t_t1 = (1 - t1_tensor)[:, None] * (logits_gauss) + t1_tensor[:, None] * (teacher_logits / T_t)
    prob_t_t2 = (1 - t2_tensor)[:, None] * (logits_gauss) + t2_tensor[:, None] * (teacher_logits / T_t)
    # 学生
    prob_s_t1 = (1 - t1_tensor)[:, None] * (logits_gauss) + t1_tensor[:, None] * (student_logits / T_s)
    prob_s_t2 = (1 - t2_tensor)[:, None] * (logits_gauss) + t2_tensor[:, None] * (student_logits / T_s)

    # 新的logits
    logits_t1 = (prob_t_t1 - prob_t_t2)
    logits_s1 = (prob_s_t1 - prob_s_t2)

    # 将logits转化为概率分布
    logits_t1 = F.softmax(logits_t1, dim=1)
    logits_s1 = F.softmax(logits_s1, dim=1)

    # 添加数值稳定性
    logits_t1 = torch.clamp(logits_t1, min=1e-10, max=1 - 1e-10)
    logits_s1 = torch.clamp(logits_s1, min=1e-10, max=1 - 1e-10)

    loss_pearson = pearson_corr(logits_s1, logits_t1, dim=1, eps=1e-8).sum(1).mean()
    # loss_kd = F.kl_div(logits_s1, logits_t1, reduction="none").sum(1).mean()
    return loss_pearson
        
# 针对教师模型输出的概率变化图生成每个时间点的熵
def cal_entropy(prob_t, epsilon=1e-8):
    # 使用PyTorch操作，保持在同一设备上
    entropies = -torch.sum(prob_t * torch.log(prob_t + epsilon), dim=-1)
    # 返回一个张量，(length,)，即一维张量
    return entropies

# 根据熵值进行不确定性分析
def uncertainty_analysis(prob_t):

    # 计算熵
    print(f"prob_t shape: {prob_t.shape}")  # 调试信息
    print(f"prob_t length: {len(prob_t)}")  # 调试信息
    
    entropies = cal_entropy(prob_t)
    print(f"entropies length: {len(entropies)}")  # 调试信息
    
    # 确保entropies在CPU上并转换为numpy数组
    if torch.is_tensor(entropies):
        entropies_np = entropies.detach().cpu().numpy()
    else:
        entropies_np = entropies
    
    # 计算熵的导数
    
    entropy_derivative = np.gradient(entropies_np)
    
    # 计算不确定性分数
    scores = entropy_derivative / (entropies_np + 1e-8)
    
    # 如果需要返回PyTorch张量，可以转换回原始设备
    if torch.is_tensor(prob_t):
        scores = torch.from_numpy(scores).to(prob_t.device)
    
    return scores

# 选择得分高的但距离又不能太近的
def select_top_points_with_min_distance(scores, n_points=10, min_distance=0.2):
    selected_points = []
    sorted_indices = np.argsort(scores)[::-1]

    for idx in sorted_indices:
        if len(selected_points) >= n_points:
            break

        # 检查是否与已选择的点距离太近
        too_close = False
        for selected_point in selected_points:
            if abs(idx - selected_point) < min_distance:
                too_close = True
                break
        if not too_close:
            selected_points.append(idx)

    # 如果所选的点不够，放宽距离限制
    if len(selected_points) < n_points:
        for idx in sorted_indices:
            if idx not in selected_points:
                selected_points.append(idx)
            if len(selected_points) >= n_points:
                break

    return sorted(selected_points)


# 生成变化概率图
def dukd_loss(logits_gauss, teacher_logits, student_logits, T_t, T_s):
    # 连续插值
    # 获取学生logits的设备
    device = student_logits.device
    batch_size = student_logits.size(0)
    num_classes = student_logits.size(1)
    teacher_logits = teacher_logits.to(device)
    logits_gauss = logits_gauss.to(device)

    # 计算一波kd损失
    loss_kd = kd_loss(student_logits, teacher_logits, T_t)

    # right_loss = step_loss(teacher_logits, student_logits, T_s, T_t)
    
    t = torch.linspace(0, 1, 200, device=device)
    t = t.view(200, 1, 1)  # [200, 1, 1]

    # 调整所有logits的维度用于广播
    logits_gauss = logits_gauss.unsqueeze(0)
    teacher_logits = teacher_logits.unsqueeze(0)
    student_logits = student_logits.unsqueeze(0)
    
    prob_t = (1 - t)[:, None] * (logits_gauss) + t[:, None] * (teacher_logits / T_t)
    prob_s = (1 - t)[:, None] * (logits_gauss) + t[:, None] * (student_logits / T_s)

    #scores = uncertainty_analysis(prob_t)
    #key_points = select_top_points_with_min_distance(scores, n_points=10, min_distance=0.2)

    #print("key points is", key_points)
    # 生成概率分布图
    #p_t = np.exp(prob_t) / np.exp(prob_t).sum(axis=1, keepdims=True)
    #p_s = np.exp(prob_s) / np.exp(prob_s).sum(axis=1, keepdims=True)

    s1, e1 = 0.1, 0.4
    s2, e2 = 0.5, 0.7
    s3, e3 = 0.7, 0.8
    s4, e4 = 0.9, 1.0
    # 将float值转换为Tensor
    s1_tensor = torch.full((batch_size, num_classes), s1, device=device)
    e1_tensor = torch.full((batch_size, num_classes), e1, device=device)
    s2_tensor = torch.full((batch_size, num_classes), s2, device=device)
    e2_tensor = torch.full((batch_size, num_classes), e2, device=device)
    s3_tensor = torch.full((batch_size, num_classes), s3, device=device)
    e3_tensor = torch.full((batch_size, num_classes), e3, device=device)
    s4_tensor = torch.full((batch_size, num_classes), s4, device=device)
    e4_tensor = torch.full((batch_size, num_classes), e4, device=device)
    
    
    # 计算前期kl损失
    # 教师模型的
    prob_t_s1 = (1-s1_tensor)[:, None] * (logits_gauss) + s1_tensor[:, None] * (teacher_logits / T_t)
    prob_t_e1 = (1-e1_tensor)[:, None] * (logits_gauss) + e1_tensor[:, None] * (teacher_logits / T_t)
    # 学生模型的
    prob_s_s1 = (1-s1_tensor)[:, None] * (logits_gauss) + s1_tensor[:, None] * (student_logits / T_s)
    prob_s_e1 = (1-e1_tensor)[:, None] * (logits_gauss) + e1_tensor[:, None] * (student_logits / T_s)
    # 新的logits
    logits_t1 = prob_t_e1 - prob_t_s1
    logits_s1 = prob_s_e1 - prob_s_s1
    # 将logits转化为概率分布
    logits_t1 = F.softmax(logits_t1, dim=1)
    logits_s1 = F.softmax(logits_s1, dim=1)
    # 添加数值稳定性
    logits_t1 = torch.clamp(logits_t1, min=1e-10, max=1-1e-10)
    logits_s1 = torch.clamp(logits_s1, min=1e-10, max=1-1e-10)
    loss_kl1 = F.kl_div(logits_s1, logits_t1, reduction="none").sum(1).mean()
    #loss_pearson1 = pearson_corr(logits_s1, logits_t1, dim=1, eps=1e-8).sum(1).mean()
    #loss_cos1 = cos_loss(logits_s1, logits_t1)
    
    # 计算中期kl损失
    prob_t_s2 = (1 - s2_tensor)[:, None] * (logits_gauss) + s2_tensor[:, None] * (teacher_logits / T_t)
    prob_t_e2 = (1 - e2_tensor)[:, None] * (logits_gauss) + e2_tensor[:, None] * (teacher_logits / T_t)

    prob_s_s2 = (1 - s2_tensor)[:, None] * (logits_gauss) + s2_tensor[:, None] * (student_logits / T_s)
    prob_s_e2 = (1 - e2_tensor)[:, None] * (logits_gauss) + e2_tensor[:, None] * (student_logits / T_s)
    
    logits_t2 = prob_t_e2 - prob_t_s2
    logits_s2 = prob_s_e2 - prob_s_s2
    
    logits_t2 = F.softmax(logits_t2, dim=1)
    logits_s2 = F.softmax(logits_s2, dim=1)
    
    logits_t2 = torch.clamp(logits_t2, min=1e-10, max=1-1e-10)
    logits_s2 = torch.clamp(logits_s2, min=1e-10, max=1-1e-10)

    
    loss_kl2 = F.kl_div(logits_s2, logits_t2, reduction="none").sum(1).mean()
    #loss_pearson2 = pearson_corr(logits_s2, logits_t2, dim=1, eps=1e-8).sum(1).mean()
    #loss_cos2 = cos_loss(logits_s2, logits_t2)
    


    # 计算后期kl损失
    prob_t_s3 = (1 - s3_tensor)[:, None] * (logits_gauss) + s3_tensor[:, None] * (teacher_logits / T_t)
    prob_t_e3 = (1 - e3_tensor)[:, None] * (logits_gauss) + e3_tensor[:, None] * (teacher_logits / T_t)

    prob_s_s3 = (1 - s3_tensor)[:, None] * (logits_gauss) + s3_tensor[:, None] * (student_logits / T_s)
    prob_s_e3 = (1 - e3_tensor)[:, None] * (logits_gauss) + e3_tensor[:, None] * (student_logits / T_s)
    
    logits_t3 = prob_t_e3 - prob_t_s3
    logits_s3 = prob_s_e3 - prob_s_s3

    logits_t3 = F.softmax(logits_t3, dim=1)
    logits_s3 = F.softmax(logits_s3, dim=1)
    
    logits_t3 = torch.clamp(logits_t3, min=1e-10, max=1-1e-10)
    logits_s3 = torch.clamp(logits_s3, min=1e-10, max=1-1e-10)
    
    loss_kl3 = F.kl_div(logits_s3, logits_t3, reduction="none").sum(1).mean()
    #loss_pearson3 = pearson_corr(logits_s3, logits_t3, dim=1, eps=1e-8).sum(1).mean()
    #loss_cos3 = cos_loss(logits_s3, logits_t3)
    
    prob_t_s4 = (1 - s4_tensor)[:, None] * (logits_gauss) + s4_tensor[:, None] * (teacher_logits / T_t)
    prob_t_e4 = (1 - e4_tensor)[:, None] * (logits_gauss) + e4_tensor[:, None] * (teacher_logits / T_t)

    prob_s_s4 = (1 - s4_tensor)[:, None] * (logits_gauss) + s4_tensor[:, None] * (student_logits / T_s)
    prob_s_e4 = (1 - e4_tensor)[:, None] * (logits_gauss) + e4_tensor[:, None] * (student_logits / T_s)
    
    logits_t4 = prob_t_e4 - prob_t_s4
    logits_s4 = prob_s_e4 - prob_s_s4

    logits_t4 = F.softmax(logits_t4, dim=1)
    logits_s4 = F.softmax(logits_s4, dim=1)
    
    logits_t4 = torch.clamp(logits_t4, min=1e-10, max=1-1e-10)
    logits_s4 = torch.clamp(logits_s4, min=1e-10, max=1-1e-10)
    
    loss_kl4 = F.kl_div(logits_s4, logits_t4, reduction="none").sum(1).mean()
    

    # 在概率变化图中选时间点去拟合
    #loss_kd2 = kd_loss(logits_s2, logits_t2, T_t)
    #loss_kd3 = kd_loss(logits_s3, logits_t3, T_t)
    '''
    print("loss_kl1:", loss_kl1)
    print("loss_kl2:", loss_kl2)
    print("loss_kl3:", loss_kl3)
    print("loss_kd:", loss_kd)'''
    
    loss = 0.3*loss_kl1 + 0.1*loss_kl2 + 0.1*loss_kl3 + 0.1*loss_kl4 + loss_kd
    #loss = loss_kd + loss_kd1 + loss_kd2 + loss_kd3
    #loss = loss_pearson1 + loss_pearson2 + loss_pearson3
    #loss = loss_cos1 + loss_cos2 + loss_cos3
    return loss


class DUKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(DUKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DUKD.CE_WEIGHT
        self.dukd_loss_weight = cfg.DUKD.DUKD_WEIGHT
        self.t_t = cfg.DUKD.T_t
        self.t_s = cfg.DUKD.T_s
        self.a = cfg.DUKD.A
        self.b = cfg.DUKD.B
        self.n = cfg.DUKD.N
        self.warmup = cfg.DUKD.WARMUP
        self.min_n = cfg.DUKD.MIN_N
        self.epochs = cfg.SOLVER.EPOCHS
        self.temperature = cfg.DUKD.TEMPERATURE

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # 获取类别数 - 从logits的形状获取
        device = logits_student.device
        batch_size = logits_student.size(0)
        
        #print(f"Student logits device: {logits_student.device}")
        #print(f"Teacher logits device: {logits_teacher.device}")
        num_classes = logits_student.size(1)  # 形状通常是 [batch_size, num_classes]
        gauss_logits = generate_gaussian_logits(batch_size, num_classes, logits_student, self.temperature, device)
        #print(f"Gauss logits device: {gauss_logits.device}")
        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dukd = self.dukd_loss_weight * dukd_loss(
            gauss_logits, logits_student, logits_teacher, self.t_t, self.t_s
        )
        
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_dukd": loss_dukd,
        }
        return logits_student, losses_dict
        
'''
class DUKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(DUKD, self).__init__(student, teacher)
        self.temperature = cfg.DUKD.TEMPERATURE
        self.ce_loss_weight = cfg.DUKD.CE_WEIGHT
        # self.kd_loss_weight = cfg.DUKD.LOSS.KD_WEIGHT
        #self.causal_loss_weight = cfg.DUKD.LOSS.CAUSAL_WEIGHT
        self.step_loss_weight = cfg.DUKD.STEP_WEIGHT
        self.t_right = cfg.DUKD.T_right
        self.t_error = cfg.DUKD.T_error
        self.a = cfg.DUKD.A
        self.b = cfg.DUKD.B
        self.n = cfg.DUKD.N
        self.warmup = cfg.DUKD.WARMUP
        self.min_n = cfg.DUKD.MIN_N
        self.epochs = cfg.SOLVER.EPOCHS

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses

        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        right_loss, error_loss = step_loss(logits_teacher, logits_student, self.t_right, self.t_error, self.a)
        right_weight = max(min(kwargs["epoch"] / self.warmup, 1.0), self.min_n)
        error_weight = min((self.epochs - kwargs["epoch"]) / max(self.epochs - self.n, 1) + self.min_n, 1.0)
        loss_step = right_weight * self.a * right_loss + error_weight * self.b * error_loss

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_step": loss_step,
        }
        return logits_student, losses_dict'''

'''
# 多区域解耦知识蒸馏
def multi_dkd(out_s_multi, out_t_multi, target, alpha, beta, temperature, img, patch_size, edge_weight):
    ###############################shape convert######################
    #  from B X C X N to N*B X C. Here N is the number of decoupled region
    #####################

    out_s_multi_t = out_s_multi.permute(2, 0, 1)
    out_t_multi_t = out_t_multi.permute(2, 0, 1)

    out_t = torch.reshape(out_t_multi_t, (out_t_multi_t.shape[0] * out_t_multi_t.shape[1], out_t_multi_t.shape[2]))
    out_s = torch.reshape(out_s_multi_t, (out_s_multi_t.shape[0] * out_s_multi_t.shape[1], out_s_multi_t.shape[2]))
    # print(out_s.shape)
    target_r = target.repeat(out_t_multi.shape[2])

    ####################### 计算蒸馏损失 ##########################
    # only conduct average or sum in the dim of calss and skip the dim of batch

    loss = dkd_loss(out_s, out_t, target_r, alpha, beta, temperature)

    ###################### 实现的是互补性和一致性局部蒸馏损失权重调整机制 #############################
	
    out_t_predict = torch.argmax(out_t, dim=1)

    mask_true = out_t_predict == target_r
    mask_false = out_t_predict != target_r

    # global_prediction = out_t_predict[len(target_r) - len(target):len(target_r)]
    global_prediction = out_t_predict[0:len(target)]
    global_prediction_true_mask = global_prediction == target
    global_prediction_false_mask = global_prediction != target

    num_tokens = out_t_multi.shape[2] 
    global_prediction_true_mask_repeat = torch.tensor(global_prediction_true_mask).repeat(out_t_multi.shape[2])
    global_prediction_false_mask_repeat = torch.tensor(global_prediction_false_mask).repeat(out_t_multi.shape[2])

    # 全局正确，局部错误
    mask_false[global_prediction_false_mask_repeat] = False
    # mask_false[len(target_r) - len(target):len(target_r)] = False
    mask_false[0:len(target)] = False
    gt_lw = mask_false

    # 全局错误，局部正确
    mask_true[global_prediction_true_mask_repeat] = False
    # mask_true[len(target_r) - len(target):len(target_r)] = False
    mask_true[0:len(target)] = False
    gw_lt = mask_true
    mask_false = out_t_predict != target_r
    mask_true = out_t_predict == target_r
    index = torch.zeros_like(loss).float()

    # 全局错误，局部错误
    mask_false[global_prediction_true_mask_repeat] = False
    gw_lw = mask_false
    
    # 全局正确，局部正确
    mask_true[global_prediction_false_mask_repeat] = False
    gt_lt = mask_true

    assert torch.sum(gt_lt) + torch.sum(gw_lw) + torch.sum(gt_lw) + torch.sum(gw_lt)==target_r.shape[0]

    # 权重分配
    index[gw_lw] = 1.0
    index[gt_lt] = 1.0
    index[gw_lt] = 2.0
    index[gt_lw] = 2.0

    loss = torch.sum(loss * index)

    if torch.isnan(loss) or torch.isinf(loss):
        print("inf")
        loss = torch.zeros(1).float().cuda()

    # 1) 原来的 multi_dkd 逻辑不变
    # 2) 计算边缘掩码 & patch 掩码
    edge_mask = get_edge_mask(img)  # B,1,H,W
    patch_mask = mask_to_patch(edge_mask, patch_size)  # B,N

    # 3) 取出边缘 patch 的 logit
    # out_s_multi / out_t_multi: [B,C,N]
    patch_mask = patch_mask.unsqueeze(1).expand(-1, out_s_multi.shape[1], -1)  # B,C,N
    logit_s_edge = out_s_multi * patch_mask
    logit_t_edge = out_t_multi * patch_mask

    # 4) 结构一致性 loss：简单 L2
    edge_loss = F.mse_loss(logit_s_edge, logit_t_edge, reduction='sum') / (patch_mask.sum() + 1e-6)

    # 5) 合并
    loss_dukd = loss + edge_weight * edge_loss

    return loss'''

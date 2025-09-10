import argparse
import torch
import torch.backends.cudnn as cudnn
import os

cudnn.benchmark = True

from mdistiller.distillers import Vanilla
from mdistiller.models import cifar10_model_dict
from mdistiller.dataset.cifar10_c_all import get_cifar10c_all_dataloader
from mdistiller.engine.utils import load_checkpoint, validate
from mdistiller.engine.cfg import CFG as cfg

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

CIFAR10C_GROUPS = {
    "noise": ["gaussian", "impulse", "shot", "speckle"],
    "blur": ["defocus", "glass", "motion", "zoom", "gaussian_blur"],
    "weather": ["brightness", "fog", "frost", "snow", "spatter"],
    "digital": ["contrast", "elastic", "jpeg_compression", "pixelate", "saturate"],
}
ALL_CIFAR10C_TYPES = sum(CIFAR10C_GROUPS.values(), [])

# 映射真实文件名
CORRUPTION_FILE_MAP = {
    "gaussian": "gaussian_noise",
    "impulse": "impulse_noise",
    "shot": "shot_noise",
    "speckle": "speckle_noise",
    "defocus": "defocus_blur",
    "glass": "glass_blur",
    "motion": "motion_blur",
    "zoom": "zoom_blur",
    "gaussian_blur": "gaussian_blur",
    "elastic": "elastic_transform",
    # 其他名字不变
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    parser.add_argument("--corruption", type=str, default="all",
                        help="Corruption type, group (e.g., 'weather') or 'all'")
    args = parser.parse_args()

    cfg.DATASET.TYPE = "cifar10-c"
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size

    model, _ = cifar10_model_dict[args.model]
    model = model(num_classes=10)
    model.load_state_dict(load_checkpoint(args.ckpt)["model"])
    model = Vanilla(model)
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    # 获取待评估扰动列表
    if args.corruption in CIFAR10C_GROUPS:
        corruption_list = CIFAR10C_GROUPS[args.corruption]
    elif args.corruption == "all":
        corruption_list = ALL_CIFAR10C_TYPES
    else:
        corruption_list = [args.corruption]

    for corruption in corruption_list:
        real_name = CORRUPTION_FILE_MAP.get(corruption, corruption)
        print(f"\n=== Evaluating: {corruption} ({real_name}) ===")
        dataloader = get_cifar10c_all_dataloader(real_name, args.batch_size, cfg.DATASET.NUM_WORKERS)
        test_acc, test_acc_top5, test_loss = validate(dataloader, model)
        print(f"[{corruption}] Acc: {test_acc:.2f}% | Top-5 Acc: {test_acc_top5:.2f}% | Loss: {test_loss:.4f}")

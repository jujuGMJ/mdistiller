import argparse
import torch
import torch.backends.cudnn as cudnn
import os

cudnn.benchmark = True

from mdistiller.distillers import Vanilla
from mdistiller.models import cifar_model_dict, imagenet_model_dict, cifar10_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import get_imagenet_val_loader
from mdistiller.engine.utils import load_checkpoint, validate
from mdistiller.engine.cfg import CFG as cfg

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "imagenet", "cifar100-c", "cifar10", "cifar10-c"],  # 增加cifar100-c和cifar10-c
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    parser.add_argument(
        "--corruption", type=str, default="snow",  # 默认corruption为snow
        choices=["fog", "frost", "snow", "brightness"],  # 可选择的腐蚀类型
        help="Specify the type of corruption for CIFAR-10-C or CIFAR-100-C (e.g., fog, frost, snow, brightness)"
    )
    args = parser.parse_args()

    cfg.DATASET.TYPE = args.dataset
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size

    if args.dataset == "cifar100-c":  # cifar100-c支持
        from mdistiller.dataset.cifar100 import get_cifar100c_dataloaders

        _, val_loader = get_cifar100c_dataloaders(args.corruption, args.batch_size, cfg.DATASET.NUM_WORKERS)
        model, _ = cifar_model_dict[args.model]
        model = model(num_classes=100)
        ckpt = args.ckpt
        model.load_state_dict(load_checkpoint(ckpt)["model"])

    elif args.dataset == "cifar10-c":  # cifar10-c支持
        from mdistiller.dataset.cifar10 import get_cifar10c_dataloaders

        _, val_loader = get_cifar10c_dataloaders(args.corruption, args.batch_size, cfg.DATASET.NUM_WORKERS)
        model, _ = cifar10_model_dict[args.model]
        model = model(num_classes=10)
        ckpt = args.ckpt
        model.load_state_dict(load_checkpoint(ckpt)["model"])

    elif args.dataset == "imagenet":
        val_loader = get_imagenet_val_loader(args.batch_size)
        if args.ckpt == "pretrain":
            model = imagenet_model_dict[args.model](pretrained=True)
        else:
            model = imagenet_model_dict[args.model](pretrained=False)
            model.load_state_dict(load_checkpoint(args.ckpt)["model"])

    elif args.dataset == "cifar100":
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model, pretrain_model_path = cifar_model_dict[args.model]
        model = model(num_classes=num_classes)
        ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
        model.load_state_dict(load_checkpoint(ckpt)["model"])

    elif args.dataset == "cifar10":
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model, pretrain_model_path = cifar10_model_dict[args.model]
        model = model(num_classes=num_classes)
        ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
        model.load_state_dict(load_checkpoint(ckpt)["model"])

    model = Vanilla(model)
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    # Validate the model
    test_acc, test_acc_top5, test_loss = validate(val_loader, model)

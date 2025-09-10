import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict, cifar10_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset, get_cifar10c_dataloaders, get_cifar100c_dataloaders
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb
            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)

    # Load dataloader for CIFAR-10-C or CIFAR-100-C
    if cfg.DATASET.TYPE == "cifar10-c":
        print(log_msg("Loading CIFAR-10-C dataset", "INFO"))
        train_loader, val_loader = get_cifar10c_dataloaders(
            corruption=cfg.CORRUPTION.TYPE,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS
        )
        num_classes = 10
        num_data = len(train_loader.dataset)

    elif cfg.DATASET.TYPE == "cifar100-c":
        print(log_msg("Loading CIFAR-100-C dataset", "INFO"))
        train_loader, val_loader = get_cifar100c_dataloaders(
            corruption=cfg.CORRUPTION.TYPE,
            batch_size=cfg.SOLVER.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS
        )
        num_classes = 100
        num_data = len(train_loader.dataset)


    else:
        # Default behavior for other datasets
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    # Ensure labels are within the correct range
    print("Checking label range in training data...")
    for batch in train_loader:
        inputs, labels = batch[0], batch[1]
        assert labels.max() < num_classes, f"Label {labels.max()} exceeds number of classes {num_classes}"
        assert labels.min() >= 0, f"Label {labels.min()} is less than 0"

    # vanilla
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        elif cfg.DATASET.TYPE == "cifar10" or cfg.DATASET.TYPE == "cifar10-c":
            model_student = cifar10_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)

    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        elif cfg.DATASET.TYPE == "cifar10" or cfg.DATASET.TYPE == "cifar10-c":
            net, pretrain_model_path = cifar10_model_dict[cfg.DISTILLER.TEACHER]
            assert pretrain_model_path is not None, f"No pretrain model for teacher {cfg.DISTILLER.TEACHER}"
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar10_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
        elif cfg.DATASET.TYPE == "cifar100" or cfg.DATASET.TYPE == "cifar100-c":
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert pretrain_model_path is not None, f"No pretrain model for teacher {cfg.DISTILLER.TEACHER}"
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
        elif cfg.DATASET.TYPE == "cifar_fs":
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert pretrain_model_path is not None, f"No pretrain model for teacher {cfg.DISTILLER.TEACHER}"
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)

        # Initialize distiller
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student, model_teacher, cfg, num_data)
        elif cfg.DISTILLER.TYPE == "FLowKD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE]
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student, model_teacher, cfg)

    distiller = torch.nn.DataParallel(distiller.cuda())
    #distiller = torch.nn.DataParallel(distiller.to('cpu'))

    if cfg.DISTILLER.TYPE != "NONE":
        print(log_msg(f"Extra parameters of {cfg.DISTILLER.TYPE}: {distiller.module.get_extra_parameters()}", "INFO"))

    # Initialize trainer and start training
    trainer = trainer_dict[cfg.SOLVER.TRAINER](experiment_name, distiller, train_loader, val_loader, cfg)
    trainer.train(resume=resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts)

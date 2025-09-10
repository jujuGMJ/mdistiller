from .cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample, get_cifar100c_dataloaders
from .cifar10 import get_cifar10_dataloaders, get_cifar10_dataloaders_sample, get_cifar10c_dataloaders
from .imagenet import get_imagenet_dataloaders, get_imagenet_dataloaders_sample
from .cifar10_c_all import get_cifar10c_all_dataloader


def get_dataset(cfg):
    if cfg.DATASET.TYPE == "cifar100":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_cifar100_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                mode=cfg.CRD.MODE,
            )
        else:
            train_loader, val_loader, num_data = get_cifar100_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 100
    elif cfg.DATASET.TYPE == "imagenet":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_imagenet_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
            )
        else:
            train_loader, val_loader, num_data = get_imagenet_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 1000
    elif cfg.DATASET.TYPE == "cifar100-c":
        # 修改为加载 cifar100-c
        val_loader = get_cifar100c_dataloaders(
            corruption=cfg.CORRUPTION.TYPE,
            batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
        )
        num_classes = 100
    elif cfg.DATASET.TYPE == "cifar10":
        if cfg.DISTILLER.TYPE == "CRD":
            train_loader, val_loader, num_data = get_cifar10_dataloaders_sample(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
                k=cfg.CRD.NCE.K,
                mode=cfg.CRD.MODE,
            )
        else:
            train_loader, val_loader, num_data = get_cifar10_dataloaders(
                batch_size=cfg.SOLVER.BATCH_SIZE,
                val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
                num_workers=cfg.DATASET.NUM_WORKERS,
            )
        num_classes = 10
    elif cfg.DATASET.TYPE == "cifar10-c":
        val_loader = get_cifar10c_dataloaders(
            corruption=cfg.CORRUPTION.TYPE,
            batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
        )
        num_classes = 10
    elif cfg.DATASET.TYPE == "cifar10-10c":  # 添加对 cifar10 和 cifar10-c 组合数据集的支持
        train_loader, val_loader, num_data = get_cifar10_10c_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            corruption=cfg.CORRUPTION.TYPE,  # 确保传递 corruption 参数
        )
        num_classes = 10
    elif cfg.DATASET.TYPE == "cifar_fs":
        from .cifar_fs import get_cifar_fs_dataloaders
        train_loader, val_loader, num_data = get_cifar_fs_dataloaders(
            batch_size=cfg.SOLVER.BATCH_SIZE,
            val_batch_size=cfg.DATASET.TEST.BATCH_SIZE,
            num_workers=cfg.DATASET.NUM_WORKERS,
            percent=cfg.DATASET.PERCENT,  # 自定义的采样比例
        )
        num_classes = 100
    else:
        raise NotImplementedError(cfg.DATASET.TYPE)

    return train_loader, val_loader, num_data, num_classes

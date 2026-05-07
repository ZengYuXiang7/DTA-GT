import os

from torch.utils.data import DataLoader
from .nasbench import NasbenchDataset
from .fixed_length_sampler import FixedLengthBatchSampler


def _split_meta_exists(base_path):
    stem = base_path[:-3] if base_path.endswith(".pt") else base_path
    return os.path.exists(f"{stem}.meta.pt")


def _resolve_nasbench_data_path(dataset, embed_type):
    default_path = f"data/{dataset}/all_{dataset}.pt"
    if _split_meta_exists(default_path):
        return default_path

    legacy_path = f"data/{dataset}/all_{dataset}_{embed_type}.pt"
    if _split_meta_exists(legacy_path):
        return legacy_path

    return default_path


def init_dataloader(args, logger):
    # Load Dataset
    if "nasbench" in args.dataset:
        args.data_path = _resolve_nasbench_data_path(args.dataset, args.embed_type)
        if args.do_train:
            runid = getattr(args, "runid", 0)
            trainset = NasbenchDataset(
                logger,
                args.dataset,
                "train",
                args.data_path,
                args.percent,
                args.lambda_consistency,
                embed_type=args.embed_type,
                runid=runid,
            )
            valset = NasbenchDataset(
                logger,
                args.dataset,
                "val",
                args.data_path,
                args.percent,
                embed_type=args.embed_type,
                runid=runid,
            )
            train_sampler = FixedLengthBatchSampler(
                trainset, args.dataset, args.batch_size, include_partial=True
            )
            val_sampler = FixedLengthBatchSampler(
                valset, args.dataset, args.batch_size, include_partial=True
            )
            train_loader = DataLoader(
                trainset,
                shuffle=(train_sampler is None),
                num_workers=0,
                pin_memory=True,
                batch_sampler=train_sampler,
            )
            val_loader = DataLoader(
                valset,
                shuffle=(val_sampler is None),
                num_workers=0,
                pin_memory=True,
                batch_sampler=val_sampler,
            )
            return train_loader, val_loader
        else:
            dataset = NasbenchDataset(
                logger,
                args.dataset,
                "test",
                args.data_path,
                args.percent,
                args.lambda_consistency,
                embed_type=args.embed_type,
                runid=getattr(args, "runid", 0),
            )
            sampler = FixedLengthBatchSampler(
                dataset, args.dataset, args.batch_size, include_partial=True
            )
            dataLoader = DataLoader(
                dataset,
                shuffle=(sampler is None),
                num_workers=0,
                pin_memory=True,
                batch_sampler=sampler,
            )
            return dataLoader

    raise ValueError(f"Unsupported dataset: {args.dataset}")

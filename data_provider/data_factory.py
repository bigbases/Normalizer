from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from torch.utils.data import DataLoader
import random
import numpy as np
import torch

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Weather': Dataset_Custom,
    'Electricity': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'custom': Dataset_Custom,
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag in ('test', 'val'):
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    # Dataset_Pred follows the original SAN/DDN signature (no `args`);
    # Dataset_ETT_* / Dataset_Custom in this repo additionally accept `args`
    # so that the augmentation hook (utils.augmentation.run_augmentation_single)
    # can use `args.augmentation_ratio` etc.
    if Data is Dataset_Pred:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )
    else:
        data_set = Data(
            args=args,
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )
    print(flag, len(data_set))
    generator = torch.Generator()
    # Use distinct but deterministic streams for each split.
    split_offset = {'train': 0, 'val': 1, 'test': 2, 'pred': 3}[flag]
    generator.manual_seed(int(args.seed) + split_offset)

    # Persistent workers avoid re-forking the loader processes every epoch
    # (train and val loaders are iterated up to 15 times per run).  Shuffling
    # still comes from the seeded generator in the main process.
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    return data_set, data_loader

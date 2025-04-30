import torch
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from utils.DataLoader import UAVDataLoaderBuilder, custom_collate_fn, IndexSampler
import importlib
from torchvision import transforms

def create_data(cfg):
    builder = UAVDataLoaderBuilder(cfg)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset, val_dataset, test_dataset, train_clean_dataset, val_clean_dataset = builder.build(
        train_ratio=cfg['dataset']['train_ratio'],
        val_ratio=cfg['dataset']['val_ratio'],
        transform=transform
    )

    # 是否使用随机索引
    train_num_samples = len(train_dataset)
    val_num_samples = len(val_dataset)
    if cfg['dataset'].get('shuffle', True):
        generator = torch.Generator().manual_seed(cfg['seed'])
        train_indices = torch.randperm(train_num_samples, generator=generator).tolist()
        val_indices = torch.randperm(val_num_samples, generator=generator).tolist()
    else:
        train_indices = list(range(train_num_samples))
        val_indices = list(range(val_num_samples))

    train_shared_sampler = IndexSampler(train_indices)
    val_shared_sampler = IndexSampler(val_indices)

    train_loader = DataLoader(train_dataset, batch_size=cfg['dataset']['batch'], sampler=train_shared_sampler,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg['dataset']['batch'], sampler=val_shared_sampler,
                            collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg['dataset']['batch'], shuffle=cfg['dataset'].get('shuffle', True),
                             collate_fn=custom_collate_fn)

    if cfg['dataset']['is_clean']:
        train_clean_loader = DataLoader(train_clean_dataset, batch_size=cfg['dataset']['batch'], sampler=train_shared_sampler,
                                  collate_fn=custom_collate_fn)
        val_clean_loader = DataLoader(val_clean_dataset, batch_size=cfg['dataset']['batch'], sampler=val_shared_sampler,
                                      collate_fn=custom_collate_fn)
    else:
        train_clean_loader = None
        val_clean_loader = None

    return train_loader, val_loader, test_loader, train_clean_loader, val_clean_loader

def create_model(cfg):
    detector = cfg['detector']
    module = importlib.import_module(f'models.{detector}.{detector}')

    # 获取函数
    func = getattr(module, detector)
    model = func(cfg)
    model.to(cfg['device'])
    model.device = cfg['device']  # Add device attribute for model to use

    # 5. Create an optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'])
    model.optimizer = optimizer  # Add optimizer to the model
    return model



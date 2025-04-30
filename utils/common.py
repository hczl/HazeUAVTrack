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
    train_dataset, val_dataset, test_dataset, clean_dataset = builder.build(
        train_ratio=cfg['dataset']['train_ratio'],
        val_ratio=cfg['dataset']['val_ratio'],
        transform=transform
    )

    # 是否使用随机索引
    num_samples = len(train_dataset)
    if cfg['dataset'].get('shuffle', True):
        generator = torch.Generator().manual_seed(cfg['seed'])
        indices = torch.randperm(num_samples, generator=generator).tolist()
    else:
        indices = list(range(num_samples))

    shared_sampler = IndexSampler(indices)

    train_loader = DataLoader(train_dataset, batch_size=cfg['dataset']['batch'], sampler=shared_sampler,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg['dataset']['batch'], shuffle=cfg['dataset'].get('shuffle', True),
                            collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg['dataset']['batch'], shuffle=cfg['dataset'].get('shuffle', True),
                             collate_fn=custom_collate_fn)

    if cfg['dataset']['is_clean']:
        clean_loader = DataLoader(clean_dataset, batch_size=cfg['dataset']['batch'], sampler=shared_sampler,
                                  collate_fn=custom_collate_fn)
    else:
        clean_loader = None

    return train_loader, val_loader, test_loader, clean_loader

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



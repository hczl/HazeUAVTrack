import torch
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from utils.DataLoader import UAVDataLoaderBuilder, custom_collate_fn
import importlib
from torchvision import transforms
def create_data(cfg):
    builder = UAVDataLoaderBuilder(cfg)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset, val_dataset, test_dataset, clean_dataset = builder.build(train_ratio=0.7, val_ratio=0.2,
                                                                            transform=transform)

    # 构造共享的 sampler（假设 train_dataset 和 clean_dataset 是一一对应的）
    generator_train = torch.Generator().manual_seed(cfg['seed'])
    sampler_train = RandomSampler(train_dataset, generator=generator_train)
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=sampler_train, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
    if cfg['is_clean']:
        generator_clean = torch.Generator().manual_seed(cfg['seed'])
        sampler_clean = RandomSampler(clean_dataset, generator=generator_clean)
        clean_loader = DataLoader(clean_dataset, batch_size=8, sampler=sampler_clean, collate_fn=custom_collate_fn)
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


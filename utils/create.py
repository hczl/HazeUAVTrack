import torch
from torch import optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, StepLR
from torch.utils.data import DataLoader

from HazeUAVTrack import HazeUAVTrack
from utils.DataLoader import UAVDataLoaderBuilder, custom_collate_fn, IndexSampler

from torchvision import transforms

def create_data(cfg):
    builder = UAVDataLoaderBuilder(cfg)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
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
                              collate_fn=custom_collate_fn, num_workers=cfg['dataset'].get('num_workers', 4),
                              pin_memory=False)

    val_loader = DataLoader(val_dataset, batch_size=cfg['dataset']['batch'], sampler=val_shared_sampler,
                            collate_fn=custom_collate_fn, num_workers=cfg['dataset'].get('num_workers', 4),
                            pin_memory=False)

    test_loader = DataLoader(test_dataset, batch_size=cfg['dataset']['batch'],
                             shuffle=cfg['dataset'].get('shuffle', True),
                             collate_fn=custom_collate_fn, num_workers=cfg['dataset'].get('num_workers', 4),
                             pin_memory=False)

    if cfg['dataset']['is_clean']:
        train_clean_loader = DataLoader(train_clean_dataset, batch_size=cfg['dataset']['batch'],
                                        sampler=train_shared_sampler,
                                        collate_fn=custom_collate_fn, num_workers=cfg['dataset'].get('num_workers', 4),
                                        pin_memory=False)

        val_clean_loader = DataLoader(val_clean_dataset, batch_size=cfg['dataset']['batch'], sampler=val_shared_sampler,
                                      collate_fn=custom_collate_fn, num_workers=cfg['dataset'].get('num_workers', 4),
                                      pin_memory=False)

    else:
        train_clean_loader = None
        val_clean_loader = None

    return train_loader, val_loader, test_loader, train_clean_loader, val_clean_loader

def create_model(cfg):
    model = HazeUAVTrack(cfg)
    model.to(cfg['device'])
    model.device = cfg['device']  # Add device attribute for model to use

    # 5. Create an optimizer
    initial_lr = cfg['train']['lr']
    # 建议使用 AdamW，特别是当你的模型包含 Weight Decay 时
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=initial_lr)

    # --- Create learning rate scheduler ---
    num_epochs = cfg['train']['epochs']
    warmup_epochs = cfg['train'].get('warmup_epochs', 0) # 从 config 获取，默认 0
    warmup_start_factor = cfg['train'].get('warmup_start_factor', 1.0) # 从 config 获取，默认 1.0

    # 获取主衰减调度器的参数
    lr_decay_step = cfg['train'].get('lr_decay_step', 30)
    lr_decay_gamma = cfg['train'].get('lr_decay_gamma', 0.1)

    # 创建主衰减调度器 (例如 StepLR)
    main_scheduler = StepLR(
        optimizer,
        step_size=lr_decay_step,
        gamma=lr_decay_gamma
    )

    # 如果设置了 warmup_epochs > 0，则创建 warmup 调度器并与主调度器组合
    if warmup_epochs > 0:
         warmup_scheduler = LinearLR(
             optimizer,
             start_factor=warmup_start_factor,
             end_factor=1.0, # Warmup 结束时达到基础学习率
             total_iters=warmup_epochs
         )
         # 使用 SequentialLR 将 warmup 和主调度器按顺序执行
         scheduler = SequentialLR(
             optimizer,
             schedulers=[warmup_scheduler, main_scheduler],
             milestones=[warmup_epochs] # 在 warmup_epochs 结束后切换
         )
    else:
         # 如果没有 warmup，直接使用主调度器
         scheduler = main_scheduler

    # --- Scheduler creation end ---

    model.optimizer = optimizer  # Add optimizer to the model
    model.scheduler = scheduler # Add scheduler to the model

    return model
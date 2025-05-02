class Settings:
    """
    FALCON 模型设置类
    """

    def __init__(self):
        """
        初始化设置，设置默认值
        """

        self.ffc = {
            'loop': 1,
            'nums': 2
        }

        self.train = {
            'w': 1.0,
            'loss_ratio': [1, 1, 1],
            'perceptual': {
                'net': 'vgg16',
                'style': ['3', '8', '15'],
                'content': ['8']
            },
            'model': 'unet_fal',
            'epoch': 100,
            'batch': 8,
            'init_weight': None,
            'input_kernel': [15]
        }

        self.sched = {
            'name': 'steplr',
            'init_lr': 1.0e-3,
            'lr_max': 5.0e-3,
            'cawr2': {
                't0': 1,
                'tmul': 1,
                'tup': 0.05,
                'gamma': 0.6
            },
            'cawr': {
                't0': 1.0,
                'tmul': 1,
                'tup': 0,
                'gamma': 0.2
            },
            'steplr_multi': {
                'milestones': [1500, 2000, 4000],
                'gamma': 0.5
            }
        }

        self.opt = {
            'name': 'AdamW',
            'wd': 0
        }
        self.model_config = {
            'init': 'he_u'
        }



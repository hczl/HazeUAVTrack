class Settings:
    """
    IA_YOLOV3 模型设置类
    """

    def __init__(self):
        """
        初始化设置，设置默认值
        """
        # 色调滤镜精细程度
        self.curve_steps = 8

        # 预测参数设置
        self.dip_nums = 7 + self.curve_steps

        # defog 的压缩范围
        self.defog_range = (0.1, 1.0)

        # gamma filter的压缩范围
        self.gamma_range = 3

        # ToneFilter的压缩范围
        self.tone_curve_range = (0.5, 2)

        # USMFilter的压缩范围
        self.usm_range = (0.0, 5)

        # 预热训练的epoch数
        self.warmup_epochs = 0



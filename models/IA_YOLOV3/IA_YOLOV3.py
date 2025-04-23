import os

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

from IA_config import Settings
from CNN_PP import CNN_PP
from models.IA_YOLOV3.DIP import DIP_Module
from models.IA_YOLOV3.utils import AtmLight,DarkIcA,DarkChannel
from utils import load_config
import torch.nn.functional as F

class IA_YOLOV3(nn.Module):
    def __init__(self, config, train_dataloader, val_dataloader):

        super(IA_YOLOV3, self).__init__()
        self.IA_config = Settings()
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.cnn_pp = CNN_PP(self.IA_config.dip_nums)
        self.dip_module = DIP_Module(self.IA_config)
        # 从pytorch hub下载yolov3
        try:
            print("Loading YOLOv3u model using ultralytics.YOLO...")
            # 使用 YOLO("yolov3u.pt") 加载模型。
            # 它会自动下载 yolov3u.pt 文件（如果本地没有）并加载权重。
            yolo_wrapper = YOLO("yolov3u.pt")
            print("YOLOv3u model loaded.")

            # 提取底层的 PyTorch nn.Module
            # 在 Ultralytics 库中，底层的模型通常存储在 .model 属性中
            # 需要检查具体版本或代码来确认，但 .model 是一个常见模式
            if hasattr(yolo_wrapper, 'model') and isinstance(yolo_wrapper.model, nn.Module):
                self.yolov3 = yolo_wrapper.model
                print("Extracted underlying nn.Module from YOLO wrapper.")
            else:
                raise AttributeError("Could not find the underlying nn.Module in the YOLO object. "
                                     "The structure of ultralytics.YOLO might have changed.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # 释放 GPU 内存（如果使用了）
            # Display model information (optional)
            # Display model information (optional)
            # self.yolov3.info()

            # 查找并修改 Detect 层
            # 尝试获取原始模型的类别数 nc，虽然对于切片操作不是必需的，但有助于理解
            self.original_nc = None
            # 查找 Detect 模块以获取 nc
            detect_module = None
            for module in self.yolov3.modules():
                # 类名可能随版本变化，用 type().__name__ 更健壮些
                if type(module).__name__ == 'Detect':
                    detect_module = module
                    break
            self.original_nc = detect_module.nc
            # print(f"检测到原始模型类别数 (nc): {self.original_nc}")

        except Exception as e:
            print(f"Error : {e}")
            raise e  # 重新抛出异常

    def forward(self,inputs):
        n, c, _, _ = inputs.shape
        resized_inputs = F.interpolate(inputs, size=(256, 256), mode='bilinear', align_corners=False)
        dark = torch.zeros((inputs.shape[0], inputs.shape[2], inputs.shape[3]),
                           dtype=torch.float32, device=self.config['device'])  # 注意 dtype，通常图像处理用浮点型
        defog_A = torch.zeros((inputs.shape[0], inputs.shape[1]), dtype=torch.float32, device=self.config['device'])
        IcA = torch.zeros((inputs.shape[0], inputs.shape[2], inputs.shape[3]), dtype=torch.float32, device=self.config['device'])

        for i in range(inputs.shape[0]):
            train_data_i = inputs[i]
            dark_i = DarkChannel(train_data_i)
            defog_A_i = AtmLight(train_data_i, dark_i)
            IcA_i = DarkIcA(train_data_i, defog_A_i)
            # print(dark_i.shape, defog_A_i.shape, IcA_i.shape)
            dark[i, ...] = dark_i
            defog_A[i, ...] = defog_A_i
            IcA[i, ...] = IcA_i
        IcA = torch.unsqueeze(IcA, dim=1)

        filter_features = self.cnn_pp(resized_inputs)
        dip_output = self.dip_module(inputs, filter_features, defog_A, IcA)
        # print(dip_output.shape)
        yolov3_outputs = self.yolov3(dip_output)
        return yolov3_outputs
        # yolov3_predictions = self.yolov3_model(dip_processed_output) # DIP output to YOLOv3
        # return cnn_pp_output, yolov3_predictions # Return both outputs for loss calculation if needed

    def train_step(self, batch):
        low_res_images, high_res_images, targets_yolov3, targets_cnn_pp = batch # Assuming batch returns both inputs and targets
        low_res_images = low_res_images.to(self.device)
        high_res_images = high_res_images.to(self.device)
        targets_yolov3 = targets_yolov3.to(self.device)
        targets_cnn_pp = targets_cnn_pp.to(self.device)

        self.optimizer.zero_grad()
        cnn_pp_output, yolov3_predictions = self(low_res_images, high_res_images) # Pass both inputs

        yolov3_loss = self.calculate_detection_loss(yolov3_predictions, targets_yolov3)
        cnn_pp_loss = self.calculate_cnn_pp_loss(cnn_pp_output, targets_cnn_pp) # Example CNN-PP loss

        total_loss = yolov3_loss + cnn_pp_loss # Combine losses - adjust as needed
        total_loss.backward()
        self.optimizer.step()

        return {'total_loss': total_loss.item(), 'yolov3_loss': yolov3_loss.item(), 'cnn_pp_loss': cnn_pp_loss.item()}

    def train_epoch(self, train_loader, epoch):
        self.train()
        epoch_loss = 0.0
        epoch_yolov3_loss = 0.0
        epoch_cnn_pp_loss = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            train_info = self.train_step(batch)
            loss = train_info['total_loss']
            yolov3_loss = train_info['yolov3_loss']
            cnn_pp_loss = train_info['cnn_pp_loss']

            epoch_loss += loss
            epoch_yolov3_loss += yolov3_loss
            epoch_cnn_pp_loss += cnn_pp_loss
            batch_count += 1

            if batch_idx % self.config['train'].get('log_interval', 10) == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, Total Loss: {loss:.4f}, YOLOv3 Loss: {yolov3_loss:.4f}, CNN-PP Loss: {cnn_pp_loss:.4f}")

        avg_loss = epoch_loss / batch_count
        avg_yolov3_loss = epoch_yolov3_loss / batch_count
        avg_cnn_pp_loss = epoch_cnn_pp_loss / batch_count

        print(f"Epoch {epoch} 训练完成，平均 Total Loss: {avg_loss:.4f}, 平均 YOLOv3 Loss: {avg_yolov3_loss:.4f}, 平均 CNN-PP Loss: {avg_cnn_pp_loss:.4f}")
        return {'avg_loss': avg_loss, 'avg_yolov3_loss': avg_yolov3_loss, 'avg_cnn_pp_loss': avg_cnn_pp_loss}

    def evaluate(self, val_loader):
        self.eval()
        val_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                low_res_images, high_res_images, targets_yolov3, targets_cnn_pp = batch
                low_res_images = low_res_images.to(self.device)
                high_res_images = high_res_images.to(self.device)
                targets_yolov3 = targets_yolov3.to(self.device)
                targets_cnn_pp = targets_cnn_pp.to(self.device)

                cnn_pp_output, yolov3_predictions = self(low_res_images, high_res_images)
                yolov3_loss = self.calculate_detection_loss(yolov3_predictions, targets_yolov3) # Using YOLOv3 loss for validation example
                cnn_pp_loss = self.calculate_cnn_pp_loss(cnn_pp_output, targets_cnn_pp)
                total_loss = yolov3_loss + cnn_pp_loss

                val_loss += total_loss.item()
                batch_count += 1

        avg_val_loss = val_loss / batch_count
        print(f"验证集平均 Total Loss: {avg_val_loss:.4f}")
        return {'avg_val_loss': avg_val_loss}

    def predict(self, high_res_images, low_res_images=None): # Added low_res_images for consistency, but DIP+YOLOv3 path is main predict path
        self.eval()
        high_res_images = high_res_images.to(self.device)
        if low_res_images is not None:
            low_res_images = low_res_images.to(self.device)

        with torch.no_grad():
            dip_processed_output = self.dip_module(high_res_images)
            yolov3_predictions = self.yolov3_model(dip_processed_output)
        return yolov3_predictions # Return YOLOv3 predictions as main output


    def save_checkpoint(self, epoch, checkpoint_path):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"检查点已保存到: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        self.config = checkpoint['config']
        print(f"检查点已加载，从 epoch {start_epoch} 继续训练 (如果需要)")
        return start_epoch

    def train_model(self, train_loader, val_loader=None, start_epoch=0, num_epochs=100, checkpoint_dir='./checkpoints'):
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(start_epoch, num_epochs):
            train_stats = self.train_epoch(train_loader, epoch)

            if val_loader:
                val_stats = self.evaluate(val_loader)

            if epoch % self.config['train'].get('checkpoint_save_interval', 5) == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                self.save_checkpoint(epoch, checkpoint_path)

        print("训练完成!")


if __name__ == '__main__':

    cfg = load_config(r'D:\FFFFFiles\bis_sheji\configs\exp1.yaml')

    # 2. Create dummy datasets and dataloaders
    class DummyDataset(Dataset):
        def __len__(self):
            return 100 # Dummy dataset size

        def __getitem__(self, idx):
            # Return dummy low_res_images, high_res_images, targets_yolov3, targets_cnn_pp
            low_res_images = torch.randn(3, 512, 512) # Example image size
            high_res_images = torch.randn(3, 512, 512)
            targets_yolov3 = torch.randn(10, 5) # Example YOLOv3 targets (adjust as needed)
            targets_cnn_pp = torch.randn(32, 128, 128) # Example CNN_PP targets (adjust as needed)
            return low_res_images, high_res_images, targets_yolov3, targets_cnn_pp

    train_dataset = DummyDataset()
    val_dataset = DummyDataset()

    train_dataloader = DataLoader(train_dataset, batch_size=4) # Small batch size for testing
    val_dataloader = DataLoader(val_dataset, batch_size=4)

    # 3. Instantiate the model
    model = IA_YOLOV3(cfg, train_dataloader, val_dataloader)

    # 4. Move model to device (if you have CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.device = device # Add device attribute for model to use

    # 5. Create an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.optimizer = optimizer # Add optimizer to the model

    # 6. Test forward pass
    dummy_input = torch.randn(4, 3, 1024, 540).to(device) # Batch of 2, 3 channels, 512x512
    output = model(dummy_input)
    def see(a):
        try:
            if len(a):
                see(a[0])
                print(len(a))
        except:
            pass

    see(output[1])
    # cnn_pp_output, yolov3_predictions = model(dummy_input)
    # print("CNN_PP Output Shape:", cnn_pp_output[0].shape) # Print shape of filter_features
    # print("YOLOv3 Predictions Shape:", yolov3_predictions.shape)
    #
    # # 7. Test training for a few epochs
    # print("\n--- Starting Training Test ---")
    # model.train_model(train_dataloader, val_dataloader, num_epochs=2, checkpoint_dir='./test_checkpoints') # Train for 2 epochs
    #
    # # 8. Test checkpoint saving and loading
    # checkpoint_path = './test_checkpoints/checkpoint_epoch_1.pth' # Path to the saved checkpoint from epoch 1
    # loaded_start_epoch = model.load_checkpoint(checkpoint_path)
    # print(f"\nLoaded checkpoint, starting epoch: {loaded_start_epoch}")
    #
    # print("\n--- Testing Completed ---")

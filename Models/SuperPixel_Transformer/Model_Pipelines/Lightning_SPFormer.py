import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from Models.SuperPixel_Transformer.config import num_epochs, learning_rate
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet18 import ResNet18
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet50 import ResNet50
from Models.SuperPixel_Transformer.DataLoaders.Data_Loader import load_dataset
from Models.SuperPixel_Transformer.PNP_CNNs.MiniCNN import SuperpixelCNN
from Models.SuperPixel_Transformer.Transformer import TransformerEncoder
from Models.SuperPixel_Transformer.config import dataset_option, cnn_option

# CUDA optimization
torch.set_float32_matmul_precision('high')


def cnn_selection(option):
    if option == "ResNet18":
        return ResNet18()
    elif option == "ResNet50":
        return ResNet50()
    elif option == "SuperpixelCNN":
        return SuperpixelCNN(in_channels=3, out_channels=512)
    else:
        raise ValueError(f"Invalid CNN selection: {option}")


class LitNetwork(pl.LightningModule):
    def __init__(self):
        super(LitNetwork, self).__init__()
        n_classes = 37  # Adjust number of classes as necessary
        out_channels = 512
        self.cnn = cnn_selection(cnn_option)
        self.transformer = TransformerEncoder(out_channels, n_classes)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=n_classes, average='micro')

    def forward(self, x, superpixel_map):
        out_image = self.cnn(x)
        superpixel_map = superpixel_map.unsqueeze(1).expand_as(out_image)

        b, d, r, c = out_image.shape

        num_pix = 50 # torch.max(superpixel_map) + 1
        buffer = torch.zeros((b, d, num_pix)).to(out_image.device)

        superpix_vectors = torch.scatter_reduce(buffer, 2, superpixel_map.view(b, d, r*c), out_image.view(b, d, r*c), reduce='mean')
        superpix_vectors = superpix_vectors.permute(0, 2, 1)
        predictions = self.transformer(superpix_vectors)
        return predictions

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Return both optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",  # Monitor validation accuracy
                "interval": "epoch",  # Adjust learning rate every epoch
                "frequency": 1  # Frequency of learning rate adjustment
            }
        }

    # self.transformer optimizer to isolate the Resnet18 weights
    def training_step(self, data, batch_idx):
        superpix_map, im, label = data[0], data[1], data[2]
        outs = self.forward(im, superpix_map)
        loss = self.loss_func(outs, label)
        self.log("train_loss", loss, batch_size=1, sync_dist=True)
        return loss

    def validation_step(self, val_data, batch_idx):
        superpix_map, im, label = val_data[0], val_data[1], val_data[2]
        outs = self.forward(im, superpix_map)
        self.val_acc(outs, label)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)


if __name__ == "__main__":
    train_loader, val_loader, class_names = load_dataset(dataset_name=dataset_option)
    model = LitNetwork()
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
    logger = pl_loggers.TensorBoardLogger(save_dir="../../my_logs")

    device = "gpu"

    trainer = pl.Trainer(max_epochs=num_epochs, accelerator=device, callbacks=[checkpoint], logger=logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

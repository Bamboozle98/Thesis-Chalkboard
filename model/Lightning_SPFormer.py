import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from model.config import num_epochs, learning_rate
from Resnet18 import ResNet18
from Data_Loader_SP import data_process_SP
from MiniCNN import SuperpixelCNN
from Transformer import TransformerEncoder
from Data_Loader_SP_IN import train_loader as t, val_loader as v

# CUDA optimization
torch.set_float32_matmul_precision('high')


# Load OxfordPets datasets
# train_loader, val_loader, class_names = data_process_SP()

# Load ImageNet Datasets
train_loader, val_loader = t, v

def positionals_encoding(superpixel_map):
    rows, cols = superpixel_map.shape
    r_vals, c_vals = torch.meshgrid(rows, cols)
    center_rs = torch.scatter(r_vals, 1, superpixel_map, reduce='mean')
    center_cs = torch.scatter(c_vals, 1, superpixel_map, reduce='mean')
    # standard positional encoding logic...


class LitNetwork(pl.LightningModule):
    def __init__(self):
        super(LitNetwork, self).__init__()
        n_classes = 1000  # Adjust number of classes as necessary
        out_channels = 512
        self.cnn = SuperpixelCNN(in_channels=3, out_channels=out_channels)
        self.res_cnn = ResNet18()
        self.transformer = TransformerEncoder(out_channels, n_classes)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy("multiclass", num_classes=n_classes, average='micro')

    def forward(self, x, superpixel_map):
        out_image = self.res_cnn(x)
        #print("Shape of superpixel_map:", superpixel_map.shape)
        #print("Shape of out_image:", out_image.shape)
        superpixel_map = superpixel_map.unsqueeze(1).expand_as(out_image)

        b,d,r,c = out_image.shape

        #print(out_image.view(b,d,r*c).shape)
        #print(superpixel_map.view(b,1,r*c).shape)

        num_pix = 50 #torch.max(superpixel_map) + 1
        buffer = torch.zeros((b,d,num_pix)).to(out_image.device)

        superpix_vectors = torch.scatter_reduce(buffer, 2, superpixel_map.view(b,d,r*c), out_image.view(b,d,r*c), reduce='mean')
        superpix_vectors = superpix_vectors.permute(0, 2, 1)
        #print(superpix_vectors.shape)
        predictions = self.transformer(superpix_vectors)
        return predictions


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.transformer.parameters(), lr=learning_rate)
        return optimizer
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
    model = LitNetwork()
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max')
    logger = pl_loggers.TensorBoardLogger(save_dir="my_logs")

    device = "gpu"  # Adjust to your system's capabilities

    trainer = pl.Trainer(max_epochs=num_epochs, accelerator=device, callbacks=[checkpoint], logger=logger)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

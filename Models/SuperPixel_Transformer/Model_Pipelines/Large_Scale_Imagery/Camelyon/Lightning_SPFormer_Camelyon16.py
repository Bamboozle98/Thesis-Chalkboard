import os
import math
import numpy as np
import openslide
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, AUROC
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
import matplotlib.pyplot as plt

from Models.SuperPixel_Transformer.PNP_CNNs.Camelyon_Resnet_18 import ResNet18
from Models.SuperPixel_Transformer.PNP_CNNs.Resnet50       import ResNet50
from Models.SuperPixel_Transformer.PNP_CNNs.MiniCNN       import SuperpixelCNN
from Models.SuperPixel_Transformer.PNP_CNNs.customResnet  import CustomResNet20
from Models.SuperPixel_Transformer.Transformer            import TransformerEncoder
from Models.SuperPixel_Transformer.DataLoaders.Camelyon_16.Camelyon16DataLoader import load_camelyon
from Models.SuperPixel_Transformer.config import (
    num_epochs, learning_rate, cnn_option, use_checkpoint, wsi_dir, ann_dir
)

torch.set_float32_matmul_precision('high')

def cnn_selection(option):
    if option == "ResNet18": return ResNet18()
    if option == "ResNet50": return ResNet50()
    if option == "CustomResNet": return CustomResNet20()
    if option == "SuperpixelCNN": return SuperpixelCNN(in_channels=3, out_channels=512)
    raise ValueError(f"Unknown CNN option {option}")

def extract_levelN_patch(slide_path, bbox, level):
    slide = openslide.OpenSlide(slide_path)
    x,y,w,h = bbox
    patch = slide.read_region((x,y), level, (w,h)).convert("RGB")
    slide.close()
    return patch

class LitNetwork(pl.LightningModule):
    def __init__(self, cnn_option, feature_dim=512):
        super().__init__()
        self.cnn = cnn_selection(cnn_option)
        self.transformer = TransformerEncoder(feature_dim, 1)
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        # now supply task for metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc   = Accuracy(task='binary')
        self.train_auc = AUROC(task='binary')
        self.val_auc   = AUROC(task='binary')

        self.pos_proj = torch.nn.Sequential(
            torch.nn.Linear(2, feature_dim), torch.nn.ReLU(), torch.nn.Linear(feature_dim, feature_dim)
        )
        self.patch_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.test_results = {}

    def forward(self, tokens): return self.transformer(tokens)

    def _step(self, sp_map, patch_tensor, y_label, slide_path, level, x0, y0, polys=None):
        y_true = y_label.to(self.device).view(-1)
        lab_np  = sp_map.squeeze(0).cpu().numpy().astype(int)
        regions = regionprops(lab_np)
        _, Hc, Wc = sp_map.shape
        all_tokens = []
        for r in regions:
            minr,minc,maxr,maxc = r.bbox
            t = patch_tensor[:,minr:maxr,minc:maxc].unsqueeze(0).to(self.device)
            if torch.mean(t).item() < 0.05: continue
            fmap = self.cnn(t)
            ds = F.adaptive_avg_pool2d(fmap, (8,8))
            C,pH,pW = ds.shape[1:]
            tok = ds.view(1,C,pH*pW).permute(0,2,1)
            cy,cx = r.centroid
            norm = torch.tensor([cx/Wc,cy/Hc],device=self.device).unsqueeze(0)
            pos  = self.pos_proj(norm)
            all_tokens.append(tok+pos.unsqueeze(1))
        if not all_tokens: return None
        seq    = torch.cat(all_tokens,dim=1)
        logits = self(seq).squeeze(-1)
        return logits, y_true

    def training_step(self,batch,batch_idx):
        logits,y_true = self._step(*batch)
        loss  = self.loss_func(logits,y_true); probs=torch.sigmoid(logits)
        self.log('train_loss',loss,on_epoch=True)
        self.log('train_acc', self.train_acc(probs,y_true.int()),on_epoch=True)
        self.log('train_auc', self.train_auc(probs,y_true.int()),on_epoch=True)
        return loss

    def validation_step(self,batch,batch_idx):
        logits,y_true = self._step(*batch)
        loss  = self.loss_func(logits,y_true); probs=torch.sigmoid(logits)
        self.log('val_loss', loss, on_epoch=True,prog_bar=True)
        self.log('val_acc',  self.val_acc(probs,y_true.int()),on_epoch=True,prog_bar=True)
        self.log('val_auc',  self.val_auc(probs,y_true.int()),on_epoch=True,prog_bar=True)

    def test_step(self,batch,batch_idx):
        sp_map,patch_tensor,y_label,slide_path,lvl,x0,y0,polys = batch
        logits,_ = self._step(sp_map,patch_tensor,y_label,slide_path,lvl,x0,y0,polys)
        probs = torch.sigmoid(logits).cpu().numpy()
        lab_np = sp_map.squeeze(0).cpu().numpy(); props=regionprops(lab_np)
        cents  = np.array([r.centroid for r in props])
        slide  = openslide.OpenSlide(slide_path); full_w,full_h=slide.level_dimensions[0]
        th_h,th_w = lab_np.shape; sx,sy=full_w/th_w,full_h/th_h
        dets=[]
        for p,(cy,cx) in zip(probs,cents): dets.append((float(p),x0+cx*sx,y0+cy*sy))
        self.test_results[slide_path]={'label':float(y_label),'slide_score':float(probs.max()),'detections':dets,'polygons':polys}

    def test_epoch_end(self,outputs): pass
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(),lr=learning_rate,weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=num_epochs)
        return {'optimizer':opt,'lr_scheduler':{'scheduler':sched,'monitor':'val_auc'}}

if __name__=='__main__':
    train_loader,val_loader=load_camelyon(root_dir=wsi_dir,annotation_dir=ann_dir,thumbnail_size=(2048,2048),num_segments=50,batch_size=1,val_split=0.2,num_workers=4)
    batch=next(iter(train_loader))
    sp_map,patch_tensor,rs,paths,lvls,x0_cs,y0_cs,polys=batch
    path, lvl, x0_c, y0_c = paths, int(lvls), int(x0_cs), int(y0_cs)
    slide=openslide.OpenSlide(path)
    _,Hc,Wc=sp_map.shape
    crop_img=extract_levelN_patch(path,(x0_c,y0_c,Wc,Hc),lvl)
    tc_np=np.array(crop_img)
    sp_map_np=sp_map.squeeze().cpu().numpy().astype(int)
    fig,ax=plt.subplots(figsize=(6,6)); ax.imshow(mark_boundaries(tc_np,sp_map_np)); ax.set_title(os.path.basename(path)); ax.axis('off'); plt.show()
    model=LitNetwork(cnn_option)
    logger=pl_loggers.TensorBoardLogger('../logs/camelyon16_cls')
    cbs=[]
    if use_checkpoint: cbs.append(pl.callbacks.ModelCheckpoint(monitor='val_auc',mode='max',save_top_k=1,filename='best-{epoch:02d}-{val_auc:.4f}'))
    trainer=pl.Trainer(max_epochs=num_epochs,accelerator='gpu' if torch.cuda.is_available() else 'cpu',callbacks=cbs,logger=logger)
    trainer.fit(model,train_loader,val_loader)
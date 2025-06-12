import os, glob

root      = 'imagenet_root'
val_dir   = os.path.join(root, 'val')
gt_txt    = os.path.join(root, 'ILSVRC2012_validation_ground_truth.txt')
paths     = sorted(glob.glob(os.path.join(val_dir, '*.JPEG')))

# read GT (one 1–1000 label per line)
with open(gt_txt) as f:
    gt = [int(l.strip())-1 for l in f]

# get the train-side synset list
from torchvision.datasets import ImageNet
# assume you’ve already pointed ImageNet(root, split='train') at the same root
train_ds     = ImageNet(root=root, split='train')
wnids        = train_ds.classes  # list of 1000 synset strings

# now move each file into its synset subfolder
for img_path, label in zip(paths, gt):
    wnid      = wnids[label]
    dest_dir  = os.path.join(val_dir, wnid)
    os.makedirs(dest_dir, exist_ok=True)
    os.rename(img_path, os.path.join(dest_dir, os.path.basename(img_path)))
import torch


class SuperpixelDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data  # The data is a list of tuples (superpixel_vectors, image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        superpixel_vectors, image = self.data[idx]
        return superpixel_vectors, image


class DataLoader(dataset):
    def __init__(self, image_paths, labels, transform=none):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image_path = self.image_paths[item]
        image = Image.open(image_path)

        transform_img = self.transform(image)

        label = self.labels[item]

        return transform_img, label




dataset = 'blah'

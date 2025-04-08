from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class cifar10(Dataset):
    def __init__(self, split):
        super(cifar10, self).__init__()
        self.split = split
        self.data = load_dataset("uoft-cs/cifar10")[split]
        self.images, self.labels = [], []
        for sample in self.data:
            self.images.append(ToTensor()(sample["img"]))
            self.labels.append(sample["label"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"img": self.images[idx], "label": self.labels[idx]}

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms import v2


class cifar10(Dataset):
    def __init__(self, split):
        super(cifar10, self).__init__()
        self.split = split
        self.data = load_dataset("uoft-cs/cifar10")[split]
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])
        train_transforms = v2.Compose(
            [
                v2.PILToTensor(),
                v2.RandomCrop(size=32, padding=4),
                v2.RandomHorizontalFlip(0.5),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )

        test_transforms = v2.Compose(
            [
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self.transform = train_transforms if split == "train" else test_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "img": self.transform(self.data[idx]["img"]),
            "label": self.data[idx]["label"],
        }

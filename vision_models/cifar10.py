import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import v2

class cifar10(Dataset):
    def __init__(self,split):
        super(cifar10,self).__init__()
        self.split = split
        self.data = load_dataset("uoft-cs/cifar10")[split]
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])
        self.train_transforms = v2.Compose([
                                            v2.RandomCrop(size=(32,32),padding=4),
                                            v2.RandomHorizontalFlip(0.5),
                                            v2.Normalize(mean=self.mean,std=self.std),
                                            ])

        self.test_transforms = v2.Compose([v2.Normalize(mean=self.mean,std=self.std)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img = pil_to_tensor(self.data[idx]["img"])/255
        if self.split == "train":
            img = self.train_transforms(img)
        else:
            img = self.test_transforms(img)
        return {
            "img": img,
            "label": self.data[idx]["label"]
        }       
        
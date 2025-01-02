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
        self.mean = torch.tensor([125.3067, 122.9510, 113.8656])/255
        # self.calculate_per_pixel_mean()
        self.train_transforms = v2.Compose([v2.ToDtype(torch.float32),
                                            v2.Normalize(mean=self.mean,std=[1,1,1]),
                                            v2.Pad(4),
                                            v2.RandomHorizontalFlip(0.5),
                                            v2.RandomResizedCrop(size=(32, 32), antialias=True)])

        self.test_transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True),
                                           v2.Normalize(mean=self.mean,std=[1,1,1])])
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        img = pil_to_tensor(self.data[idx]["img"]) 
        if self.split == "train":
            img = self.train_transforms(img)/255
        else:
            img = self.test_transforms(img)/255
        return {
            "img": img,
            "label": self.data[idx]["label"]
        }

    def get_mean(self):
        return self.mean

    def calculate_per_pixel_mean(self):
        self.mean = torch.zeros(3)
        for item in self.data:
            self.mean += torch.sum(pil_to_tensor(item["img"]),dim=(1,2))
        self.mean /= len(self.data) * 32 * 32
        
        
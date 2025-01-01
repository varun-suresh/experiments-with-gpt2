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
        self.calculate_mean_image()
        self.transforms = v2.Compose([v2.Pad(4),
            v2.RandomHorizontalFlip(0.5),
            v2.RandomResizedCrop(size=(32, 32), antialias=True)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        # TODO: Save the train data mean image and use it for test
        img = pil_to_tensor(self.data[idx]["img"]) - self.mean_img
        if self.split == "train":
            img = self.transforms(img)
        return {
            "img": img,
            "label": self.data[idx]["label"]
        }

    def get_mean_image(self):
        return self.mean_img

    def calculate_mean_image(self):
        self.mean_img = torch.zeros((3,32,32))
        for item in self.data:
            self.mean_img += pil_to_tensor(item["img"])
        self.mean_img /= len(self.data)
        
        
"""
Evaluate ResNet on CIFAR10
"""

from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from cifar10 import cifar10
from resnet import ResNetCifar
from resnet_config import ResNetCIFAR10Config, ResNetTestConfig


class Eval:
    def __init__(
        self,
        test_set: cifar10,
        eval_config: ResNetTestConfig,
        model_config: ResNetCIFAR10Config,
    ):
        self.test_set = test_set
        self.eval_config = eval_config
        self.model_config = model_config
        self.load_model()

    def load_model(self):
        ckpt = torch.load(
            self.eval_config.checkpoint_path, map_location=self.eval_config.device
        )
        model_config = ResNetCIFAR10Config(**ckpt["model_config"])
        model_config.load_from_checkpoint = self.model_config.load_from_checkpoint
        model_config.checkpoint_path = self.model_config.checkpoint_path
        self.model_config = model_config
        self.model = ResNetCifar(self.model_config.n)
        self.model.load_state_dict(ckpt["model"])

        self.model.to(self.eval_config.device)
        if self.eval_config.compile:
            self.model = torch.compile(self.model)
        self.model.eval()

    def evaluate(self):
        if self.eval_config.subset:
            subset_range = torch.arange(
                0, len(self.test_set), self.eval_config.interval
            )
            dl = DataLoader(
                torch.utils.data.Subset(self.test_set, subset_range),
                batch_size=self.eval_config.batch_size,
            )
        else:
            dl = DataLoader(self.test_set, batch_size=self.eval_config.batch_size)
        correct = 0
        for batch in tqdm(dl):
            with torch.no_grad():
                logits = self.model(batch["img"].to(self.eval_config.device))
                predictions = torch.argmax(logits, dim=1)
                labels = torch.tensor(batch["label"]).to(self.eval_config.device)
                correct += torch.eq(predictions, labels).sum()
        print(
            f"Accuracy: {correct / len(self.test_set)}\nError: {1 - correct / len(self.test_set)}"
        )

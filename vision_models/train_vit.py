import click
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.transforms import v2, ToTensor

from lib.baseTrainer import BaseTrainer
from lib.baseScheduler import CosineSchedulerWithWarmup
from lib.baseEval import BaseEval

from vision_models.cifar10 import cifar10
from vision_models.vit_config import VitTrainConfig, VitTestConfig


class Trainer(BaseTrainer):
    def __init__(self, config, train_set, val_set, test_set, criterion):
        super(Trainer, self).__init__(config, train_set, val_set, test_set, criterion)

    def create_dataloader(self, split="train"):
        train_transforms, test_transforms = create_train_test_transforms()
        if split == "train":
            transforms = train_transforms
            dataset = self.train_set
        elif split == "val":
            transforms = test_transforms
            dataset = self.val_set
        else:
            transforms = test_transforms
            dataset = self.test_set

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.micro_batch_size,
            shuffle=True,
            collate_fn=self.make_collate_fn(transforms),
        )
        return dataloader

    def make_collate_fn(self, transforms):
        def collate_fn(batch):
            imgs = torch.stack([item["img"] for item in batch])  # (B, C, H, W), uint8
            labels = torch.tensor([item["label"] for item in batch])
            imgs = imgs.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)

            imgs = transforms(imgs)  # Now apply batch-wise GPU transforms

            return {"img": imgs, "label": labels}

        return collate_fn

    def freeze_layers(self):
        print("Not implemented yet")

    def run_inference(self, batch):
        return self.model(batch["img"].to(self.config.device))

    def load_optimizer_scheduler(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
        self.scheduler = CosineSchedulerWithWarmup(
            self.optimizer,
            lr=self.config.learning_rate,
            min_lr=self.config.min_lr,
            decay_iters=self.config.decay_iters,
            warmup_iters=self.config.warmup_iters,
        )
        if self.config.init_from == "resume":
            self.optimizer.load_state_dict(self.ckpt["optimizer"])
            self.scheduler.load_state_dict(self.ckpt["scheduler"])
        self.optimizer.zero_grad()


class Eval(BaseEval):
    def __init__(self, test_set, test_config):
        super(Eval, self).__init__(test_set, test_config)


def create_train_test_transforms():
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    train_transforms = v2.Compose(
        [
            v2.RandomCrop(size=32, padding=4),
            v2.RandomHorizontalFlip(0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )

    test_transforms = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    return train_transforms, test_transforms


@click.command()
@click.option("--out", default="out/vit", help="Output directory")
def train(out):
    all_train_data = cifar10("train")
    train_size = int(0.9 * len(all_train_data))
    val_size = len(all_train_data) - train_size
    train_set, val_set = random_split(all_train_data, [train_size, val_size])
    test_set = cifar10("test")
    test_set = random_split(test_set, [len(test_set)])[0]
    config = VitTrainConfig(out_dir=out)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    trainer = Trainer(config, train_set, val_set, test_set, criterion)
    trainer.train()


def eval():
    test_set = cifar10("test")
    test_config = VitTestConfig()
    evaluator = Eval(test_set, test_config)
    evaluator.evaluate()


if __name__ == "__main__":
    train()
    # eval()

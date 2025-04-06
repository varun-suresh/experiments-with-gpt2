"""
Base Trainer Class with default methods to
1. Load model
2. Load scheduler and optimizer
3. Run a training job
4. Calculate train and test error
"""

import os
from abc import ABC, abstractmethod
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
from lib.baseScheduler import LRSchedulerWithWarmup
from lib.model_lib import AvailableModels

available_models = AvailableModels()


class BaseTrainer(ABC):
    def __init__(self, config, train_set, val_set, test_set, criterion):
        self.config = config
        self.train_set = train_set
        self.train_dataloader = self.create_dataloader(train_set)
        self.val_set = val_set
        self.val_dataloader = self.create_dataloader(val_set)
        self.test_set = test_set
        self.test_dataloader = self.create_dataloader(test_set)
        self.criterion = criterion
        self.writer = SummaryWriter(log_dir=config.out_dir)
        self.load_model()
        self.load_optimizer_scheduler()

    def load_model(self):
        """
        Loads the model based on the config
        """
        model_def, model_config = available_models.get(self.config.model_type)
        if self.config.init_from == "resume":
            ckpt_path = os.path.join(self.config.out_dir, self.config.checkpoint_name)
            print(f"Loading model from {ckpt_path}")
            self.ckpt = torch.load(ckpt_path, map_location=self.config.device)
            self.model_config = model_config(**self.ckpt["model_config"])
            self.model_config.load_from_checkpoint = True
            self.start_epoch = self.ckpt["epoch"]
            self.best_val_error = self.ckpt["best_val_error"]
            # Update some params
            # model_config.load_from_checkpoint = model_config.load_from_checkpoint
            # model_config.checkpoint_path = self.model_config.checkpoint_path
            # self.model_config = model_config
            self.model = model_def(self.model_config)
            self.model.load_state_dict(self.ckpt["model"])
        else:
            self.model_config = model_config()
            self.model = model_def(self.model_config)
            self.start_epoch = 0
            self.best_val_loss = 1e9
        self.model.to(self.config.device)
        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Loaded Model.\nNumber of parameters = {n_parameters}")
        if self.config.freeze_layers > 0:
            self.freeze_layers(self.config.freeze_layers)

    def load_optimizer_scheduler(self):
        """
        By default, uses a AdamW optimizer
        """
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )
        self.scheduler = LRSchedulerWithWarmup(
            self.optimizer,
            lr=self.config.learning_rate,
            step_size=self.config.step_size,
            gamma=0.1,
            warmup_iters=self.config.warmup_iters,
        )
        if self.config.init_from == "resume":
            self.optimizer.load_state_dict(self.ckpt["optimizer"])
            self.scheduler.load_state_dict(self.ckpt["scheduler"])
        self.optimizer.zero_grad()

    @abstractmethod
    def freeze_layers(self):
        """
        This method needs to be implemented in the train class for a particular model
        """
        pass

    @abstractmethod
    def create_dataloader(self, dataset, batch_size, collate_fn=None, shuffle=True):
        """
        Should return a dataloader. The dataloader must have a field label
        """
        pass

    @abstractmethod
    def run_inference(self):
        """
        self.model(whatever inputs are needed for this particular model)
        """
        pass

    def log_epoch_info(self, epoch, loss, accuracy):
        print(
            f"Epoch: {epoch}\nTrain Loss: {loss['train']}\nValidation Loss:{loss['val']}"
        )
        print(
            f"Train Accuracy: {accuracy['train']}\nValidation Accuracy: {accuracy['val']}\nTest Accuracy: {accuracy['test']}"
        )

        self.writer.add_scalars(
            "Loss", {split: loss for split, loss in loss.items()}, epoch
        )
        self.writer.add_scalars(
            "Accuracy", {split: error for split, error in accuracy.items()}, epoch
        )
        self.writer.add_scalar("Learning Rate", self.scheduler.get_last_lr()[0], epoch)

    def save_model(self, epoch):
        ckpt = {
            "model": self.model.state_dict(),
            "config": asdict(self.config),
            "model_config": asdict(self.model_config),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
        }
        output_path = os.path.join(self.config.out_dir, self.config.checkpoint_name)
        print(f"Saving checkpoint to {output_path}")
        if not os.path.exists(self.config.out_dir):
            os.makedirs(self.config.out_dir)
            torch.save(ckpt, output_path)

    def train(self):
        print("Training..")
        self.model.train()

        accumulation_steps = self.config.batch_size // self.config.micro_batch_size
        for epoch in tqdm(range(self.start_epoch, self.config.epochs)):
            for iter_num in tqdm(range(len(self.train_dataloader))):
                if self.config.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                batch = next(iter(self.train_dataloader))
                model_output = self.run_inference(batch)
                loss = self.criterion(
                    model_output, batch["label"].to(self.config.device)
                ) / (self.config.micro_batch_size * accumulation_steps)
                loss.backward()

                if iter_num % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.scheduler.step()

            loss, accuracy = self.calculate_loss_and_accuracy()
            self.log_epoch_info(epoch, loss, accuracy)
            if self.best_val_loss > loss["val"] or self.config.always_save_checkpoint:
                self.best_val_loss = loss["val"]
                self.save_model(epoch)

    def calculate_subset_loss_and_accuracy(self, dl):
        loss = 0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(dl):
                output = self.run_inference(batch)
                labels = batch["label"].to(self.config.device)
                loss += self.criterion(output, labels)
                predictions = torch.argmax(output, dim=-1)
                correct += torch.eq(predictions, labels).sum()
        loss /= len(dl.dataset)
        accuracy = correct / len(dl.dataset)
        return loss, accuracy

    def calculate_loss_and_accuracy(self):
        self.model.eval()
        loss, accuracy = {}, {}
        pairs = [
            ("train", self.train_dataloader),
            ("val", self.val_dataloader),
            ("test", self.test_dataloader),
        ]
        for split, subset in pairs:
            print(f"Evaluating {split} split")
            loss[split], accuracy[split] = self.calculate_subset_loss_and_accuracy(
                subset
            )

        return loss, accuracy

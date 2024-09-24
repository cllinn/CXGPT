import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 
import os 
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter


class XlipMultiGPUSerialTrainer:
    def __init__(
        self,
        epochs, 
        optimizer,
        model,
        dataloader,
        lr_scheduler,
        output_dir,
        log_freq=20,
        device=torch.device("cuda", 0),
        v_device=torch.device("cuda", 1),
        is_resume=True
    ):
        super().__init__()
        self.epochs = epochs 
        self.optimizer = optimizer
        self.model = model 
        self.dataloader = dataloader 
        self.lr_scheduler = lr_scheduler
        self.output_dir = output_dir
        self.log_freq = log_freq
        self.device = device
        self.v_device = v_device
        self.is_resume = is_resume
        # self.writer = SummaryWriter("xlip_pretrain")
    
    def save_checkpoint(self, current_epoch):
        if not os.path.isdir(self.output_dir + "/checkpoint"):
            os.mkdir(self.output_dir + "/checkpoint")
        else:
            checkpoint = {
                "model": self.model.cpu().state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": current_epoch,
                "lr_scheduler": self.lr_scheduler.state_dict()
            }
            file_to_remove = []
            for filename in os.listdir(self.output_dir + "/checkpoint"):
                if "checkpoint" in filename:
                    file_to_remove.append(filename)
            for f in file_to_remove:
                file_path = os.path.join(self.output_dir, "checkpoint", f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            torch.save(checkpoint, self.output_dir + "/checkpoint/" + f"checkpoint.pt-{current_epoch}")

    def save(self, current_epoch):
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        else:
            checkpoint = {
                "model": self.model.cpu().state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": current_epoch,
                "lr_scheduler": self.lr_scheduler.state_dict
            }
            torch.save(checkpoint, self.output_dir + "/model.pt")

    def train(self):
        self.model = self.model.to(self.device)
        # self.model.visual_encoder = self.model.visual_encoder.to(self.v_device)
        # self.model.ln_vision = self.model.ln_vision.to(self.v_device)

        start_epoch = -1
        if self.is_resume and os.path.isdir(os.path.join(self.output_dir, "checkpoint")) and len(os.listdir(os.path.join(self.output_dir, "checkpoint"))) != 0:
            file_name = os.listdir(os.path.join(self.output_dir, "checkpoint"))[-1]
            checkpoint = torch.load(os.path.join(self.output_dir, "checkpoint", file_name))
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            current_epoch = checkpoint["epoch"]
            start_epoch = current_epoch
        
        for current_epoch in tqdm(range(start_epoch + 1, self.epochs), desc="TRAIN LOOP"):

            self.model = self.model.to(self.device)
            self.model.visual_encoder = self.model.visual_encoder.to(self.v_device)
            self.model.ln_vision = self.model.ln_vision.to(self.v_device)

            for batch_idx, sample in tqdm(enumerate(self.dataloader), desc="EPOCH LOOP"):
                sample["image"] = sample["image"].to(self.device)
                output = self.model(sample)
                if batch_idx % self.log_freq == 0:
                    print(f"EPOCH: {current_epoch}, steps: {batch_idx}, itc_loss: {output.loss_itc.item()}, \
                    itm_loss: {output.loss_itm.item()}, lm_loss: {output.loss_lm.item()}")
                loss = output.loss 
                current_step = batch_idx + len(self.dataloader) * current_epoch
                # self.writer.add_scalar("loss/loss", loss, current_step)
                # self.writer.add_scalar("loss/loss_itc", output.loss_itc, current_step)
                # self.writer.add_scalar("loss/loss_itm", output.loss_itm, current_step)
                # self.writer.add_scalar("loss/loss_lm", output.loss_lm, current_step)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.lr_scheduler.step()
            self.save_checkpoint(current_epoch)
            print(f"EPOCH {current_epoch} is saved!")
        
        self.save(current_epoch)
    
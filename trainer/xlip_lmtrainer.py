import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 
import sys
sys.path.append("/root")
import os 
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
from mulcon.trainer.multiserial_trainer import XlipMultiGPUSerialTrainer
from rouge import FilesRouge, Rouge


class XlipLMTrainer(XlipMultiGPUSerialTrainer):
    def __init__(
        self,
        epochs, 
        optimizer,
        model,
        dataloader,
        val_loader,
        lr_scheduler,
        output_dir,
        log_freq=10,
        device=torch.device("cuda", 0),
        v_device=torch.device("cuda", 0),
        lm_device=torch.device("cuda", 0),
        is_resume=True
    ):
        super().__init__(epochs, optimizer, model, dataloader, lr_scheduler, \
                output_dir, log_freq, device, v_device, is_resume)
        self.writer = SummaryWriter("chestxraygpt4")
        self.lm_device = lm_device
        self.val_loader = val_loader

    def train(self):
        # self.model = self.model.to("cpu")
        # self.model.visual_encoder = self.model.visual_encoder.to(self.v_device)
        # self.model.ln_vision = self.model.ln_vision.to(self.v_device)
        # self.model.llm_model = self.model.llm_model.to(self.lm_device)
        # self.model.Qformer = self.model.Qformer.to(self.device)
        # self.model.query_tokens = nn.Parameter(self.model.query_tokens.to(self.device))
        # self.model.llm_proj = self.model.llm_proj.to(self.device)
        # self.model = self.model.to("cpu")
        self.model.train()

        start_epoch = -1
        if self.is_resume and os.path.isdir(os.path.join(self.output_dir, "checkpoint")) and len(os.listdir(os.path.join(self.output_dir, "checkpoint"))) != 0:
            file_name = os.listdir(os.path.join(self.output_dir, "checkpoint"))[-1]
            # print(str(os.path.join(self.output_dir, "checkpoint", file_name)))
            checkpoint = torch.load(os.path.join(self.output_dir, "checkpoint", file_name), map_location="cuda:0")
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            for param in self.optimizer.state.values():
                for k, v in param.items():
                    if isinstance(v, torch.Tensor):
                        # print(v.device)
                        v.data = v.data.to(self.device)
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            current_epoch = checkpoint["epoch"]
            start_epoch = current_epoch
            del checkpoint
            print("resume from checkpoint!")
        
        for current_epoch in tqdm(range(start_epoch + 1, self.epochs), desc="TRAIN LOOP"):

            self.model = self.model.to(self.device)
            # self.model.visual_encoder = self.model.visual_encoder.to(self.v_device)
            # self.model.ln_vision = self.model.ln_vision.to(self.v_device)

            for batch_idx, sample in tqdm(enumerate(self.dataloader), desc="EPOCH LOOP"):
                sample["image"] = sample["image"].to(self.device).float()
                output = self.model(sample)
                loss = output["loss"]
                if batch_idx % self.log_freq == 0:
                    print(f"EPOCH: {current_epoch}, steps: {batch_idx}, loss: {loss.cpu().item()}")
                # loss = output.loss 
                # current_step = batch_idx + len(self.dataloader) * current_epoch
                # self.writer.add_scalar("loss/loss", loss, current_step)
                self.optimizer.zero_grad()
                loss.backward()
                # print(self.model.device, loss.device)
                # self.model.llm_model = self.model.llm_model.to("cpu")
                self.optimizer.step()
                # self.model.llm_model = self.model.llm_model.to(self.lm_device)
                torch.cuda.empty_cache()
            
            self.lr_scheduler.step()
            
            # self.save_checkpoint(current_epoch)
            # print(f"EPOCH {current_epoch} is saved!")
            hyp_path = f"output/{current_epoch}_hyp.txt"
            ref_path = f"output/{current_epoch}_ref.txt"
            # checkpoint_path = f"/root/autodl-tmp/checkpoint/xlip-lm-fuxian/checkpoint/checkpoint.pt-{current_epoch}"
            # self.generate_eval_files(checkpoint_path, hyp_path, ref_path)
            self.generate_eval_files_without_ckpt(hyp_path, ref_path)
            # score = XlipLMTrainer.evaluate_by_lines(hyp_path, ref_path)
            # print(score)
            # with open("score.txt", "a") as f:
            #     f.write(str(current_epoch) + " : " + str(score) + "\n")
            # file_name = os.listdir(os.path.join(self.output_dir, "checkpoint"))[-1]
            # checkpoint_path = os.path.join(self.output_dir, "checkpoint", file_name)
            # self.generate_eval_files(checkpoint_path, hyp_path, ref_path)
            # score = XlipLMTrainer.evaluate(hyp_path, ref_path)
            # with open("score.txt", "a") as f:
            #     f.write(f"current epoch is {current_epoch} score is " + str(score) + "\n")
        # self.save(current_epoch)
    
    @torch.no_grad()
    def generate_eval_files(self, checkpoint_path, hyp_path, ref_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model"])
        del checkpoint
        self.model = self.model.to(self.device)
        self.model.eval()
        hyp = open(hyp_path, "w")
        ref = open(ref_path, "w")
        for batch_idx, sample in tqdm(enumerate(self.val_loader), desc="EVAL LOOP"):
            sample["image"] = sample["image"].to(self.device).float()
            output = self.model.generate(sample, max_length=32)
            for i in output:
                hyp.write(i + "\n")
            for i in sample["text_output"]:
                ref.write(i + "\n")
        hyp.close()
        ref.close()
    
    @torch.no_grad()
    def generate_eval_files_without_ckpt(self, hyp_path, ref_path):
        self.model = self.model.to(self.device)
        self.model.eval()
        hyp = open(hyp_path, "w")
        ref = open(ref_path, "w")
        for batch_idx, sample in tqdm(enumerate(self.val_loader), desc="EVAL LOOP"):
            sample["image"] = sample["image"].to(self.device).float()
            output = self.model.generate(sample, max_length=32)
            for i in output:
                hyp.write(i + "\n")
            for i in sample["text_output"]:
                ref.write(i + "\n")
        hyp.close()
        ref.close()
    
    @classmethod
    def evaluate(cls, hyp_path, ref_path):
        files_rouge = FilesRouge()
        scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
        return scores
   
    @classmethod
    def evaluate_by_lines(cls, hyp_path, ref_path):
        rouge = Rouge()
        hyp = open(hyp_path, "r")
        ref = open(ref_path, "r")
        hyp_lines = [line[:-1] for line in hyp if line[:-1]]
        # print(hyp_lines[1])
        n_lines = len(hyp_lines)
        ref_lines = [line[:-1] for line in ref if line[:-1]]
        a, b = [], []
        for i, line in enumerate(hyp_lines):
            if line[:-1] and len(line[:-1]) >= 2:
                a.append(line)
                b.append(ref_lines[i])
        
        scores = rouge.get_scores(a, b, avg=False)
        sum = 0.0
        for score in scores:
            sum += score["rouge-l"]["f"]
        return sum / n_lines
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 
import os 
from tqdm import tqdm 
from rouge import FilesRouge


def concat_text_input_output(input_ids, input_atts, output_ids, output_atts):
    input_part_targets_len = []
    llm_tokens = {"input_ids": [], "attention_mask": []}
    for i in range(input_ids.size(0)):
        this_input_ones = input_atts[i].sum()
        input_part_targets_len.append(this_input_ones)
        llm_tokens['input_ids'].append(
            torch.cat([
                input_ids[i][:this_input_ones],
                output_ids[i][1:],
                input_ids[i][this_input_ones:]
            ])
        )
        llm_tokens['attention_mask'].append(
            torch.cat([
                input_atts[i][:this_input_ones],
                output_atts[i][1:],
                input_atts[i][this_input_ones:]
            ])
        )
    llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
    llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
    return llm_tokens, input_part_targets_len

class LMTrainer:
    def __init__(
        self,
        epochs, 
        optimizer,
        model,
        tokenizer,
        dataloader,
        val_loader,
        lr_scheduler,
        output_dir,
        log_freq=10,
        device=torch.device("cuda", 0),
        is_resume=True,
        max_txt_len=100,
        max_output_len=50
    ):
        super().__init__()
        self.epochs = epochs 
        self.optimizer = optimizer
        self.model = model 
        self.tokenizer = tokenizer
        self.dataloader = dataloader 
        self.val_loader = val_loader
        self.lr_scheduler = lr_scheduler
        self.output_dir = output_dir
        self.log_freq = log_freq
        self.device = device
        self.is_resume = is_resume
        self.max_txt_len = max_txt_len
        self.max_output_len = max_output_len
    
    def save_checkpoint(self, current_epoch):
        if not os.path.isdir(self.output_dir + "/checkpoint"):
            os.mkdir(self.output_dir + "/checkpoint")
        
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
            p_bar = tqdm(enumerate(self.dataloader), desc="EPOCH LOOP", total=len(self.dataloader))
            for batch_idx, samples in p_bar:
                # sample["image"] = sample["image"].to(self.device)
                self.tokenizer.padding_side = "right"
                self.tokenizer.truncation_side = 'left'
                text_input_tokens = self.tokenizer(
                    samples['text_input'],
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_txt_len,
                ).to(self.device)
                text_output_tokens = self.tokenizer(
                    [t + self.tokenizer.eos_token for t in samples['text_output']],
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.max_output_len,
                ).to(self.device)
                llm_tokens, input_part_targets_len = concat_text_input_output(
                    text_input_tokens.input_ids,
                    text_input_tokens.attention_mask,
                    text_output_tokens.input_ids,
                    text_output_tokens.attention_mask,
                )
                targets = llm_tokens['input_ids'].masked_fill(
                    llm_tokens['input_ids'] == self.tokenizer.pad_token_id, -100
                )
                # do not apply loss to the text input (i.e., instruction)
                for i, l in enumerate(input_part_targets_len):
                    targets[i][:l] = -100
                outputs = self.model(
                    input_ids=llm_tokens["input_ids"],
                    attention_mask=llm_tokens["attention_mask"],
                    return_dict=True,
                    labels=targets,
                )
                if batch_idx % self.log_freq == 0:
                    # print(f"EPOCH: {current_epoch}, steps: {batch_idx}, loss: {outputs.loss.item()}")
                    p_bar.write(f"EPOCH: {current_epoch}, steps: {batch_idx}, loss: {outputs.loss.item()}")
                loss = outputs.loss 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            self.lr_scheduler.step()
            # self.save_checkpoint(current_epoch)
            print(f"EPOCH {current_epoch} is saved!")
            if True:
                hyp_path = f"output/{current_epoch}_hyp.txt"
                ref_path = f"output/{current_epoch}_ref.txt"
                checkpoint_path = f"/root/autodl-tmp/checkpoint/mimic-biogptlft/checkpoint/checkpoint.pt-{current_epoch}"
                self.generate_eval_files(checkpoint_path, hyp_path, ref_path)
        
        # self.save(current_epoch)
        
    @torch.no_grad()    
    def generate_eval_files(self, checkpoint_path, hyp_path, ref_path):
        # checkpoint = torch.load(checkpoint_path)
        # self.model.load_state_dict(checkpoint["model"])
        # del checkpoint 
        self.model = self.model.to(self.device)
        self.model.eval()
        hyp = open(hyp_path, "w")
        ref = open(ref_path, "w")
        for batch_idx, samples in tqdm(enumerate(self.val_loader), desc="EVAL LOOP"):
            self.tokenizer.padding_side = "left"
            self.tokenizer.truncation_side = 'left'
            text_input_tokens = self.tokenizer(
                samples['text_input'],
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(self.device)
            # print(text_input_tokens)
            input_ids = text_input_tokens.input_ids
            use_nucleus_sampling=False
            num_beams=5
            max_length=256
            min_length=1
            top_p=0.9
            repetition_penalty=1.5
            length_penalty=1
            num_captions=1
            temperature=1
            outputs = self.model.generate(
                input_ids=text_input_tokens.input_ids,
                attention_mask=text_input_tokens.attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_length=max_length,
                min_length=min_length,
                # eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            # outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
            output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_text = [text.strip() for text in output_text]
            for i in output_text:
                hyp.write(i.split("Impression:")[1] + "\n")
            for i in samples["text_output"]:
                ref.write(i + "\n")
        hyp.close()
        ref.close()
        
    @classmethod
    def evaluate(cls, hyp_path, ref_path):
        files_rouge = FilesRouge()
        scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
        return scores
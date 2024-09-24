import torch
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import AutoImageProcessor, ResNetModel
from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

PRETRAIN_MODEL_DICT = {
    "cnn": "/root/autodl-tmp/checkpoint/resnet50",
    "qformer": "/root/autodl-tmp/checkpoint/bert-base-uncased",
    "llm": "/root/autodl-tmp/checkpoint/biogpt-large-pubmedqa"
}

CONFIG_DICT = {
    "visual_dim": 2048,
    "qformer_dim": 768
}

class CNNLLM(nn.Module):
    def __init__(self, max_output_txt_len=100, max_txt_len=100):
        super().__init__()
        
        self.visual_encoder = ResNetModel.from_pretrained(PRETRAIN_MODEL_DICT["cnn"])
        # self.visual_proj = nn.Linear(CONFIG_DICT["visual_dim"], CONFIG_DICT["qformer_dim"])
        # self.qformer = BertModel.from_pretrained(PRETRAIN_MODEL_DICT["qformer"])
        # self.tokenizer = BertTokenizer.from_pretrained(PRETRAIN_MODEL_DICT["qformer"])
        self.llm_model = AutoModelForCausalLM.from_pretrained(PRETRAIN_MODEL_DICT["llm"])
        self.llm_tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_MODEL_DICT["llm"])
        self.llm_proj = nn.Linear(CONFIG_DICT["visual_dim"], self.llm_model.config.hidden_size)
        
        self.max_output_txt_len = max_output_txt_len
        self.max_txt_len = max_txt_len
    
    def forward(self, samples):
        image = samples["image"]
        text_input = samples["text_input"]
        text_output = samples["text_output"]
        # image = self.image_processor(image, return_tensors="pt")
        image_embeds = self.visual_encoder(image, return_dict=True).last_hidden_state
        B, C, H, W = image_embeds.size()
        image_embeds = image_embeds.permute(0, 2, 3, 1).view(B, H * W, C)
        inputs_llm = self.llm_proj(image_embeds)
        image_mask = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
        # qformer_cls = self.tokenizer([self.tokenizer.cls_token], add_special_tokens=False, return_tensors="pt")
        # # print(qformer_cls)
        # qformer_cls_embeds = self.qformer.get_input_embeddings()(qformer_cls.input_ids.to(image.device))
        # inputs_qformer = torch.cat([qformer_cls_embeds.tile(image_embeds.size(0), 1, 1), image_embeds], dim=1)
        # image_mask = torch.ones(inputs_qformer.size()[:-1], dtype=torch.long).to(image.device)
        # inputs_llm = self.qformer(
        #     inputs_embeds=inputs_qformer,
        #     attention_mask=image_mask,
        #     return_dict=True
        # )
        # inputs_llm = inputs_llm.last_hidden_state
        # inputs_llm = self.llm_proj(inputs_llm)
        
        # text_input = [self.llm_tokenizer.bos_token + text_input] * B
        text_input = [self.llm_tokenizer.bos_token + i for i in text_input]
        print(f"text_input = {text_input}")
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            text_input,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)
        
        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)
        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )
        # print("llm_tokens[0] ", self.llm_tokenizer.decode(llm_tokens["input_ids"][0]))
        
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100
        # print("targets:", targets[0])
        empty_targets = (
            torch.ones(image_mask.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        # print(empty_targets.size(), targets.size())
        targets = torch.cat([empty_targets, targets], dim=1)
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids']).to(image.device)
        # print(inputs_embeds.size())
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        # print(inputs_embeds.size())
        attention_mask = torch.cat([image_mask, llm_tokens['attention_mask']], dim=1)
        outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )

        loss = outputs.loss

        return {"loss": loss}       
    
    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.llm_tokenizer.padding_side = "left"
        image = samples["image"]
        text_input = samples["text_input"]
        image_embeds = self.visual_encoder(image, return_dict=True).last_hidden_state
        B, C, H, W = image_embeds.size()
        image_embeds = image_embeds.permute(0, 2, 3, 1).view(B, H * W, C)
        inputs_llm = self.llm_proj(image_embeds)
        image_mask = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
        # image_embeds = self.visual_proj(image_embeds)
        # qformer_cls = self.tokenizer([self.tokenizer.cls_token], add_special_tokens=False, return_tensors="pt")
        # # print(qformer_cls)
        # qformer_cls_embeds = self.qformer.get_input_embeddings()(qformer_cls.input_ids.to(image.device))
        # inputs_qformer = torch.cat([qformer_cls_embeds.tile(image_embeds.size(0), 1, 1), image_embeds], dim=1)
        # image_mask = torch.ones(inputs_qformer.size()[:-1], dtype=torch.long).to(image.device)
        # inputs_llm = self.qformer(
        #     inputs_embeds=inputs_qformer,
        #     attention_mask=image_mask,
        #     return_dict=True
        # )
        # inputs_llm = inputs_llm.last_hidden_state
        # inputs_llm = self.llm_proj(inputs_llm)
        
        # text_input = [self.llm_tokenizer.bos_token] * B
        text_input = [self.llm_tokenizer.bos_token + i for i in text_input]
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        llm_tokens = self.llm_tokenizer(
            text_input,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(image.device)
        
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids']).to(image.device)
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([image_mask, llm_tokens['attention_mask']], dim=1)

        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
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

        outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]

        return output_text
        
    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
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

    
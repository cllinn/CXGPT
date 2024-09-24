import torch
import torch.nn as nn 
import torch.nn.functional as F 
from mulcon.model.xlip import (
    Blip2Base,
    compute_sim_matrix
)
from mulcon.model.xlip_outputs import BlipOutput, BlipOutputFeatures
from mulcon.model.xlip_qformer import Blip2Qformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from mulcon.model.xlip_model import XlipModel 
from mulcon.model.xlip_qformer import disabled_train
import logging 


class XlipLM(Blip2Base):
    def __init__(
        self,
        cvt_config,
        encoder_config,
        config,
        llm_model="/root/autodl-tmp/checkpoint/biogpt-large-pubmedqa",
        prompt="",
        xlip_model_path="/root/autodl-tmp/checkpoint/xlip-base/checkpoint/checkpoint.pt-19",
        freeze_cvt=False,
        num_query_token=25,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=100,
        max_output_txt_len=100,
        qformer_text_input=True
    ):
        super().__init__()
        xlip_model = XlipModel(cvt_config, encoder_config, config)
        # checkpoint = torch.load(xlip_model_path, map_location="cuda:0")
        # xlip_model.load_state_dict(checkpoint["model"])
        # print("load finished")
        # del checkpoint
        
        self.tokenizer = self.init_tokenizer(truncation_side="left")

        # self.visual_encoder, self.ln_vision = self.init_vision_encoder(cvt_config)
        self.visual_encoder, self.ln_vision = xlip_model.visual_encoder, xlip_model.ln_vision
        self.cvt_proj = xlip_model.cvt_proj
        if freeze_cvt:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.ln_vision.requires_grad = False 
            # self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # self.Qformer, self.query_tokens = self.init_Qformer(
        #     num_query_token, self.visual_encoder.config.embed_dim[2], encoder_config, cross_attention_freq
        # )
        self.Qformer, self.query_tokens = xlip_model.Qformer, xlip_model.query_tokens
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None
        # freeze the Qformer and query_tokens
#         for name, param in self.Qformer.named_parameters():
#             param.requires_grad = False
        
#         self.query_tokens.requires_grad = False

        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float32
        )
        # test if not load finetune model state_dict
        self.llm_model.load_state_dict(torch.load("/root/autodl-tmp/checkpoint/mimic-biogptlft/checkpoint/checkpoint.pt-4", map_location="cpu")["model"])
        # print("llm load finished")
        # self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        # self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        # self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        # self.llm_tokenizer.pad_token = self.llm_tokenizer.unk_token

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # self.eos_token_id = self.llm_tokenizer(
        #     self.llm_tokenizer.eos_token, add_special_tokens=False
        # ).input_ids[0]

        # do not freeze llm model
        # for name, param in self.llm_model.named_parameters():
        #     param.requires_grad = False

        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llm_model.config.hidden_size
        )
        # self.llm_proj = nn.Sequential(
        #     nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
        # )

        self.max_txt_len = max_txt_len
        self.max_output_txt_len = max_output_txt_len
        self.prompt = prompt
        prompt_tokens = self.llm_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

        self._lemmatizer = None

        self.qformer_text_input = qformer_text_input
    
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

    def vision_encode(self, image):
        if image.device == self.visual_encoder.device:
            image_output = self.visual_encoder(image)
            last_hidden_state = image_output.last_hidden_state
            batch_size, embed_dim, _, _ = last_hidden_state.shape 
            cls_token_value = image_output.cls_token_value
            image_embeds = torch.cat([
                cls_token_value,
                last_hidden_state.permute(0, 2, 3, 1).view(batch_size, -1, embed_dim)
                ], dim=1)
            image_embeds = self.cvt_proj(image_embeds)
            image_embeds = self.ln_vision(image_embeds)
            return image_embeds 
        else:
            orig_device = image.device 
            image = image.to(self.visual_encoder.device)
            image_output = self.visual_encoder(image)
            last_hidden_state = image_output.last_hidden_state
            batch_size, embed_dim, _, _ = last_hidden_state.shape 
            cls_token_value = image_output.cls_token_value
            image_embeds = torch.cat([
                cls_token_value,
                last_hidden_state.permute(0, 2, 3, 1).view(batch_size, -1, embed_dim)
                ], dim=1)
            image_embeds = self.cvt_proj(image_embeds)
            image_embeds = self.ln_vision(image_embeds)
            return image_embeds.to(orig_device)
    
    def checknan(self, a):
        return torch.any(torch.isnan(a))

    def forward(self, samples):
        # print('-----------------')
        # print(samples["text_input"])
        # print(samples["text_output"])
        # print('-----------------')
        image = samples["image"]
        # with self.maybe_autocast():
            # image_embeds = self.ln_vision(self.visual_encoder(image))
        image_embeds = self.vision_encode(image)
            # if self.checknan(image_embeds):
            #     print("image_embeds is nan")
        
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        bs = image.size(0)
        
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # if self.checknan(query_tokens):
        #     print("query_tokens is nan")
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                samples["text_input"],
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids.to(self.Qformer.device),
                attention_mask=Qformer_atts.to(self.Qformer.device),
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds.to(self.Qformer.device),
                encoder_attention_mask=image_atts.to(self.Qformer.device),
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens.float(),
                encoder_hidden_states=image_embeds.float().to(self.Qformer.device),
                encoder_attention_mask=image_atts.to(self.Qformer.device),
                return_dict=True,
            )
        
        # if self.checknan(query_output):
        #     print("query_output is nan")
        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:]).to(image.device)
        # global pooling
        # inputs_llm = inputs_llm.mean(1).unsqueeze(1)
        # if self.checknan(inputs_llm):
        #     print("input_llm is nan")
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            samples['text_input'],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)
        # if self.checknan(text_input_tokens):
        #     print("text input tokens is nan")
        
        self.llm_tokenizer.truncation_side = 'right'
        text_output_tokens = self.llm_tokenizer(
            [t + self.llm_tokenizer.eos_token for t in samples['text_output']],
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_output_txt_len,
        ).to(image.device)
        
        # if self.checknan(text_output_tokens):
        #     print("text_output_tokens is nan")
        llm_tokens, input_part_targets_len = self.concat_text_input_output(
            text_input_tokens.input_ids,
            text_input_tokens.attention_mask,
            text_output_tokens.input_ids,
            text_output_tokens.attention_mask,
        )
        # if self.checknan(llm_tokens["input_ids"]):
        #     print("llm_tokens is nan")
        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.llm_tokenizer.pad_token_id, -100
        )
        # if self.checknan(targets):
        #     print("targets is nan")
        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        # do not apply loss to the prefix prompts
        # prefix_targets = prefix_tokens["input_ids"].to(image.device).fill_(-100)
        # print("prefix_targets ", prefix_targets)
        targets = torch.cat([empty_targets, targets], dim=1)
        
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens['input_ids']).to(image.device) * self.llm_model.biogpt.embed_scale
        # prefix_embeds = self.llm_model.get_input_embeddings()(prefix_tokens["input_ids"]).to(image.device)
        # if self.checknan(inputs_embeds):
        #     print("before inputs_embeds is nan")
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        # if self.checknan(inputs_embeds):
        #     print("inputs_embeds is nan")
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)
        # attention_mask = llm_tokens["attention_mask"]

        # with self.maybe_autocast():
        #     outputs = self.llm_model(
        #         inputs_embeds=inputs_embeds,
        #         attention_mask=attention_mask,
        #         return_dict=True,
        #         labels=targets,
        #     )
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
        # prefix_prompt = "Chest X-ray feature:"

        if "text_input" in samples.keys():
            prompt = samples["text_input"]
        else:
            prompt = self.prompt

        image = samples["image"]

        bs = image.size(0)
        # prefix_prompt = [prefix_prompt] * bs
        # print(f"batch_size = {bs}")
        if isinstance(prompt, str):
            prompt = [prompt] * bs
        else:
            assert len(prompt) == bs, "The number of prompts must be equal to the batch size."

        # For TextCaps
        if "ocr_tokens" in samples.keys() and "{}" in prompt[0]:
            prompt = [p.format(', '.join(samples['ocr_tokens'][i][:30])) for i, p in enumerate(prompt)]

        query_tokens = self.query_tokens.expand(bs, -1, -1)
        if self.qformer_text_input:
            # remove ocr tokens in q_former (for eval textvqa)
            # qformer_prompt = prompt
            # qformer_prompt = ['Question: ' + qp.split(' Question: ')[1] for qp in qformer_prompt]

            text_Qformer = self.tokenizer(
                prompt,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

        # For video data
        if image.dim() == 5:
            inputs_llm, atts_llm = [], []
            for j in range(image.size(2)):
                this_frame = image[:,:,j,:,:]
                with self.maybe_autocast():
                    frame_embeds = self.ln_vision(self.visual_encoder(this_frame))
                frame_atts = torch.ones(frame_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.qformer_text_input:
                    frame_query_output = self.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                else:
                    frame_query_output = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=frame_embeds,
                        encoder_attention_mask=frame_atts,
                        return_dict=True,
                    )
                frame_inputs_llm = self.llm_proj(frame_query_output.last_hidden_state[:,:query_tokens.size(1),:])
                frame_atts_llm = torch.ones(frame_inputs_llm.size()[:-1], dtype=torch.long).to(image.device)
                inputs_llm.append(frame_inputs_llm)
                atts_llm.append(frame_atts_llm)
            inputs_llm = torch.cat(inputs_llm, dim=1)
            atts_llm = torch.cat(atts_llm, dim=1)
        else:
            # with self.maybe_autocast():
                # image_embeds = self.ln_vision(self.visual_encoder(image))
            image_embeds = self.vision_encode(image)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

            if self.qformer_text_input:
                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            else:
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

            inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        llm_tokens = self.llm_tokenizer(
            prompt,
            padding="longest",
            return_tensors="pt"
        ).to(image.device)

        # prefix_llm_tokens = self.llm_tokenizer(
        #     prefix_prompt,
        #     padding="longest",
        #     return_tensors="pt"
        # ).to(image.device)
        # print(prefix_llm_tokens.input_ids.size())

        # with self.maybe_autocast():
        inputs_embeds = self.llm_model.get_input_embeddings()(llm_tokens.input_ids) * self.llm_model.biogpt.embed_scale
        # prefix_embeds = self.llm_model.get_input_embeddings()(prefix_llm_tokens.input_ids)
        # print(prefix_embeds.size())
        # print(inputs_llm.size())
        # print(inputs_embeds.size())
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

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
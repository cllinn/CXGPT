import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 
import json 
import pandas as pd 
from transformers import AutoFeatureExtractor, AutoImageProcessor
from PIL import Image 
import numpy as np 
import random 
from tqdm import tqdm 
import numpy as np 
import re


PROMPT_TEMPLATE = {
    "i": [
        "corresponding radiology reports is",
        "based on the chest x-ray the impression is"
    ],
    "it": [
        "radiology reports is {findings}, the corrseponding diagnostic opinions is",
        "the radiology reports {findings}, and the impression is",
        "Findings:{findings}, Impression:"
        ]
}

class OPENI(Dataset):
    def __init__(self):
        csv_path = "../dataset/processed_data/new_data.csv"
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df.index)
        self.image_processor = AutoFeatureExtractor.from_pretrained("/root/autodl-tmp/checkpoint/cvt-21")
    
    def __getitem__(self, index):
        if index < self.len:
            findings = self.df.loc[index % self.len]["FINDING"]
            prompts = PROMPT_TEMPLATE["i"][0]
            image_path = self.df.loc[index % self.len]["image_path_0"]
            image = Image.open(image_path)
            image = self.image_processor(image, return_tensors="pt")
            image = image["pixel_values"].squeeze()
            sample = {
                "image": image,
                "text_input": prompts,
                "text_output": findings
            }
        elif index < 2 * self.len:
            findings = self.df.loc[index % self.len]["FINDING"]
            impression = self.df.loc[index % self.len]["IMPRESSION"]
            p = random.random()
            if p >= 0.5:
                prompts = PROMPT_TEMPLATE["it"][0]
            else:
                prompts = PROMPT_TEMPLATE["it"][1]
            image_path = self.df.loc[index % self.len]["image_path_0"]
            image = Image.open(image_path)
            image = self.image_processor(image, return_tensors="pt")
            image = image["pixel_values"].squeeze()
            sample = {
                "image": image,
                "text_input": prompts.format(findings=findings),
                "text_output": impression
            }
        else:
            findings = self.df.loc[index % self.len]["FINDING"]
            impression = self.df.loc[index % self.len]["IMPRESSION"]
            prompts = PROMPT_TEMPLATE["it"][2]
            image_path = self.df.loc[index % self.len]["image_path_0"]
            image = Image.open(image_path)
            image = self.image_processor(image, return_tensors="pt")
            image = image["pixel_values"].squeeze()
            sample = {
                "image": image,
                "text_input": prompts.format(findings=findings),
                "text_output": impression
            }
        return sample 
    
    def __len__(self):
        return 3 * self.len 
    
class OPENI_i2t(Dataset):
    def __init__(self):
        csv_path = "../dataset/processed_data/new_data.csv"
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df.index)
        self.image_processor = AutoFeatureExtractor.from_pretrained("/root/autodl-tmp/checkpoint/cvt-21")
    
    def __getitem__(self, index):
        findings = self.df.loc[index % self.len]["FINDING"]
        prompts = PROMPT_TEMPLATE["i"][0]
        image_path = self.df.loc[index % self.len]["image_path_0"]
        image = Image.open(image_path)
        image = self.image_processor(image, return_tensors="pt")
        image = image["pixel_values"].squeeze()
        sample = {
            "image": image,
            "text_input": prompts,
            "text_output": findings
        }
        return sample 
    
    def __len__(self):
        return self.len

class OPENI_image2impression(Dataset):
    def __init__(self):
        csv_path = "../dataset/processed_data/new_data.csv"
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df.index)
        self.image_processor = AutoFeatureExtractor.from_pretrained("/root/autodl-tmp/checkpoint/cvt-21")
    
    def __getitem__(self, index):
        impression = self.df.loc[index % self.len]["IMPRESSION"]
        impression = re.sub("[.,]*", "", impression)
        prompts = PROMPT_TEMPLATE["i"][1]
        image_path = self.df.loc[index]["image_path_0"]
        image = Image.open(image_path)
        image = self.image_processor(image, return_tensors="pt")
        image = image["pixel_values"].squeeze()
        sample = {
            "image": image,
            "text_input": prompts,
            "text_output": impression
        }
        return sample 
    
    def __len__(self):
        return self.len

class OPENI_if2i(Dataset):
    def __init__(self):
        csv_path = "../dataset/processed_data/new_data.csv"
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df.index)
        self.image_processor = AutoFeatureExtractor.from_pretrained("/root/autodl-tmp/checkpoint/cvt-21")
    
    def __getitem__(self, index):
        impression = self.df.loc[index % self.len]["IMPRESSION"]
        findings = self.df.loc[index % self.len]["FINDING"]
        impression = re.sub("[.,]*", "", impression)
        prompts = PROMPT_TEMPLATE["it"][2]
        image_path = self.df.loc[index]["image_path_0"]
        image = Image.open(image_path)
        image = self.image_processor(image, return_tensors="pt")
        image = image["pixel_values"].squeeze()
        sample = {
            "image": image,
            "text_input": prompts.format(findings=findings),
            "text_output": impression
        }
        return sample 
    
    def __len__(self):
        return self.len
    

class OPENI_image2findings(Dataset):
    def __init__(self):
        csv_path = "../dataset/processed_data/new_data.csv"
        self.df = pd.read_csv(csv_path)
        self.image_processor = AutoImageProcessor.from_pretrained("/root/autodl-tmp/checkpoint/resnet50")
        self.len = len(self.df.index)
    
    def __getitem__(self, index):
        findings = self.df.loc[index % self.len]["FINDING"]
        image_path = self.df.loc[index % self.len]["image_path_0"]
        image = Image.open(image_path)
        image = self.image_processor(image, return_tensors="pt")
        image = image["pixel_values"].squeeze()
        sample = {
            "image": image,
            "text_output": findings
        }
        return sample 
    
    def __len__(self):
        return self.len
    
class OPENI_it2t(Dataset):
    def __init__(self):
        csv_path = "../dataset/processed_data/new_data.csv"
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df.index)
        self.image_processor = AutoFeatureExtractor.from_pretrained("/root/autodl-tmp/checkpoint/cvt-21")
    
    def __getitem__(self, index):
        findings = self.df.loc[index % self.len]["FINDING"]
        impression = self.df.loc[index % self.len]["IMPRESSION"]
        prompts = PROMPT_TEMPLATE["it"][2]
        image_path = self.df.loc[index % self.len]["image_path_0"]
        image = Image.open(image_path)
        image = self.image_processor(image, return_tensors="pt")
        image = image["pixel_values"].squeeze()
        sample = {
            "image": image,
            "text_input": prompts.format(findings=findings),
            "text_output": impression
        }
        return sample 
    
    def __len__(self):
        return self.len 
    
class OPENI_i_im2t(Dataset):
    def __init__(self):
        csv_path = "../dataset/processed_data/new_data.csv"
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df.index)
        self.image_processor = AutoFeatureExtractor.from_pretrained("/root/autodl-tmp/checkpoint/cvt-21")
    
    def __getitem__(self, index):
        findings = self.df.loc[index % self.len]["FINDING"]
        impression = self.df.loc[index % self.len]["IMPRESSION"]
        prompts = PROMPT_TEMPLATE["it"][2]
        image_path = self.df.loc[index % self.len]["image_path_0"]
        image = Image.open(image_path)
        image = self.image_processor(image, return_tensors="pt")
        image = image["pixel_values"].squeeze()
        sample = {
            "image": image,
            "text_input": impression
        }
        return sample 
    
    def __len__(self):
        return self.len 
    
    
class OPENI_t2t(Dataset):
    def __init__(self):
        csv_path = "../dataset/processed_data/new_data.csv"
        self.df = pd.read_csv(csv_path)
        self.len = len(self.df.index)
        # self.image_processor = AutoFeatureExtractor.from_pretrained("/root/autodl-tmp/checkpoint/cvt-21")
    
    def __getitem__(self, index):
        findings = self.df.loc[index % self.len]["FINDING"]
        impression = self.df.loc[index % self.len]["IMPRESSION"]
        # p = random.random()
        # if p >= 0.5:
        #     prompts = PROMPT_TEMPLATE["it"][0]
        # else:
        #     prompts = PROMPT_TEMPLATE["it"][1]
        prompts = PROMPT_TEMPLATE["it"][2]
        sample = {
            "text_input": prompts.format(findings=findings),
            "text_output": impression
        }
        return sample 
    
    def __len__(self):
        return self.len 

if __name__ == "__main__":
    # csv_path = "../dataset/processed_data/all_data.csv"
    # df = pd.read_csv(csv_path)
    # for i, j in enumerate(df.image_path_0):
    #     df.loc[i, "image_path_0"] = j.replace("/data/user/MYF/IU X-RAY", "/root/mulcon/dataset")
    
    # df.to_csv("../dataset/processed_data/new_data.csv")
    data = OPENI_image2impression()
    # # torch.save(data, "OPENI.pt")
    # # data = torch.load("OPENI.pt")
    l = len(data)
    tr = int(0.8 * l)
    te = l - tr
    train, test = torch.utils.data.random_split(data, [tr, te])
    torch.save(train, "OPENI_if2i_train.pt")
    torch.save(test, "OPENI_if2i_test.pt")
    
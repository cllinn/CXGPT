import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader 
import json 
from transformers import AutoFeatureExtractor, AutoImageProcessor
from PIL import Image 
# from torchvision import utils 
import numpy as np 
from tqdm import tqdm 
import random 

PROMPT_TEMPLATE = {
    "i": ["Generate corresponding radiology reports based on previous chest X-ray image."],
    "it": [
        "Generate diagnostic opinions based on chest X-ray image and radiology reports {findings}",
        "Generate impression based on chest X-ray image and the radiology reports {findings}.",
        "Findings:{findings},Impression:"
        ]
}

class LMDataset(Dataset):
    def __init__(self):
        super().__init__()
        json_path = "../dataset/MIMIC-CXR/data_processed/new_data.json"
        with open(json_path, "r") as f:
            fd = json.load(f)
        self.ls = list(fd)
        self.fd = fd 
        self.image_processor = AutoFeatureExtractor.from_pretrained("../checkpoint/cvt-21")
        self.len = len(self.ls)

    def __getitem__(self, index):
        if index < self.len:
            out_k = self.ls[index % self.len]  
            in_d = self.fd[out_k]
            findings = in_d["FINDINGS"]
            # impr = in_d["IMPRESSION"]
            prompts = PROMPT_TEMPLATE["i"][0]
            image_path = in_d.get("IMAGE")
            image = Image.open(image_path)
            image = self.image_processor(image, return_tensors="pt")
            image = image["pixel_values"].squeeze()
            sample = {
                "image": image,
                "text_input": prompts,
                "text_output": findings
            }
        elif index < 2 * self.len:
            out_k = self.ls[index % self.len]
            in_d = self.fd[out_k]
            findings = in_d["FINDINGS"]
            impression = in_d["IMPRESSION"]
            p = random.random()
            if p >= 0.5:
                prompts = PROMPT_TEMPLATE["it"][0]
            else:
                prompts = PROMPT_TEMPLATE["it"][1]
            image_path = in_d.get("IMAGE")
            image = Image.open(image_path)
            image = self.image_processor(image, return_tensors="pt")
            image = image["pixel_values"].squeeze()
            sample = {
                "image": image,
                "text_input": prompts.format(findings=findings),
                "text_output": impression
            }
        else:
            out_k = self.ls[index % self.len]
            in_d = self.fd[out_k]
            findings = in_d["FINDINGS"]
            impression = in_d["IMPRESSION"]
            prompts = PROMPT_TEMPLATE["it"][2]
            image_path = in_d.get("IMAGE")
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

if __name__ == "__main__":
    data = LMDataset()
    torch.save(data, "LMDataset.pt")
    # data = MIMIC_CXR()
    # torch.save(data, "MIMIC_CXR.pt")
    # json_path = "../dataset/MIMIC-CXR/data_processed/new_data.json"
    # with open(json_path, "r") as f:
    #     fd = json.load(f)
    # ls = list(fd)
    # print(fd[ls[0]].keys())
    # image_processor = AutoFeatureExtractor.from_pretrained("../checkpoint/cvt-21")
    # index = 0
    # out_k = ls[index]
    # in_d = fd[out_k]
    # image_path = in_d["IMAGE"]
    # print(image_path)
    # image = Image.open(image_path)
    # print(image)
    # image = image_processor(image, return_tensors="pt")
    # image = image["pixel_values"].clone().detach().cpu()
    # r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
    # print((r-g).sum(), (r-g).sum(), (g-b).sum())
    # utils.save_image(image, "image2.png")
    # data = MIMIC_CXR()
    # R_channels = 0.0
    # G_channels = 0.0
    # B_channels = 0.0
    # loader = DataLoader(data, batch_size=32)
    # num = 0
    # for i in tqdm(loader):
    #     image = i["image"]
    #     R_image = image[:, 0, :, :]
    #     G_image = image[:, 1, :, :]
    #     B_image = image[:, 2, :, :]
    #     R_channels += R_image.sum()
    #     G_channels += G_image.sum()
    #     B_channels += B_image.sum()
    #     num += image.size(0) * image.size(2) * image.size(3)
    # R_mean = R_channels / num
    # G_mean = G_channels / num
    # B_mean = B_channels / num
    # R_channels = 0.0
    # G_channels = 0.0
    # B_channels = 0.0
    # for i in loader:
    #     image = i["image"]
    #     R_image = image[:, 0, :, :]
    #     G_image = image[:, 1, :, :]
    #     B_image = image[:, 2, :, :]
    #     R_channels += ((R_image - R_mean) ** 2).sum()
    #     G_channels += ((G_image - G_mean) ** 2).sum()
    #     B_channels += ((B_image - B_mean) ** 2).sum()
    # R_var, G_var, B_var = np.sqrt(R_channels / num), np.sqrt(G_channels / num), np.sqrt(B_channels / num)
    # print(R_mean, G_mean, B_mean, R_var, G_var, B_var)
    # print(num, len(data))


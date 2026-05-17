import torch
from model.clip import CLIP
from model.model_configs import clip_config_set
import torchinfo

if __name__ == "__main__":
    # model = CLIP(clip_cfg=clip_config_set.clip_150M_normal).to(device="cuda")
    # print((torchinfo.summary(model, verbose=0).total_params - 65706496)/10**9, "B normal")

    # model = CLIP(clip_cfg=clip_config_set.clip_1_5B_moe).to(device="cuda")
    # print((torchinfo.summary(model, verbose=0).total_params - 65706496)/10**9, "B moe")

    model = CLIP(clip_cfg=clip_config_set.clip_1_5B_engram).to(device="cuda")
    print((torchinfo.summary(model, verbose=0).total_params - 65706496)/10**9, "B engram")

import torch
from model.clip import CLIP
from model.model_configs import clip_config_set
import torchinfo

if __name__ == "__main__":
    # model = CLIP(clip_cfg=clip_config_set.clip_150M_normal).to(device="cuda")
    # print((torchinfo.summary(model, verbose=0).total_params - 65706496)/10**9, "B normal")

    # model = CLIP(clip_cfg=clip_config_set.clip_1_5B_moe).to(device="cuda")
    # print((torchinfo.summary(model, verbose=0).total_params - 65706496)/10**9, "B moe")

    img = torch.randn((1, 3, 224, 224), dtype=torch.float16)
    text = torch.randint(0, 100, (1, 77))

    model = CLIP(clip_cfg=clip_config_set.clip_1_5B_engram).to(device="cpu", dtype=torch.float16)

    print(model(img, text))
    print((torchinfo.summary(model, verbose=1).total_params - 65706496)/10**9, "B engram")

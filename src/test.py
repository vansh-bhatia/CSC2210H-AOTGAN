import importlib
import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor,Resize
from utils.option import args
from tqdm import tqdm
import time


def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)


def main_worker(args, use_gpu=True):
    device = torch.device("cuda") if use_gpu else torch.device("cpu")

    # Model and version
    net = importlib.import_module("model." + args.model)
    model = net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load(args.pre_train, map_location="cuda"))
    model.eval()

    # prepare dataset
    image_paths = []
    for ext in [".jpg", ".png"]:
        image_paths.extend(glob(os.path.join(args.dir_image, args.data_test, "*" + ext)))
    image_paths.sort()
    mask_paths = sorted(glob(os.path.join(args.dir_mask, "*.png")))
    os.makedirs(args.outputs, exist_ok=True)

    total_time = 0
    # iteration through datasets
    for ipath, mpath in tqdm(zip(image_paths, mask_paths)):
        temp = time.time()
        image = ToTensor()(Image.open(ipath).resize((args.image_size,args.image_size)).convert("RGB"))
        image = (image * 2.0 - 1.0).unsqueeze(0)
        mask = ToTensor()(Image.open(mpath).resize((args.image_size,args.image_size)).convert("L"))
        mask = mask.unsqueeze(0)
        # image = Resize(args.image_size, interpolation=transforms.InterpolationMode.NEAREST)(image)
        image, mask = image.cuda(), mask.cuda()
        image_masked = image * (1 - mask.float()) + mask

        with torch.no_grad():
            pred_img = model(image_masked, mask)

        comp_imgs = (1 - mask) * image + mask * pred_img
        total_time += time.time() - temp
        image_name = os.path.basename(ipath).split(".")[0]
        postprocess(image_masked[0]).save(os.path.join(args.outputs, f"{image_name}_masked.png"))
        postprocess(pred_img[0]).save(os.path.join(args.outputs, f"{image_name}_pred.png"))
        postprocess(comp_imgs[0]).save(os.path.join(args.outputs, f"{image_name}_comp.png"))
        # print(f"saving to {os.path.join(args.outputs, image_name)}")\
    average_time = total_time/len(image_paths)
    print(f"Average time for each inference: {average_time:.5f}s" )


if __name__ == "__main__":
    main_worker(args)

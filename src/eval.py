import argparse
from glob import glob
from multiprocessing import Pool

import numpy as np
from metric import metric as module_metric
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Image Inpainting")
parser.add_argument("--real_dir", default="data/aotgan/images/test", type=str)
parser.add_argument("--fake_dir", default="data/outputs", type=str)
parser.add_argument("--metric", type=str, nargs="+", default="mae psnr ssim")
parser.add_argument("--image_size", type=int, default=128, help="image size used during training")
args = parser.parse_args()


# def read_img(name_pair):
#     rname, fname = name_pair
#     rimg = Image.open(rname).resize((args.image_size,args.image_size))
#     fimg = Image.open(fname)
#     return np.array(rimg), np.array(fimg)

def read_img(name):
    rname = name
    fname = args.fake_dir + "/" + name.split("/")[-1][:-4] + "_comp.png"
    rimg = Image.open(rname).resize((args.image_size,args.image_size))
    fimg = Image.open(fname)
    return np.array(rimg), np.array(fimg)


def main(num_worker=8):
    real_names = sorted(glob(f"{args.real_dir}/*.jpg"))
    # fake_names = sorted(glob(f"{args.fake_dir}/*comp.png"))
    # print(f"real images: {len(real_names)}, fake images: {len(fake_names)}")
    # for i in range(len(real_names)):
    #     rname_name = real_names[i].split("/")[-1][:-4]
    #     fname_name = fake_names[i].split("/")[-1][:-9]
    #     if rname_name != fname_name:
    #         print(rname_name,fname_name)
    # exit()
    real_images = []
    fake_images = []
    pool = Pool(num_worker)
    # for rimg, fimg in tqdm(
    #     pool.imap_unordered(read_img, zip(real_names, fake_names)), total=len(real_names), desc="loading images"
    # ):
    for rimg, fimg in tqdm(
        pool.imap_unordered(read_img, real_names), total=len(real_names), desc="loading images"
    ):
        if len(rimg.shape) > 2:
            real_images.append(rimg)
            fake_images.append(fimg)

    # metrics prepare for image assesments
    metrics = {met: getattr(module_metric, met) for met in args.metric.split()}
    evaluation_scores = {key: 0 for key, val in metrics.items()}
    for key, val in metrics.items():
        evaluation_scores[key] = val(real_images, fake_images, num_worker=num_worker)
    print(" ".join(["{}: {:6f},".format(key, val) for key, val in evaluation_scores.items()]))


if __name__ == "__main__":
    main()

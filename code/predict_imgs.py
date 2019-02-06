import argparse
from pathlib import Path
from PIL import Image
import torch
import torchvision
from skimage.morphology import erosion
import matplotlib.pyplot as plt
import time

from utils import init_weights
from dataloader import pad_pair_256, normalize
from model import SegRoot

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image", default="test.jpg", type=str, help="filename of one test image"
)
parser.add_argument(
    "--thres", default=0.9, type=float, help="threshold of the final binarization"
)
parser.add_argument(
    "--all", action="store_true", help="make prediction on all images in the folder"
)
parser.add_argument(
    "--data_dir",
    default="../data/prediction",
    type=Path,
    help="define the data directory",
)
parser.add_argument(
    "--weights",
    default="../weights/best_segnet-(8,5)-0.6441.pt",
    type=Path,
    help="path of pretrained weights",
)
parser.add_argument("--width", default=8, type=int, help="width of SegRoot")
parser.add_argument("--depth", default=5, type=int, help="depth of SegRoot")


def pad_256(img_path):
    image = Image.open(img_path)
    W, H = image.size
    img, _ = pad_pair_256(image, image)
    NW, NH = img.size
    img = torchvision.transforms.ToTensor()(img)
    img = normalize(img)
    return img, (H, W, NH, NW)


def predict(model, test_img, device):
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    # test_img.shape = (3, 2304, 2560)
    test_img = test_img.unsqueeze(0)
    output = model(test_img)
    # output.shape = (1, 1, 2304, 2560)
    output = torch.squeeze(output)
    torch.cuda.empty_cache()
    return output


def predict_gen(model, img_path, thres, device, info):
    img, dims = pad_256(img_path)
    H, W, NH, NW = dims
    img = img.to(device)
    prediction = predict(model, img, device)
    prediction[prediction >= thres] = 1.0
    prediction[prediction < thres] = 0.0
    if device.type == "cpu":
        prediction = prediction.detach().numpy()
    else:
        prediction = prediction.cpu().detach().numpy()
    prediction = erosion(prediction)
    # reverse padding
    prediction = prediction[
        (NH - H) // 2 : (NH - H) // 2 + H, (NW - W) // 2 : (NW - W) // 2 + W
    ]
    save_path = img_path.parent / (
        img_path.parts[-1].split(".jpg")[0] + "-pre-mask-segnet-({},5).jpg".format(info)
    )
    plt.imsave(save_path.as_posix(), prediction, cmap="gray")
    print("{} generated!".format(save_path.parts[-1]))


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define model
    print("using segnet, width : {}, depth : {}".format(args.width, args.depth))
    model = SegRoot(args.width, args.depth).to(device)
    weights_path = args.weights

    if device.type == "cpu":
        print("load weights to cpu")
        print(weights_path.as_posix())
        model.load_state_dict(torch.load(weights_path.as_posix(), map_location="cpu"))
    else:
        print("load weights to gpu")
        print(weights_path.as_posix())
        model.load_state_dict(torch.load(weights_path.as_posix()))

    # define the prediction's saving directory
    pre_dir = Path("../data/prediction")
    pre_dir.mkdir(parents=True, exist_ok=True)
    if not args.all:
        # load and pad image
        img_path = pre_dir / args.image
        start_time = time.time()
        predict_gen(model, img_path, args.thres, device, 8)
        end_time = time.time()
        print("{:.4f}s for one image".format(end_time - start_time))
    else:
        img_paths = args.data_dir.glob("*.jpg")
        for img_path in img_paths:
            start_time = time.time()
            predict_gen(model, img_path, args.thres, device, 8)
            end_time = time.time()
            print("{:.4f}s for one image".format(end_time - start_time))


import argparse
import glob

import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.inception import InceptionScore


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inception = InceptionScore()

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i","--image_path"
                        ,default='../sample_output/',
                        type=str)
    
    args = parser.parse_args()
    img_paths = args.image_path

    img_paths = glob.glob(img_paths+"*")
    for img_path in img_paths:
        img = Image.open(img_path)

        preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        ])

        img_tensor = preprocess(img).unsqueeze(0)

        inception.update(img_tensor.to(torch.uint8))

    mean_score, std_score = inception.compute()

    print("Inception Score:", mean_score, "+/-", std_score)
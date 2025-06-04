# image blur tool placeholder
from PIL import Image, ImageFilter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

img = Image.open(args.input)
blurred = img.filter(ImageFilter.GaussianBlur(radius=2))
blurred.save(args.output)

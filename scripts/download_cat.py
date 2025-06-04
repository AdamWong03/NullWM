from PIL import Image
import requests
from io import BytesIO

url = "https://images.unsplash.com/photo-1518791841217-8f162f1e1131"
img = Image.open(BytesIO(requests.get(url).content)).convert("RGB")
img.save("C:\\Users\\AdamWong\\Desktop\\NullWM\\data\\cat.jpg")

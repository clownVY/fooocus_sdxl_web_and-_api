import base64
import io

import requests
from PIL import Image

url = "http://172.17.240.85:7864/v1/generate_image/"

result = requests.post(url, json={"prompt": "park, flower, tree, winter", "image_number": 1},
                       headers={"Content-Type": "application/json"})

print(result.status_code)
for i, img_string in enumerate(result.json()):
    # 将Base64编码的图片字符串解码为图片
    img_data = base64.b64decode(img_string)
    img = Image.open(io.BytesIO(img_data))
    img.save(f'output_{i}.png')

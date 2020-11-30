import os
import os.path
from PIL import Image


for address, dirs, files in os.walk("shawarma"):

    for file in files:
        file_lower = file.lower()
        if ".png" in file_lower:
            image_format = "PNG"
        elif ".jpg" in file_lower or ".jpeg" in file_lower:
            image_format = "JPEG"
        else:
            continue

        file_path = os.path.join(address, file)

        print(f"Prepare: {file_path}")
        image = Image.open(file_path)
        aspect_ratio = image.size[0] / image.size[1]
        if image.size[0] > image.size[1]:
            if image.size[1] <= 1024:
                continue

            image = image.resize((int(aspect_ratio * 1024), 1024))
        else:
            if image.size[0] <= 1024:
                continue

            image = image.resize((1024, int(1024 / aspect_ratio)))
        print("resize: ", image.size)
        image.save(file_path, image_format)

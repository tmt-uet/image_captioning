import requests
from datetime import datetime
import shutil

# r = requests.get(settings.STATICMAP_URL.format(**data), stream=True)
# if r.status_code == 200:
#     with open(path, 'wb') as f:
#         r.raw.decode_content = True
#         shutil.copyfileobj(r.raw, f)


# def is_url_image(image_url):

#     image_formats = ("image/png", "image/jpeg", "image/jpg")
#     r = requests.head(image_url)
#     if r.headers["content-type"] in image_formats:
#         return True
#     return False


# created1 = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
# url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSx3GbsqG2u5Z53ZF5G-RS8etYDnwYy648xrg&usqp=CAU'
# result = is_url_image(url)
# print(result)

# response = requests.get(url, stream=True)
# with open('storage/{}.jpg'.format(created1), 'wb') as out_file:
#     shutil.copyfileobj(response.raw, out_file)
# del response

url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSx3GbsqG2u5Z53ZF5G-RS8etYDnwYy648xrg&usqp=CAU'

VALID_IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
]


def valid_url_extension(url, extension_list=VALID_IMAGE_EXTENSIONS):
    # http://stackoverflow.com/a/10543969/396300
    return any([url.endswith(e) for e in extension_list])


result = valid_url_extension(url)
print(result)

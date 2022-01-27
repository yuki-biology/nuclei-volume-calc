import numpy as np
import os
import cv2


# 空の画像
def empty_image(shape, dtype=np.uint8):
    return np.zeros(shape, dtype=dtype)


# 画像の保存
def save_image(filepath, image):
    dirname = os.path.dirname(filepath)
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    cv2.imwrite(filepath, cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))

#連続画像をスタック
def z_stack_image(images, dtype=np.uint16):
    if len(images) == 0:
        return None
    
    result = empty_image(images[0].shape, dtype=dtype)
    for img in images:
        result = cv2.add(result, img.astype(dtype))
    return result

#画像をX軸/Y軸で分割
def separate_image(img, split_x=1):
    h, w = img.shape[:2]
    x0 = int(h/split_x)

    # 分割した画像を内包表記でリスト化
    c = [img[x0*x:x0*(x+1), 0:h] for x in range(split_x)]
    return c

#画像をグレースケールにする
def gray_scale(image, mode = cv2.COLOR_BGR2GRAY):
    if len(image.shape) == 2:
        return image
    else:
        return cv2.cvtColor(image, mode)

# グレースケール画像を色付け
colors = {
    "Red": [255,0,0],
    "Green": [0,255,0],
    "Blue": [0,0,255],
    "Magenta": [255,0,255],
    "Pan": [255,255,255],
}

def set_image_color(_image, rgb):
    if type(rgb) == "string":
        rgb = colors[rgb]
    if len(_image.shape) == 3:
        _image = gray_scale(_image)

    _image = cv2.cvtColor(_image, cv2.COLOR_GRAY2BGR)
    image = _image.copy()

    # 16ビット数なのでちょっと足してみる
    #color_mult = (2**16 - 1) / (2**12 - 1)
    color_mult = 1
    
    r = np.array(image, dtype=np.float64)
    r[:, :, (0,1)] = 0

    g = np.array(image, dtype=np.float64)
    g[:, :, (0,2)] = 0
    
    b = np.array(image, dtype=np.float64)
    b[:, :, (1,2)] = 0
    
    image = b*rgb[0]/255 + g*rgb[1]/255 + r*rgb[2]/255    
    return np.array(image, dtype=np.uint8)

def normalize_image(image, max_value=255):
    return normalize_images([image], max_value)[0]

def normalize_images(images, max_value=255):
    return np.array(
        [np.array(image, dtype=np.uint8) * (max_value / np.max(images)) for image in images], 
    dtype=np.uint8)
## 画像関連 (OpenCV)
import numpy as np
import cv2
import tempfile
import os
from ..utils.b64 import *

# ビット数を変換
def convert_image_bit_number(img, bit_from=16, bit_to=8):
    img = img.astype(np.float64)
    img *= (2**bit_to-1)/(2**bit_from-1)
    
    if bit_to > 8:
        return img.astype(np.uint16)
    else:
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    
def binary_to_cv2(s, bit_from=12, bit_to=8):
    img = None
    # Tempfileを作成して即読み込む
    fp = tempfile.NamedTemporaryFile(dir='./', delete=False)
    fp.write(s)
    fp.close()
    img = cv2.imread(fp.name, -1)
    os.remove(fp.name)

    #ビット数を変換
    return convert_image_bit_number(img, bit_from, bit_to)

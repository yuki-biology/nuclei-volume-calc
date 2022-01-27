import cc3d
import numpy as np
from matplotlib import pyplot as plt

from module.image_utils.image_utils import gray_scale, set_image_color
from module.stats.plot import show_image

from .stats import *

def cc3d_detect_regions(images, connectivity = 6):
    print("analyzing areas")
    if type(images) == list:
        images = np.array(images)

    labels_out = cc3d.connected_components(images,connectivity=connectivity)
    
    #領域数を出力
    numbers = np.max(labels_out) - 1
    print(numbers, "regions found")
    
    # 空洞体積の計算
    if numbers:
        areas = []
        for num in range(np.max(labels_out)):
            print(f"\rlabel_{num}", end="")
            
            count = np.count_nonzero(labels_out == num)
            areas.append(int(count))
    
        print("--done")
        result =  {
            "Labels": labels_out,
            "Volumes": areas,
            "Values": [{
                "Id": i, "Volume_px3": int(area)
            } for i, area in enumerate(areas)]
        }
    #一個もない場合、空の配列を返す
    else:
        result =  {
            "Labels": [],
            "Volumes": [],
            "Values": []
        }
    
    return result

def extract_label_3D(Labels, index):
    result = []
    for i, l in enumerate(Labels):
        img =np.where(l == index, 255, 0)
        result.append(img)
    return result

# CC3Dの画像スタックを色付け (WIP)
import random
import cv2
def set_random_label_color(labels_out):
    ## それぞれのラベルに対応したRGBカラーをランダムに選択
    param_list = np.linspace(0,np.max(labels_out),np.max(labels_out)+1,dtype=np.uint16)
    color_list = {}
    for i in param_list:
        if(i != 0):
            color_list[str(i)] = [random.randint(0,255),
                                    random.randint(0,255),
                                    random.randint(0,255)]
        else:
            color_list[str(i)] = [0,0,0]

    ## それぞれの空洞領域を色付け
    void_colored = []
    for img in labels_out:
        h,w = img.shape
        img_flat = img.flatten()
        img_colored = np.zeros((img_flat.shape[0],3),dtype=np.uint8)
        non_zero_idx = np.where(img_flat != 0)
        for n in non_zero_idx[0]:
            img_colored[n] = color_list[str(img_flat[n])]
        void_colored.append(img_colored)
    void_colored = np.array(void_colored)
    void_colored = void_colored.reshape(void_colored.shape[0],h,w,3)
    return void_colored
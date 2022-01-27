import cv2
import numpy as np
import scipy.ndimage as ndimage
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from .image_mask import *


def detect_contours(th_fill):
    contours, hierarchy = cv2.findContours(th_fill.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def set_contours(img_raw, th_fill, color=(255,255,255), thickness=1, index=-1):
    h, w = th_fill.shape[:2]
    
    if img_raw is not None:
        h_raw, w_raw = img_raw.shape[:2]
        img_result = cv2.resize(img_raw, (w, h))
    else:
        h_raw, w_raw = h, w
        img_result = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    
    #境界検出と描画
    contours = detect_contours(th_fill)
    img_result = cv2.drawContours(img_result, contours, -1, color=color, thickness=thickness)
    img_result = cv2.resize(img_result, (w_raw, h_raw))

    return {
        "image": img_result,
        "contours": contours,
    }


def analyze_contour(contour):
    #面積(px*px)
    area = cv2.contourArea(contour)

    #円形度
    arc = cv2.arcLength(contour, True) or 1

    circularity = 4 * np.pi * area / (arc * arc)

    #等価直径(px)
    eq_diameter = np.sqrt(4*area/np.pi)
    
    expected_volume = 3 / 4 * np.pi * (eq_diameter**3)
    
    return {
        "area": area,
        "circularity": circularity,
        "eq_diameter": eq_diameter,
        "expected_volume": expected_volume
    }


class Image_Particles(object):
    def __init__(self, img, mag=3, mode="RAW", block_size=21):
        self.images = {}
        self.mag=mag
        self.mode = mode

        #近傍サイズの設定
        block_size = int(block_size)
        if block_size % 2 == 0:
            block_size += 1
        self.block_size = block_size

        if self.mode == "RAW":
            self.images = {
                "raw" : img,
                "gray": img
            }
            
            # カラー画像ならグレーに直す
            if (len(img.shape)==3):
                self.images["gray"] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            self.height, self.width = img.shape[:2]
        
        else:
            # 想定: mode="TRESH_FILL"
            self.images = {
                "thresh_fill": img
            }
            
            #サイズの設定
            self.height, self.width = img.shape[:2]
            self.height = int(self.height/mag)
            self.width = int(self.width/mag)
            
            self.images["raw"] = self.empty_image()
            self.images["gray"] = self.empty_image(isGray=True)
    
    # メイン関数
    def proc_image(self):
        if self.mode == "RAW":
            self.extend()
            self.extrude_noise()
            self.blur()
            self.thresh()
            self.fill_holes()
            self.set_contours(thickness=1, color=(255,255,255))
        else:
            self.set_contours(thickness=1, color=(255,255,255))

    def adaptive_proc_image(self, C=2):
        if self.mode == "RAW":
            self.extend()
            self.extrude_noise()
            self.blur()
            self.adaptive_thresh(C=C)
            self.fill_holes()
            self.set_contours(thickness=1, color=(255,255,255))
        else:
            self.set_contours(thickness=1, color=(255,255,255))


    def empty_image_extended(self, isGray=False):
        if isGray:
            return np.zeros((self.height*self.mag, self.width*self.mag), dtype=np.uint8)
        else:
            return np.zeros((self.height*self.mag, self.width*self.mag, 3), dtype=np.uint8)

    def empty_image(self, isGray=False):
        if isGray:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        else:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
    #画像の前処理(拡大)
    def extend(self):
        img = self.images["gray"]
        h, w = img.shape
        mag = self.mag
        
        if mag !=1:
            img_extended = cv2.resize(img, (w*mag, h*mag))
        else:
            img_extended = img
        
        self.images["extended"] = img_extended
        return img_extended

    #前処理 (ノイズ除去)
    def extrude_noise(self):
        img = self.images["extended"]
        img_med = cv2.medianBlur(img, 5)
        
        self.images["median_blur"] = img_med
        return img_med
    
    #画像の前処理(ぼかし)
    def blur(self):
        img = self.images["median_blur"]
        img_blur = cv2.GaussianBlur(img,(self.block_size, self.block_size),0)
        
        self.images["Gaussian_blur"] = img_blur
        return img_blur
    
    #2値画像を取得
    def thresh(self, threshold=0):
        img = self.images["Gaussian_blur"]

        ret,th = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #モルフォロジー変換(膨張)
        kernel = np.ones((3,3),np.uint8)
        th = cv2.dilate(th,kernel,iterations = 1)
        
        self.images["thresh"] = th
        return th
    
    def adaptive_thresh(self, C=2):
        img = self.images["Gaussian_blur"]
        th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.block_size, C)
        self.images["thresh"] = th
        return th

    #Fill Holes処理
    def fill_holes(self):
        th = self.images["thresh"]
        th_fill = ndimage.binary_fill_holes(th).astype(int) * 255

        self.images["thresh_fill"] = th_fill
        return th_fill

    # 輪郭をセットする
    def set_contours(self, color = (255,255,255), thickness=1):
        result = set_contours(
            self.images["raw"],
            self.images["thresh_fill"]
        )
        
        contours = result["contours"]
        img_cnt = result["image"]
        
        self.contours = contours
        self.images["contour"] = img_cnt
        return img_cnt

    #領域をマスク
    def mask_regions(self, image_target=None, contour_thickness=1):
        img_Region = self.images["thresh_fill"].copy()

        if image_target is None:
            image_target = np.zeros(img_Region.shape, dtype=np.uint8)

        # Positive領域
        img_positive = mask_image(image_target, img_Region, reverse=False)
        
        # Nagative領域: ちょっと膨らませて反転
        img_mask_negative = self.images["thresh_fill"].copy()
        img_mask_negative = cv2.cvtColor(img_mask_negative.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img_mask_negative = cv2.drawContours(img_mask_negative, self.contours, -1, color=(255,255,255), thickness=contour_thickness)        
        img_negative = mask_image(image_target, img_mask_negative, reverse=True)

        return img_positive, img_negative

    # 画像の輪郭を描画する
    def draw_contours(self, image_target,index=-1, color=(255,255,255), thickness=1):
        img = cv2.resize(image_target, (self.width*self.mag, self.height*self.mag))
        img = cv2.drawContours(img, self.contours, index, color=color, thickness=thickness)
        return img

    # ひとつの輪郭から画像をマスクする
    def mask_single_contour(self, image_target, index, reverse=False):
        img_empty = self.empty_image_extended()

        img_mask = cv2.drawContours(img_empty.copy(), self.contours, index, color=(255,255,255), thickness=1)
        img_mask = ndimage.binary_fill_holes(img_mask).astype(int) * 255
        img_mask = img_mask.astype(np.uint8)
        
        return  mask_image(image_target, img_mask, reverse=reverse)

    def analyze(self):
        #面積、円形度、等価直径を求める。
        self.result = []
        
        for i, contour in enumerate(self.contours):    
            result = analyze_contour(contour)
            
            if result["area"]:
                self.result.append(result)
            else:
                del self.contours[i]

        return self.result


# 画像の二値化処理
def binarize_image(image, mag=1):
    img_particles = Image_Particles(image, mag=mag)
    img_particles.proc_image()
    img_th = img_particles.images["thresh_fill"]
    return img_th

def adaptive_thresh_image(image, mag=1, block_size=11, C=2):
    img_particles = Image_Particles(image, mag=mag, block_size=block_size)
    img_particles.adaptive_proc_image(C=C)
    img_th = img_particles.images["thresh_fill"]
    return img_th

#バッチ処理
def batch_binarize_images(images, mag=1):
    img_stack = cv2.vconcat(images)
    img_th = binarize_image(img_stack, mag=mag)
    
    # 画像を分離
    imgs_spq = separate_image(img_th, split_x=len(images))
    return imgs_spq

def adaptive_batch_thresh_images(images, mag=1, block_size=11, C=2):
    img_stack = cv2.vconcat(images)
    img_th = adaptive_thresh_image(img_stack, mag=mag, block_size=block_size, C=C)
    
    # 画像を分離
    imgs_spq = separate_image(img_th, split_x=len(images))
    return imgs_spq

#結果を表示する

def plot_particle_info(Areas, Circularities, Eq_diameters):
    fig = plt.figure(figsize=(8,6))
    plt.subplot(2,2,1)
    plt.title("Areas (px^2)")
    plt.hist(Areas, bins=25, range=(0,150), rwidth=0.7)
    plt.subplot(2,2,2)
    plt.title("Circularity")
    plt.hist(Circularities, bins=25, range=(0.5,1), rwidth=0.7)
    plt.subplot(2,2,3)
    plt.title("Equal Diameters (px)")
    plt.hist(Eq_diameters, bins=25, range=(3.0, 15.0), rwidth=0.7)

    return plt


def plot_gaussian_plot(Eq_diameters):

    from scipy.optimize import curve_fit

    #numpyでヒストグラムを作成
    hist = np.histogram(Eq_diameters, bins=30, range=(3.0, 15.0))
    #ヒストグラムの中央値の配列を作成
    diff = np.diff(hist[1])
    hist_center = hist[1][:-1] + diff

    #フィッティング用のガウス関数を作成
    def Gaussian(x, *params):
        amp = params[0]
        wid = params[1]
        ctr = params[2]
        y = amp * np.exp( -((x - ctr)/wid)**2)
        return y

    #ガウシアンでフィッティング
    guess = [5000, 4, 9]
    params, cov = curve_fit(Gaussian, hist_center, hist[0], p0=guess)

    #強度, 半値幅, 平均値を表示
    fmhm = 2 * np.sqrt(2*np.log(2)) * params[1]
    print("Amplitude = "+str(params[0]))
    print("FMHM = "+str(fmhm))
    print("Mean = "+str(params[2]))

    #ヒストグラムと分布関数(ガウシアン)を重ねて表示
    plt.title("Equal Diameters (px)")
    plt.bar(hist_center, hist[0], width=0.3)
    x = np.linspace(5,15, 1000)
    plt.plot(x, Gaussian(x, *params), c="red",lw=2, ls="--")
    
    return plt

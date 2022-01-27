from oiffile import *
import json
import re
import os
import cv2
import datetime
import numpy as np

from module.utils.utils import save_json
from .image_mask import *
from .image_utils import *


# メタ情報を抽出
def extract_file_metadata(mainfile):
    display_data = mainfile["2D Display"]
    acq_data = mainfile["Acquisition Parameters Common"]
    ref_data = mainfile["Reference Image Parameter"]
    file_data = mainfile["File Info"]
    z_data = mainfile["Axis 3 Parameters Common"]
    
    dt = acq_data["ImageCaputreDate"]
    dt = "T".join(dt.split(" "))
    dt += "." + str(acq_data["ImageCaputreDate+MilliSec"]).zfill(3)
    #dt += "+0900"
    
    result = {
        "FileName": file_data["DataName"],
        "Datetime": dt,
        "Size": [ref_data["ImageWidth"], ref_data["ImageHeight"]],
        "SizeConverterValue": [ref_data["WidthConvertValue"], ref_data["HeightConvertValue"], z_data["Interval"]/1000],
        "Zoom": acq_data["ZoomValue"],
        "nSteps": int(display_data["Z End Pos"]) - int(display_data["Z Start Pos"]) + 1,
        "nChannels": int(display_data["View Max CH"]),
        "Channels" : [],
    }
    
    for i in range(1, result["nChannels"]+1):
        channel_data = mainfile[f"Channel {i} Parameters"]
        result["Channels"].append({
            "Channel_Number": channel_data['Physical CH Number'],
            "Gain" : channel_data['AnalogPMTGain'],
            "Offset" : channel_data['AnalogPMTOffset'],
            "HV" : channel_data['AnalogPMTVoltage'],
            "DyeName": channel_data['DyeName']
        })
        
    return result


# OIBファイルを読み込み
class OIBFile(object):
    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.result_folder = kwargs.get("result_folder", "result/OIB") 
        self.image_folder = kwargs.get("image_folder", "result/OIB")
        self.colors = kwargs.get("colors",{
            "FITC": colors["Green"], 
            "Cy3": colors["Magenta"], 
            "Cy5": colors["Blue"] 
        })

        self.images = {}
        self.images_raw = {}
        self.images_stacked = {}
        self.images_colored = {}

    # 空の画像
    def _empty_image(self, isGray=False, dtype=np.uint8):
        if isGray:
            return np.zeros([self.metadata["Size"][1], self.metadata["Size"][0]], dtype=dtype)
        else:
            return np.zeros([self.metadata["Size"][1], self.metadata["Size"][0], 3], dtype=dtype)
    
    # ファイルの読み込み
    def read(self):
        self.read_metadata()
        self.read_images()
    
    # メタ情報の読み込み
    def read_metadata(self):
        with OifFile(self.filename) as oib:
            self.metadata = extract_file_metadata(oib.mainfile)
            self.metadata_raw = oib.mainfile
            self.dye_names = [ channel["DyeName"] for channel in self.metadata["Channels"] ]
            
            date = datetime.datetime.fromisoformat(self.metadata["Datetime"]).strftime("%y%m%d")
            oibFileName = os.path.basename(self.filename).split('.', 1)[0]

            self.result_folder = f'{self.result_folder}/{date}_{oibFileName}'
            self.image_folder = f'{self.image_folder}/{date}_{oibFileName}'

            if len(self.metadata["Channels"]) == 2:
                self.colors["Cy5"] = colors["Green"]
        
            return self.metadata

    # 画像の読み込み
    def read_images(self):
        for DyeName in self.dye_names:
            self.images[DyeName] = []
        
        with OibFileSystem(self.filename) as oib:
            for file in oib.files():        
                if re.search(r"tif", file):
                    print("\rreading: ", file, end="")
                    
                    channel = re.findall(r"\d+", file)[1]
                    step = re.findall(r"\d+", file)[2]
                    dyeName = self.dye_names[int(channel)-1]
                    
                    image_data = oib.open_file(file).read()
                    image_data = binary_to_cv2(image_data, bit_from=12, bit_to=8)
                    self.images[dyeName].append(image_data)
            print("--done")

    # ファイルの保存
    def save(self):
        self.save_metadata()
        self.save_images()

    # JSONを保存
    def save_json(self, fileName, data):
        filePath = os.path.join(self.result_folder, f'{fileName}.json')
        save_json(filePath, data)

    # メタ情報をJSONで保存
    def save_metadata(self):
        self.save_json("metadata", self.metadata)
        self.save_json("metadata_raw", self.metadata_raw)

    # 画像を保存
    def save_images_by_key(self, key):
        for i , image in  enumerate(self.images[key]):
            filePath = os.path.join(self.image_folder, key, f'{str(i).zfill(3)}.tif')
            save_image(filePath, image)

    # 画像をすべて保存
    def save_images(self):
        for key in self.images.keys(): 
            print("\rsaving: ", key, end="")
            self.save_images_by_key(key)
        print("--done")
    
    #画像を正規化
    def normalize(self):
        for DyeName in self.dye_names:
            print("\rnormalizing: ", DyeName, end="")
            result = normalize_image(self.images[DyeName])
            self.images[DyeName+"_Normalized"] = result
        print("--done")
        return result
            
    # グレースケール画像を色付け
    def set_channel_color(self):        
        for DyeName in self.dye_names:
            self.images[f'{DyeName}_Colored'] = []
        
            for image in self.images[DyeName]:
                image_colored = set_image_color(image, self.colors[DyeName])
                self.images[f'{DyeName}_Colored'].append(image_colored)

    # カラー画像をマージ
    def merge_images(self):
        self.images['Merged'] = []

        for i in range(len(self.images[ self.dye_names[0] ])):
            image_merged = self._empty_image()
            for DyeName in self.dye_names:
                image_merged = cv2.add(image_merged, self.images[f'{DyeName}_Colored'][i])
            
            self.images['Merged'].append(image_merged)

    # スライス画像をスタック
    def stack_images_by_key(self, key, key_stacked):        
        z = z_stack_image(self.images[key])
        z = convert_image_bit_number(z, bit_from=12)
        self.images[key_stacked] = [z]
    
    # スライスのカラー画像をすべてスタック 
    def stack_images(self):        
        for DyeName in self.dye_names:
            self.stack_images_by_key(f'{DyeName}_Colored', f'{DyeName}_Stacked')
            
        if "Merged" in self.images:
            self.stack_images_by_key('Merged', f'Merged_Stacked')
    
    #メイン関数
    def convert_and_export(self):
        self.read_metadata()

        # 変換済みでないなら処理する
        if not os.path.exists(self.result_folder):
            self.read_images()
            self.set_channel_color()
            self.normalize()
            self.merge_images()
            self.stack_images()
            self.save()
        else:
            print("---already converted---")

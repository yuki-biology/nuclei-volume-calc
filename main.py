from numpy.core.getlimits import _register_known_types
from numpy.lib.function_base import extract
from module import *
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import re

TITLES = {
    "Volume_um3": "Volume [um3]",
    
    "Intensity_Cy3": "Cy3 Intensity [a.u.]",
    "Intensity_Cy5": "Cy5 Intensity [a.u.]",
    "Intensity_FITC": "FITC Intensity [a.u.]",
    
    "Intensity_FITC_per_Volume_um3": "FITC Intensity per Volume [/um3]",
    "Intensity_Cy5_per_Volume_um3": "Cy5 Intensity per Volume [/um3]",
    "Intensity_Cy3_per_Volume_um3": "Cy3 Intensity per Volume [/um3]",

    "Intensity_FITC_per_Intensity_Cy5": "FITC Intensity per Cy5 Intensity []",
    "Intensity_Cy3_per_Intensity_Cy5": "Cy3 Intensity per Cy5 Intensity []",
}


import glob
files = glob.glob("./result/OIB/**")
files = [file for file in files if "L186" in file and "Patch" in file]

print(files)

target_title  = "result_Cy5_Ad_Th_stat.csv"
data = [
    pd.read_csv(os.path.join(file, target_title), index_col=0) 
for file in files]

# 対象データ
x = ["positive", "negative"]  # x軸の値

# figureを生成する
fig = plt.figure(figsize=(18,18))

keys = ["Intensity_FITC", "Volume_um3", "Intensity_Cy5", "Intensity_Cy3", "Intensity_Cy3_per_Volume_um3", "Intensity_Cy3_per_Intensity_Cy5"]
for i, k in enumerate(keys):
    # axをfigureに設定する
    ax = fig.add_subplot(3, 2, i+1)
    
    values_positive = [d.at[k,"pos_mean"] for d in data]
    values_negative = [d.at[k,"neg_mean"] for d in data]

    for positive, negative in zip(values_positive, values_negative):
        ax.plot(x, [positive, negative], "-", c="#ff0000", linewidth=1)
        
    t,p,s = t_test_rel_compare(values_positive, values_negative)

    # 汎用要素を表示
    ax.grid(True)  # grid表示ON
    ax.set_title(TITLES[k] + "\n" + s)  # グラフタイトル

#グラフを表示
plt.show()

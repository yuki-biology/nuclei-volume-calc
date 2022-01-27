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

def load_data(folders, target_title  = "result_Cy5_Ad_Th_stat.csv"):
    data = [
        pd.read_csv(os.path.join(folder, target_title), index_col=0) 
    for folder in folders]
    return data

def stat_mosaic(data):
    df_stat = pd.DataFrame(index=data[0].index, columns=data[0].columns)
    for i in df_stat.index:
        v_pos = pd.DataFrame( [[data.loc[i,"pos_mean"]] for data in data] )[0]
        v_neg = pd.DataFrame( [[data.loc[i,"neg_mean"]] for data in data] )[0]
        t, p, s = t_test(v_pos, v_neg)
        df_stat.loc[i, "t-test p-value"] = p
        df_stat.loc[i, "significance"] = s
        t, p, s = t_test(v_pos, v_neg, option="single-sided")
        df_stat.loc[i, "t-test p-value (pos > neg)"] = p
        df_stat.loc[i, "significance (pos > neg)"] = s
        t, p, s = t_test(v_neg, v_pos, option="single-sided")
        df_stat.loc[i, "t-test p-value (neg > pos)"] = p
        df_stat.loc[i, "significance (neg > pos)"] = s
        df_stat.loc[i, "pos_mean"] = v_pos.mean()
        df_stat.loc[i, "neg_mean"] = v_neg.mean()
        df_stat.loc[i, "pos_std"] = v_pos.std()
        df_stat.loc[i, "neg_std"] = v_neg.std()
        df_stat.loc[i, "pos_median"] = v_pos.median()
        df_stat.loc[i, "neg_median"] = v_neg.median()
        df_stat.loc[i, "pos_min"] = v_pos.min()
        df_stat.loc[i, "neg_min"] = v_neg.min()
        df_stat.loc[i, "pos_max"] = v_pos.max()
        df_stat.loc[i, "neg_max"] = v_neg.max()
    return df_stat

def plot_mosaic(data):

    # 対象データ
    x = ["positive", "negative"]  # x軸の値

    # figureを生成する
    fig = plt.figure(figsize=(18,18))

    keys = ["Volume_um3", "Intensity_Cy3", "Intensity_Cy5", "Intensity_Cy3_per_Volume_um3", "Intensity_Cy3_per_Intensity_Cy5"]
    for i, k in enumerate(keys):
        # axをfigureに設定する
        ax = fig.add_subplot(3, 2, i+1)
        
        values_positive = [d.at[k,"pos_mean"] for d in data]
        values_negative = [d.at[k,"neg_mean"] for d in data]

        for positive, negative in zip(values_positive, values_negative):
            ax.plot(x, [1, negative/positive], "-", c="#ff0000", linewidth=1)
            
        t,p,s = t_test_rel_compare(values_positive, values_negative)

        # 汎用要素を表示
        ax.grid(True)  # grid表示ON
        ax.set_title(TITLES[k] + "\n" + s)  # グラフタイトル

    return plt


def main(files, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    data = load_data(files, target_title+".csv")
    
    plt=plot_mosaic(data)
    plt.savefig(os.path.join(save_folder, target_title+".jpg"))
    #plt.show()

    df_stat = stat_mosaic(data)
    df_stat.to_csv(os.path.join(save_folder, target_title+".csv"))


if __name__ == "__main__":
    import glob
    _files = glob.glob("./result/OIB/**")
    target_title = "result_Cy5_Ad_Th_stat"

    #L186モザイク
    save_folder = "./result/stats/L186_mosaic"
    files = [file for file in _files if "L186" in file and "Patch" in file]
    print(files)
    main(files, save_folder)

    #E181モザイク
    save_folder = "./result/stats/E181_mosaic"
    files = [file for file in _files if "E181" in file and "Patch" in file]
    print(files)
    main(files, save_folder)

    #E38モザイク
    save_folder = "./result/stats/E38_mosaic"
    files = [file for file in _files if "E38" in file and "Patch" in file]
    print(files)
    main(files, save_folder)

    # Ay>InR[DN]モザイク
    save_folder = "./result/stats/Ay_InR_DN_mosaic"
    files = [file for file in _files if "Ay" in file and "inr" in file.lower()]
    print(files)
    main(files, save_folder)

    # Ay>TorIRモザイク
    save_folder = "./result/stats/Ay_Tor_IR_mosaic"
    files = [file for file in _files if "Ay" in file and "tor" in file.lower()]
    print(files)
    main(files, save_folder)

    # Ay>DveIRモザイク
    save_folder = "./result/stats/Ay_Dve_IR_mosaic"
    files = [file for file in _files if "Ay" in file and "dve" in file.lower() and file and "ir" in file.lower()]
    print(files)
    main(files, save_folder)
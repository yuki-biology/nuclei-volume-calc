
from numpy.core.getlimits import _register_known_types
from numpy.lib.function_base import extract
from module import *
import glob

class BackGorundSub(object):
    def __init__(self, folder, **kwargs):
        self.sample_folder = folder
        self.images= {}
        self.regions= {}
        self.images_proced = {}
        self.result = []
        self.metadata = None

    def load(self):
        self.load_images()
        self.load_normalized_images()

    def load_metadata(self):
        filepath  = os.path.join(self.sample_folder,"metadata.json")
        self.metadata = load_json(filepath)
        return self.metadata

    def load_images(self):
        if not self.metadata:
            self.load_metadata()
        for channel in self.metadata["Channels"]:
            DyeName = channel["DyeName"]
            self.load_images_by_prefix(DyeName, DyeName)
        return self.images
    
    def load_normalized_images(self):
        if not self.metadata:
            self.load_metadata()
        for channel in self.metadata["Channels"]:
            DyeName = channel["DyeName"] + "_Normalized"
            self.load_images_by_prefix(DyeName, DyeName)
        return self.images
    
    def load_images_by_prefix(self, prefix = "Cy5", image_key="Cy5"):
        paths = glob.glob(os.path.join(self.sample_folder,prefix,"*.tif"))
        result = []
        for path in paths:
            print(f"\rloading: {path}", end="")
            result.append(cv2.imread(path))
        print("--done")
        self.images[image_key] = result
        return result

    def save(self):
        self.save_regions()
    
    def _save_images_dict(self, images, prefix= "Region_"):
        for k, v in images.items():
            for i, image in enumerate(v):
                filePath = os.path.join(self.sample_folder, prefix+k, f"{str(i).zfill(3)}.tif")
                print("\rSaving: "+filePath, end="")
                save_image(filePath, image)
            print("--done")

    def save_regions(self):
        self._save_images_dict(self.regions, prefix= "Region_")

    # 背景を除去
    def substract_background(self, DyeName="Cy5_Normalized", kernel_sizz_um=10):
        print("\rSubstracting background...", end="")
        images = self.images[DyeName]
        kernel_size = 1 / self.metadata["SizeConverterValue"][0] * kernel_sizz_um #10umだけ膨張させる
        result_b, result_s = batch_background_substract_open(images, kernel_size)
        self.regions[f"{DyeName}_Background"] = result_b
        self.regions[f"{DyeName}_Subtracted"] = result_s
        print("--done")
        return result_b, result_s

    # 二値化処理
    def binarize(self, RegionKey="Cy5_Normalized_Subtracted", mag=5):
        print("\rBinarize...", end="")
        images = self.regions[RegionKey]
        result = batch_binarize_images(images, mag)
        self.regions[RegionKey + "_Particle"] = result
        print("--done")
        return result


if __name__ == "__main__":
    import glob
    files = glob.glob("./result/OIB/**")
    #files = ["./result/OIB/211201_Ay-DveIR_HS96hrs_Fat-Larva_#2"]
    files = ["./result/OIB/190406_CG-AMPK_FAT-Larva_x10_#1"]
    print(files)
    
    #不鮮明なデータのテスト
    #files = ["result/OIB/211128_Ay-DveIR-HS24hrs_FAT-Larva_#1"]

    @handle_elaped_time(len(files))
    def convert_oib_to_data(i):
        print("converting: " + files[i])
        bgs = BackGorundSub(files[i])
        bgs.load()
        bgs.substract_background()
        #bgs.binarize()
        bgs.save()
        #print(bgs.result)
    
    convert_oib_to_data()
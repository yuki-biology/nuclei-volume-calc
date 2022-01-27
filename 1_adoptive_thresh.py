
from numpy.core.getlimits import _register_known_types
from numpy.lib.function_base import extract
from module import *
import glob

class BackGorundSub(ImageProc):
    def __init__(self, folder, **kwargs):
        super().__init__(folder, **kwargs)

    # 二値化処理
    def adoptive_binarize(self, RegionKey="Cy5", mag=3):
        images = self.images[RegionKey]
        kernel_size = 1 / self.metadata["SizeConverterValue"][0] * 25
        
        result = []
        for i, img in enumerate(images):
            print("\rprocessing: {}".format(i), end="")
            result.append(adaptive_threshold(img, block_size = kernel_size, mag=mag))
        print("--done")
        self.regions[RegionKey + "_Adoptive_Thresh"] = result
        return result
    
    def proc(self):
        self.load_metadata()
        self.load_images_by_prefix("Cy5_Normalized", "Cy5")
        self.adoptive_binarize()
        self.save_regions()
        return self.regions


if __name__ == "__main__":
    import glob
    files = glob.glob("./result/OIB/**")
    #files = ["./result/OIB/211201_Ay-DveIR_HS96hrs_Fat-Larva_#2"]
    #files = ["./result/OIB/190406_CG-AMPK_FAT-Larva_x10_#1"]
    print(files)
    
    @handle_elaped_time(len(files))
    def convert_oib_to_data(i):
        print("converting: " + files[i])
        bgs = BackGorundSub(files[i])
        bgs.proc()
    
    convert_oib_to_data()
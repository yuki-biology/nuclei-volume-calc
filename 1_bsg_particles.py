
from numpy.core.getlimits import _register_known_types
from numpy.lib.function_base import extract
from module import *
import glob

class BackGorundSub(object):
    def __init__(self, folder, **kwargs):
        self.sample_folder = folder
        self.images= {}
        self.regions= {}
        self.result = []
        self.metadata = None

    def load(self):
        self.load_images()

    def load_metadata(self):
        filepath  = os.path.join(self.sample_folder,"metadata.json")
        self.metadata = load_json(filepath)
        return self.metadata

    def load_images(self):
        if not self.metadata:
            self.load_metadata()
        
        for channel in self.metadata["Channels"]:
            DyeName = channel["DyeName"]

            paths = glob.glob(os.path.join(self.sample_folder,DyeName,"*.tif"))
            self.images[DyeName] = []
            for path in paths:
                print(f"\rloading: {path}", end="")
                self.images[DyeName].append(cv2.imread(path))

            print("--done")
        return self.images

    def save(self):
        self.save_result()
        self.save_regions()

    def save_result(self):
        filepath = os.path.join(self.sample_folder,"result_bsg.json")
        save_json(filepath, self.result)
        return self.result
    
    def save_regions(self):
        for k in self.regions.keys():
            self.save_regions_by_key(k)

    def save_regions_by_key(self, key):
        for i, image in enumerate(self.regions[key]):
            filePath = os.path.join(self.sample_folder, "Region_"+key, f"{str(i).zfill(3)}.tif")
            save_image(filePath, image)

    def analyze_images_by_bsg(self, DyeName="Cy5", mag = 5):
        images = self.images[DyeName]
        kernel_size = self.metadata["SizeConverterValue"][0]
        result = background_substract_KNN(images, mag, kernel_size)
        self.regions[f"{DyeName}_BSG"] = result
        return result

    def detect_3D_regions(self, RegionKey="Cy5_BSG", connectivity=6, size_max =1e+4, size_min = 200):
        images = self.regions[RegionKey]
        result = cc3d_detect_regions(images, connectivity)
        self.regions[RegionKey+"_CC3D"] = result["Labels"]
        self.result = result["Values"]
        
        for i in range(len(self.result)):
            self.result[i]["Volume_um3"] = self.result[i]["Volume_px3"] * self.metadata["SizeConverterValue"][0] * self.metadata["SizeConverterValue"][1] * self.metadata["SizeConverterValue"][2]
        
        self.result = [i for i in self.result if i["Volume_um3"] < size_max and i["Volume_um3"] > size_min]
        
        return result
    
    def set_labels_color(self, Key = "Cy5_BSG_CC3D"):
        res = set_random_label_color(self.regions[Key])
        self.regions[Key+ "_Colored"] = res
        return res
    
    def calc_region_intensity(self, RegionKey="Cy5_BSG", DyeName="Cy3"):
        region_lebels = self.regions[RegionKey + "_CC3D"]
        images = self.images[DyeName]

        print("calc region intensity: ", DyeName)
        for v in self.result:
            print("\rId: ", v["Id"], end="")
            _region_lebels = extract_label_3D(region_lebels, v["Id"])
            intensity = count_masked_region_intensity_3d(images, _region_lebels)
            v["Intensity_" + DyeName] = intensity
        print("--done")

if __name__ == "__main__":
    import glob
    files = glob.glob("./result/OIB/**")
    
    print(files)
    
    #不鮮明なデータのテスト
    #files = ["result/OIB/211128_Ay-DveIR-HS24hrs_FAT-Larva_#1"]

    @handle_elaped_time(len(files))
    def convert_oib_to_data(i):
        print("converting: " + files[i])
        bgs = BackGorundSub(files[i])
        bgs.load()
        bgs.analyze_images_by_bsg()
        bgs.detect_3D_regions()
        bgs.set_labels_color()
        if "Cy3" in bgs.images:
            bgs.calc_region_intensity(DyeName="Cy3")
        if "Cy5" in bgs.images:
            bgs.calc_region_intensity(DyeName="Cy5")
        if "FITC" in bgs.images:
            bgs.calc_region_intensity(DyeName="FITC")
        bgs.save()
        #print(bgs.result)
    
    convert_oib_to_data()
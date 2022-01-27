from numpy.core.getlimits import _register_known_types
from numpy.lib.function_base import extract
from module import *
import glob

class DetectRegion(ImageProc):
    def __init__(self, folder, targetFolder, targetTitle= "Cy5_CC3D", **kwargs):
        super().__init__(folder, **kwargs)
        self.targetFolder = targetFolder
        self.targetTitle = targetTitle

    def load_target(self):
        self.load_images_by_prefix(self.targetFolder, self.targetTitle, gray=True)
        return self.images[self.targetTitle]

    def detect_3D_regions(self, connectivity=6, size_max =1e+4, size_min = 200):
        Key = self.targetTitle + "_CC3D"
        images = self.images[self.targetTitle]
        result = cc3d_detect_regions(images, connectivity)
        self.regions[Key] = result["Labels"]
        self.result = result["Values"]
        
        for i in range(len(self.result)):
            self.result[i]["Volume_um3"] = self.result[i]["Volume_px3"] * self.metadata["SizeConverterValue"][0] * self.metadata["SizeConverterValue"][1] * self.metadata["SizeConverterValue"][2]
        
        found = len(self.result)
        self.result = [i for i in self.result if i["Volume_um3"] < size_max and i["Volume_um3"] > size_min]
        print("found: ", found, " after filtering: ", len(self.result), "excluded: ", found - len(self.result))
        return result
    
    def set_labels_color(self):
        Key = self.targetTitle + "_CC3D"
        res = set_random_label_color(self.regions[Key])
        self.regions[Key+ "_Colored"] = res
        return res
    
    def calc_region_intensity(self, DyeName="Cy3"):
        Key = self.targetTitle + "_CC3D"
        region_lebels = self.regions[Key]

        if DyeName in self.images:
            images = self.images[DyeName]
            print("calc region intensity: ", DyeName)
            for v in self.result:
                print("\rId: ", v["Id"], end="")
                _region_lebels = extract_label_3D(region_lebels, v["Id"])
                intensity = count_masked_region_intensity_3d(images, _region_lebels)
                v["Intensity_" + DyeName] = intensity
            print("--done")
        else:
            print("No image: ", DyeName)
    
    def proc(self):
        self.load()
        self.load_target()
        self.detect_3D_regions()
        self.set_labels_color()
        self.calc_region_intensity("Cy3")
        self.calc_region_intensity("Cy5")
        self.calc_region_intensity("FITC")
        self.save_result("result_" + self.targetTitle +".json")
        self.save_regions()


if __name__ == "__main__":
    targetFolder = "Region_Cy5_WS"
    targetTitle = "Cy5_Ad_Th"
    
    import glob
    files = glob.glob("./result/OIB/**")
    print(files)
    

    @handle_elaped_time(len(files))
    def convert_oib_to_data(i):
        print("converting: " + files[i])
        bgs = DetectRegion(files[i], targetFolder=targetFolder, targetTitle=targetTitle)
        bgs.proc()
    
    convert_oib_to_data()


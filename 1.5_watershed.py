from numpy.core.getlimits import _register_known_types
from numpy.lib.function_base import extract
from module import *
import glob

class Watershed(ImageProc):
    def __init__(self, folder, targetFolder, targetTitle= "Cy5_WS", **kwargs):
        super().__init__(folder, **kwargs)
        self.targetFolder = targetFolder
        self.targetTitle = targetTitle

    def load_target(self):
        self.load_images_by_prefix(self.targetFolder, self.targetTitle, gray=True)
        return self.images[self.targetTitle]
    
    def watershed(self):
        kernel_size = 1 / self.metadata["SizeConverterValue"][0] * 10
        print("Watershed")
        images = self.images[self.targetTitle]
        result = []
        for i in range(len(images)):
            print("\rImage: ", i, end="")
            image = images[i]
            image = watershed_separate(image, kernel_size)
            result.append(image)
        self.regions[self.targetTitle] = result
        print("\n--done")
        return result
    
    def proc(self):
        self.load_metadata()
        self.load_target()
        self.watershed()
        self.save_regions()

if __name__ == "__main__":
    targetFolder = "Region_Cy5_Adoptive_Thresh"
    targetTitle = "Cy5_WS"
    
    import glob
    files = glob.glob("./result/OIB/**")
    print(files)
    

    @handle_elaped_time(len(files))
    def convert_oib_to_data(i):
        print("converting: " + files[i])
        bgs = Watershed(files[i], targetFolder=targetFolder, targetTitle=targetTitle)
        bgs.proc()
    
    convert_oib_to_data()

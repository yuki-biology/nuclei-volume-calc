from .image_particle import *
from .image_utils import *
from .image_mask import *
from .oib import OIBFile
from .stats.plot import *

from .utils import *
from .image_utils import *
from .stats import *
from .cc3d import *

from .image_background_substract import *
from .image_particle_adaptive import *

from .watershed import *

# CC3Dのラベル連続画像から、領域の値を取得する
def region_intensity_from_cc3d_label(imgs, region_labels):
    param_list = np.linspace(0,np.max(region_labels),np.max(region_labels)+1,dtype=np.uint16)
    result = []
    for i in param_list:
        print(f"\rcalc intensity at label_{i}", end="")
        result.append({
            "Id": i,
            "intensity": count_masked_region_intensity_3d(imgs, extract_label_3D(region_labels, i))
        })
    print("--done")
    return result

import glob
class ImageProc(object):
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
    
    def load_images_by_prefix(self, prefix = "Cy5", image_key="Cy5", gray=False):
        paths = glob.glob(os.path.join(self.sample_folder,prefix,"*.tif"))
        result = []
        for path in paths:
            print(f"\rloading: {path}", end="")
            img = cv2.imread(path)
            if gray and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result.append(img)
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

    def save_result(self, filename = "result.json"):
        filepath = os.path.join(self.sample_folder, filename)
        save_json(filepath, self.result)
        return self.result

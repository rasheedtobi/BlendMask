        
"""This functions is to run inference on the whoe test dataset with the trained model. You may want to change
1. the model path with MODEL.WEIGHTS
2. the names of registered coco instances
3, the dirctory where the image files are saved

"""


import pandas as pd
import torch
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, pairwise_iou
from adet.config import get_cfg
#from adet.checkpoint import AdetCheckpointer
from demo.predictor import VisualizationDemo
from detectron2.config import CfgNode







DatasetCatalog.clear()
MetadataCatalog.clear()


# load config from file and command-line arguments
cfg = get_cfg()

cfg.merge_from_file('configs/BlendMask/R_50_1x.yaml')
# Set score_threshold for builtin models
#config_file_pth = 'BlendMask/R_50_1x.yaml'
cfg.set_new_allowed(False)
conf =0.3
#cfg.merge_from_file = 

cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.WEIGHTS = 'training_dir/vac_all_aug_blendmask_R_50_1x/model_final.pth'
cfg.MODEL.RETINANET.SCORE_THRESH_TEST =conf
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf
cfg.MODEL.FCOS.INFERENCE_TH_TEST = conf
cfg.MODEL.MEInst.INFERENCE_TH_TEST = conf
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = conf
#cfg.MODEL.MASK_ON= True
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.freeze()

predictorr=VisualizationDemo(cfg)


#predictorr=DefaultPredictor(cfg)

CLASS_NAMES = ["Last Tip Cutting", "Section Cutting", "V Cutting", "Bottom End", "Discarded Cutting"]	
val_json = "datasets/coco/annotations/instances_val2017.json"
val_imgs = 'datasets/coco/val2017'

train_json = "datasets/coco/annotations/instances_train2017.json"
train_imgs = 'datasets/coco/train2017'

register_coco_instances("vacc_train", {}, 'datasets/coco/annotations/instances_train2017.json', 'datasets/coco/train2017')
MetadataCatalog.get("vacc_train").set(json_file=train_json,
	                           thing_classes = CLASS_NAMES,
	                           image_root=train_imgs,
	                           evaluator_type="coco")


register_coco_instances("vacc_val", {}, val_json, val_imgs)
MetadataCatalog.get("vacc_val").set(thing_classes = CLASS_NAMES,
	            json_file= val_json,
	                        image_root=val_imgs,
	                        evaluator_type="coco")

register_coco_instances("vacc_test", {}, 'datasets/coco/annotations/instances_test2017.json', 'datasets/coco/test2017')
#custom_metatdata = MetadataCatalog.get("vacc_val")

dataset_custom = DatasetCatalog.get('vacc_test')
dataset_custom_metadata = MetadataCatalog.get('vacc_test')
count =0
for d in dataset_custom:
    a= d["file_name"]
    b = os.path.split(a)
    c=os.path.join("vac_all_aug_40k_blendmask_R_50_1x",b[1])
    img = cv2.imread(a)
    #img = img[:, :, ::-1]
    outputs, vs= predictorr.run_on_image(img)
    
    #v = Visualizer(img[:,:,::-1], metadata =dataset_custom_metadata, scale = 0.5)
   
    
    #v= v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2_imshow(out.get_image()[:, :, ::-1], 'img')
    plt.figure(figsize = (30,45))
    plt.imshow(vs.get_image())
    print(a)
    plt.savefig(c)

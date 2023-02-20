from utilis import *

CLASS_NAMES = class_definition(["Last Tip Cutting", "Section Cutting", "V Cutting", "Bottom End", "Discarded Cutting"])

val_json = "datasets/coco/annotations/instances_val2017.json"
val_imgs = 'datasets/coco/val2017'

test_json = "datasets/coco/annotations/instances_test2017.json"
test_imgs = 'datasets/coco/test2017'

inference_on_dataset(0.3, 'configs/BlendMask/R_50_1x.yaml', 'training_dir/vac10_4_lr0_01blendmask_R_50_1x/model_final.pth', CLASS_NAMES,val_json, val_imgs,test_json, test_imgs, "test_what_is_important2")

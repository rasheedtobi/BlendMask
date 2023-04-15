import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
from detectron2.data.datasets import register_coco_instances
import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode


from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.data.fcpose_dataset_mapper import FCPoseDatasetMapper
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import numpy as np

import PIL
from PIL import Image

import demo
import pandas as pd

from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, pairwise_iou
from adet.config import get_cfg
#from adet.checkpoint import AdetCheckpointer
from demo.predictor import VisualizationDemo
from detectron2.config import CfgNode
import os, json, cv2, random

#
from opencv_jupyter_ui import cv2_imshow
import matplotlib.pyplot as plt





#To plot groud truth annotation for a specific image named img_name
def plot_spec_anno_ (dataset_name, img_name):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
      
    for s in dataset_custom:
        if(s["file_name"] == img_name):
            img = cv2.imread(img_name)
            v = Visualizer(img[:,:,::-1], metadata =dataset_custom_metadata, scale = 0.5)
            v = v.draw_dataset_dict(s)
            plt.figure(figsize = (22,36))
            plt.imshow(v.get_image())
            #plt.savefig('file_pth/a')
            #plt.show(0)
        
#To plot groud truth annotations and saved in the directory with name dir_nam
#dataset_name must have been registered
def plot_groundt_anno (dataset_name, dir_name): #dataset_name must have been registered


    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
    
    if not os.path.exists(dir_name):
    	# Use os.makedirs to create the folder and any necessary parent folders
    	os.makedirs(dir_name)
    
   
    for s in dataset_custom:
        a= s["file_name"]
        b = os.path.split(a)
        img = cv2.imread(a)
        v = Visualizer(img[:,:,::-1], metadata =dataset_custom_metadata, scale = 0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize = (22,36))
        plt.imshow(v.get_image())
        c=os.path.join(dir_name,b[1])
        plt.savefig(c)      


#To plot random samples from a registered dataset            
def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
    
    for s in random.sample(dataset_custom, n):
        print(s["file_name"])
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15, 24))
        plt.imshow(v.get_image())
        plt.show(0)

#returns object class as a list
def class_definition(obj_cls):
	
	return obj_cls
	
	
        
def register_data(name,json_file_pth, img_file_pth):
	register_coco_instances(name, {}, json_file_pth, img_file_pth)
	
	
	
#To register dataset in line with framework guidlines #Useful to plot ground truth annotations	
def register_data_with_meta(name, json_file_pth, img_file_pth, CLASS_NAMES):
	obj_classes =[]
	
	register_coco_instances(name, {}, json_file_pth, img_file_pth)
	MetadataCatalog.get(name).set(thing_classes = CLASS_NAMES,
		            json_file= json_file_pth,
		                        image_root=img_file_pth,
		                        evaluator_type="coco")
#To make prediction on new image 	
def inference_on_img(conf, yaml_file, weights, CLASS_NAMES,val_json, val_imgs, img):
	DatasetCatalog.clear()
	MetadataCatalog.clear()

	

	cfg = get_cfg()

	cfg.merge_from_file(yaml_file)
	cfg.set_new_allowed(False)

	cfg.MODEL.DEVICE = 'cpu'
	cfg.MODEL.WEIGHTS = weights
	cfg.MODEL.FCOS.INFERENCE_TH_TEST = conf
	
	cfg.DATASETS.TEST = ('vacc_val',)
	

	cfg.freeze()
	predictorr=VisualizationDemo(cfg)
	
	register_coco_instances("vacc_val", {}, val_json, val_imgs)
	MetadataCatalog.get("vacc_val").set(thing_classes = CLASS_NAMES,
		            json_file= val_json,
		                        image_root=val_imgs,
		                        evaluator_type="coco")
		                        

	image = cv2.imread(img)
	outputs, vs= predictorr.run_on_image(image)
	    
	plt.figure(figsize = (30,45))
	plt.imshow(vs.get_image())


#To make prediction on images on a dataset "test_imgs"
def inference_on_dataset(conf, yaml_file, weights, CLASS_NAMES,val_json, val_imgs, test_imgs, path_to_save):
	DatasetCatalog.clear()
	MetadataCatalog.clear()


	cfg = get_cfg()

	cfg.merge_from_file(yaml_file)
	cfg.set_new_allowed(False)

	cfg.MODEL.DEVICE = 'cpu'
	cfg.MODEL.WEIGHTS = weights
	cfg.MODEL.FCOS.INFERENCE_TH_TEST = conf

	cfg.freeze()
	predictorr=VisualizationDemo(cfg)
	

	register_coco_instances("vacc_val", {}, val_json, val_imgs)
	MetadataCatalog.get("vacc_val").set(thing_classes = CLASS_NAMES,
		            json_file= val_json,
		                        image_root=val_imgs,
		                        evaluator_type="coco")
		                        
	
	
	
	file_list = []
	# loop over the files in the directory and append their names to the list
	for filename in os.listdir(test_imgs):
		if os.path.isfile(os.path.join(test_imgs, filename)):
			file_list.append(os.path.join(test_imgs, filename))
	
	if not os.path.exists(path_to_save):
		# Use os.makedirs to create the folder and any necessary parent folders
		os.makedirs(path_to_save)
	for d in file_list:
	    
	    b = os.path.split(d)
	    c=os.path.join(path_to_save,b[1])
	    img = cv2.imread(d)
	    #img = img[:, :, ::-1]
	    outputs, vs= predictorr.run_on_image(img)
	    
	    
	    plt.figure(figsize = (30,45))
	    plt.imshow(vs.get_image())
	    print(d)
	    plt.savefig(c)
	    
	    
#To visualise ground truth and prediction in one window    
def comb_imgs(groundtruth_path, inference_path, new_path):
	gt_files =[]
	test_imgs =[]
	path1 = groundtruth_path
	path2 = inference_path
	
	if not os.path.exists(new_path):
		os.makedirs(new_path)
	for x in os.listdir(path1):
	    gt_files.append(os.path.join(path1,x))
	for y in os.listdir(path2):
	    test_imgs.append(os.path.join(path2,y))
	count =0
	for d,b in zip(gt_files, test_imgs):
	    list_im = [d, b] 
	    imgs    = [Image.open(i) for i in list_im]
	    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
	    imgs_comb = np.hstack([i.resize(min_shape) for i in imgs])
	    #save that beautiful picture
	    imgs_comb = Image.fromarray( imgs_comb)
	    imgs_comb.save( os.path.join(new_path, os.path.split(b)[1]))
 


	

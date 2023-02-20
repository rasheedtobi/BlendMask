-------------------------------------------------------READ ME------------------------------------------------------------------------------

The Blendmask implementaion can be done using the aim-uofa/AdelaiDet git repository - https://github.com/aim-uofa/AdelaiDet - This repository contains details of how some image object detection and segmentation models can be implemented 
using Meta Detectron2 model as a base. The steps involved in the implementation of the Blendmask instance sehgmentation methods are detaled as follows:

1. Install pytorch compatible with Cuda version on the pc and also the one that works for Detectron2. It is important to check the latest version of Torch Detectron2 supports and check if there is a corresponding CUDA tolkit available. 
Older combinations worked E.g. pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

2. Install pyyaml. E.g pip install pyyaml==5.1 

3. Install detectron2 using the correct combinatinon of torch and CUDA as seen on https://detectron2.readthedocs.io/en/latest/tutorials/install.html#common-installation-issues E.g. -   pip install 'git+https://github.com/facebookresearch/detectron2.git -
NOTE: It is very important to get the correct combination of CUDA toolkit and torch before installing Detectron2. Installing torch and Cuda after installing Detectron2 may produce undesirable result.

4. Navigate to AdelaiDET and run the command: 'python setup.py build develop'

5. A demo can be run to see if the AdelaiDEt:
	wget -O blendmask_r101_dcni3_5x.pth https://cloudstor.aarnet.edu.au/plus/s/vbnKnQtaGlw8TKv/download
	python demo/demo.py \
   	 --config-file configs/BlendMask/R_101_dcni3_5x.yaml \
    	--input datasets/coco/val2017/000000005992.jpg \
    	--confidence-threshold 0.35 \
    	--opts MODEL.WEIGHTS blendmask_r101_dcni3_5x.pth


6. For the traininng proper, the following have to be taken care of:
	I.	 The dataset has to be in the specified format. E>g. COCO dataset format.
	II.	 While still in the AdelaiDet directory run the command - "python datasets/prepare_thing_sem_from_instance.py" to get .npz files of the images to extract relevant information from json
	III.	 If only one GPU is being used,  navigate to AdelaiDET/defaults.py, and edit to be _C.MODEL.BASIS_MODULE.NORM = "BN".
	IV. 	 Hyperparametere settings can be made using the AdelaiDet/configs/BlendMask/Base-BlendMask.yaml. 
	V. 	 Again in Base-BlendMask it is important to use the correct registered dataset name otherwise incorrect class labels will be used on images The default values 
		 DATASETS:
		 #TRAIN: ("coco_2017_train",)
		 #TEST: ("coco_2017_val",)


7. To train use the

 OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file configs/BlendMask/R_50_1x.yaml \
    --num-gpus 4 \
    OUTPUT_DIR training_dir/blendmask_R_50_1x

8. To add the augmentations the DatasetMapper.py can be edited accordingly.





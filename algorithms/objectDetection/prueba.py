import os 
import sys
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import pydicom
from imgaug import augmenters as iaa
from tqdm import tqdm
import pandas as pd 
import glob 
import tensorflow as tf

DATA_DIR = '../../data'
ROOT_DIR = './'

# Import Mask RCNN
sys.path.append(os.path.join(ROOT_DIR, 'Mask_RCNN'))  # To find local version of the library
from mrcnn import config
from mrcnn.config import Config
from mrcnn import visualize
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log


train_dicom_dir = os.path.join(DATA_DIR, 'stage_1_train_images')
test_dicom_dir = os.path.join(DATA_DIR, 'stage_1_test_images')




def get_dicom_fps(dicom_dir):
    dicom_fps = glob.glob(dicom_dir+'/'+'*.dcm')
    return list(set(dicom_fps))

def parse_dataset(dicom_dir, anns): 
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows(): 
        fp = os.path.join(dicom_dir, row['patientId']+'.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations 



class DetectorConfig(Config):
    """Configuration for training pneumonia detection on the RSNA pneumonia dataset.
    Overrides values in the base Config class.
    """
    
    # Give the configuration a recognizable name  
    NAME = 'pneumonia'
    
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    
    BACKBONE = 'resnet50'
    
    NUM_CLASSES = 2  # background + 1 pneumonia classes
    
    IMAGE_MIN_DIM = 256.
    IMAGE_MAX_DIM = 256.
    RPN_ANCHOR_SCALES = (32, 64, 128, 256)
    TRAIN_ROIS_PER_IMAGE = 64
    MAX_GT_INSTANCES = 8
    DETECTION_MAX_INSTANCES = 8
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.2
    STEPS_PER_EPOCH = 1000
    
config = DetectorConfig()
config.display()


class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.
    """

    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        
        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')
   
        # add images 
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp, 
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)
            
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        fp = info['path']
        ds = pydicom.read_file(fp)
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x = int(a['x'])
                    y = int(a['y'])
                    w = int(a['width'])
                    h = int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x+w, y+h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)


with tf.device('/gpu:0'):


	# training dataset
	anns = pd.read_csv(os.path.join(DATA_DIR, 'stage_1_train_labels.csv'))
	image_fps, image_annotations = parse_dataset(train_dicom_dir, anns=anns)
	ds = pydicom.read_file(image_fps[0]) # read dicom image from filepath 
	#image = ds.pixel_array # get image array
	#print ds

	ORIG_SIZE = 1024



	#####################################################################
	# Modify this line to use more or fewer images for training/validation. 
	# To use all images, do: image_fps_list = list(image_fps)

	image_fps_list = list(image_fps) 
	#####################################################################

	# split dataset into training vs. validation dataset 
	# split ratio is set to 0.9 vs. 0.1 (train vs. validation, respectively)
	sorted(image_fps_list)
	random.seed(42)
	random.shuffle(image_fps_list)

	validation_split = 0.01
	split_index = int((1 - validation_split) * len(image_fps_list))

	image_fps_train = image_fps_list[:split_index]
	image_fps_val = image_fps_list[split_index:]

	print(len(image_fps_train), len(image_fps_val))


	# prepare the training dataset
	dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
	dataset_train.prepare()

	# Show annotation(s) for a DICOM image 
	test_fp = random.choice(image_fps_train)
	image_annotations[test_fp]

	# prepare the validation dataset
	dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
	dataset_val.prepare()


	model = modellib.MaskRCNN(mode='training', config=config, model_dir=ROOT_DIR)
	from keras import backend as K
	K.tensorflow_backend._get_available_gpus()

	# Image augmentation 
	augmentation = iaa.SomeOf((0, 1), [
	    iaa.Fliplr(0.5),
	    iaa.Affine(
	        scale={"x": (0.7, 1.4), "y": (0.7, 1.4)},
	        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
	        rotate=(-25, 25),
	        shear=(-8, 8)
	    ),
	    iaa.Multiply((0.9, 1.1))
	])

	NUM_EPOCHS = 100

	# Train Mask-RCNN Model 
	import warnings 
	warnings.filterwarnings("ignore")
	model.train(dataset_train, dataset_val, 
	            learning_rate=config.LEARNING_RATE, 
	            epochs=NUM_EPOCHS, 
	            layers='all',
	            augmentation=augmentation)



	# select trained model 
	dir_names = next(os.walk(model.model_dir))[1]
	key = config.NAME.lower()
	dir_names = filter(lambda f: f.startswith(key), dir_names)
	dir_names = sorted(dir_names)

	if not dir_names:
	    import errno
	    raise FileNotFoundError(
	        errno.ENOENT,
	        "Could not find model directory under {}".format(self.model_dir))
	    
	fps = []
	# Pick last directory
	for d in dir_names: 
	    dir_name = os.path.join(model.model_dir, d)
	    # Find the last checkpoint
	    checkpoints = next(os.walk(dir_name))[2]
	    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
	    checkpoints = sorted(checkpoints)
	    if not checkpoints:
	        print('No weight files in {}'.format(dir_name))
	    else: 
	      
	      checkpoint = os.path.join(dir_name, checkpoints[-1])
	      fps.append(checkpoint)

	model_path = sorted(fps)[-1]
	print('Found model {}'.format(model_path))

	class InferenceConfig(DetectorConfig):
	    GPU_COUNT = 1
	    IMAGES_PER_GPU = 1

	inference_config = InferenceConfig()

	# Recreate the model in inference mode
	model = modellib.MaskRCNN(mode='inference', 
	                          config=inference_config,
	                          model_dir=ROOT_DIR)

	# Load trained weights (fill in path to trained weights here)
	assert model_path != "", "Provide path to trained weights"
	print("Loading weights from ", model_path)
	model.load_weights(model_path, by_name=True)

	# Get filenames of test dataset DICOM images
	test_image_fps = get_dicom_fps(test_dicom_dir)
	# Make predictions on test images, write out sample submission 
	def predict(image_fps, filepath='submission.csv', min_conf=0.95): 
	    
	    # assume square image
	    resize_factor = ORIG_SIZE / config.IMAGE_SHAPE[0]
	    #resize_factor = ORIG_SIZE 
	    with open(filepath, 'w') as file:
	      for image_id in tqdm(image_fps): 
	        ds = pydicom.read_file(image_id)
	        image = ds.pixel_array
	        # If grayscale. Convert to RGB for consistency.
	        if len(image.shape) != 3 or image.shape[2] != 3:
	            image = np.stack((image,) * 3, -1) 
	        image, window, scale, padding, crop = utils.resize_image(
	            image,
	            min_dim=config.IMAGE_MIN_DIM,
	            min_scale=config.IMAGE_MIN_SCALE,
	            max_dim=config.IMAGE_MAX_DIM,
	            mode=config.IMAGE_RESIZE_MODE)
	            
	        patient_id = os.path.splitext(os.path.basename(image_id))[0]

	        results = model.detect([image])
	        r = results[0]

	        out_str = ""
	        out_str += patient_id 
	        out_str += ","
	        assert( len(r['rois']) == len(r['class_ids']) == len(r['scores']) )
	        if len(r['rois']) == 0: 
	            pass
	        else: 
	            num_instances = len(r['rois'])
	  
	            for i in range(num_instances): 
	                if r['scores'][i] > min_conf: 
	                    out_str += ' '
	                    out_str += str(round(r['scores'][i], 2))
	                    out_str += ' '

	                    # x1, y1, width, height 
	                    x1 = r['rois'][i][1]
	                    y1 = r['rois'][i][0]
	                    width = r['rois'][i][3] - x1 
	                    height = r['rois'][i][2] - y1 
	                    bboxes_str = "{} {} {} {}".format(x1*resize_factor, y1*resize_factor, \
	                                                       width*resize_factor, height*resize_factor)   
	#                     bboxes_str = "{} {} {} {}".format(x1, y1, \
	#                                                       width, height)
	                    out_str += bboxes_str

	        file.write(out_str+"\n")

	# predict only the first 50 entries
	submission_fp = os.path.join(ROOT_DIR, 'submission.csv')
	print(submission_fp)
	predict(test_image_fps, filepath=submission_fp)

	output = pd.read_csv(submission_fp, names=['patientId', 'PredictionString'])
	print output.head(100)
	'''
	Note that the Mask-RCNN detector configuration parameters have been selected 
	to reduce training time for demonstration purposes, they are not optimal.

	imagenes mas grandes, solo un canal.. quitar lo de MEAN_PIXEL
	'''
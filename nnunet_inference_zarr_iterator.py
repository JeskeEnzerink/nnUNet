#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 05:57:13 2026

@author: wirrbel
"""

import zarr 
import itertools
import os
import tempfile
import json
import datetime
import torch
import numpy as np
from dask import delayed, compute
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from tqdm import tqdm


def generate_slice_blocks(shape, block_size=(200, 200, 200), overlap=0.5):
    # generates an iterator that yields tuples of slice objects
    bz, by, bx = block_size
    
    # Create ranges for each dimension
    ranges = [range(0, s, int(b*overlap)) for s, b in zip(shape, block_size)]

    for z, y, x in itertools.product(*ranges):
        # We create a tuple of slice objects
        # slice(start, stop) is equivalent to the [start:stop] syntax
        yield (
            slice(z, z + bz), 
            slice(y, y + by), 
            slice(x, x + bx)
        )
        
def create_zarr_group(output_location,input_data,output_dtype):
    group = zarr.open_group(output_location, mode='w-', zarr_format=3)
    group.create_array(
        name='0',
        shape=input_data.shape,
        chunks=input_data.chunks,
        shards=input_data.shards,
        dtype=output_dtype,
        dimension_names=('z', 'y', 'x'),
        compressors=[zarr.codecs.BloscCodec(cname='lz4', clevel=5)]
    )

def _preprocess_and_infer(predictor, input_zarr_location, block, output_inference_values_location, output_count_map_location, lower_bound, upper_bound):
    begin = datetime.datetime.now()
    #load zarr 
    input_data = zarr.open_group(input_zarr_location, mode='r')
    input_data = input_data["0"]
    
    output_inference_values = zarr.open_group(output_inference_values_location, mode='a')
    output_inference_values = output_inference_values["0"]

    output_count_map = zarr.open_group(output_count_map_location, mode='a')
    output_count_map = output_count_map["0"]
    
    #load chunk 
    image = input_data[block]
    
    #if chunk is empty, skip
    if image.max() == 0:
        #print(f"skipping empty block {block}")
        output_inference_values[block] += -1000.0
        output_count_map[block] += 1
        pass
 
    else:
        #print(f"infering block {block}")
        #preprocess chunk by minimum & maximum 95% percentile
        image = image.astype(np.float32)
     
        #normalize image from 0 - 1
        np.clip(image, lower_bound, upper_bound, out=image)
        image = (image - lower_bound) / upper_bound
        
        #add channel info (only for batch size = 1)
        image = np.expand_dims(image, axis=0)
        #image = np.expand_dims(image, axis=0)

        
        '''
        #convert data to torch format and load 
        window_data = torch.from_numpy(image)
        window_data = window_data.cuda()
        window_data = window_data.to(torch.float32)
        sw_device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        window_data.to(sw_device)
        
        #actual inference
        seg_prob = model(window_data)
        seg_prob = seg_prob.to(torch.float16).to(device).numpy()
        '''
        
        #debug, use nnunet's predictor
        props = {'spacing': [1.0, 1.0, 1.0]} # Replace with actual voxel spacing!

        seg_prob = predictor.predict_single_npy_array(image,props)
        
        #write back out 
        output_inference_values[block] += seg_prob
        output_count_map[block] += 1
               
        #duration = datetime.datetime.now() - begin
        #print(f"finished infering block {block}, took {duration}")
        return        

def load_json(json_location):
    with open(json_location,encoding='utf-8') as stream:
         json_content = json.load(stream)
    return json_content

#define locations
input_zarr_location = "/home/wirrbel/2026-11-02_jeske_stz_test_P0005/I10_P0005_C00_WFA-488.ome.zarr/"


#clean up trailing slash
input_zarr_location = os.path.abspath(input_zarr_location)
#
output_zarr_location, dataset_name = os.path.split(input_zarr_location)

#load zarr 
input_data = zarr.open_group(input_zarr_location, mode='r')
input_data = input_data["0"]

#read lower & upper bound from dataset json
bounds = load_json(input_zarr_location + ".intensity.json")
lower_bound = bounds["local_min"]
upper_bound = bounds["local_95"]

#create temp storage zarrs
output_inference_values_location = tempfile.mkdtemp(prefix=dataset_name + '.inference_values-', dir=output_zarr_location)
output_count_map_location = tempfile.mkdtemp(prefix=dataset_name + '.count_map-', dir=output_zarr_location)

create_zarr_group(output_inference_values_location, input_data, output_dtype=np.float16)
create_zarr_group(output_count_map_location, input_data, output_dtype=np.uint8)

'''
##== from mouseMapper's sliding window inferer -> predict_from_dask ==##

#load on GPU if available   
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

#use nnUnetv2's PlansManager to load config
plans_file = load_json(os.path.join(results_dir, 'plans.json'))
plans_manager = PlansManager(plans_file)
configuration_manager = plans_manager.get_configuration('3d_fullres')


# Get actual input channels from nnUNet plans
num_input_channels_model = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
label_manager = plans_manager.get_label_manager(dataset_json)
num_output_classes = label_manager.num_segmentation_heads # Get number of output classes
print(f"nnUNet model expects {num_input_channels_model} input channels and produces {num_output_classes} output classes.")



#initialize nnUNet model 

model = get_network_from_plans(
    configuration_manager.network_arch_class_name,
    configuration_manager.network_arch_init_kwargs,
    configuration_manager.network_arch_init_kwargs_req_import,
    num_input_channels_model, # Pass the correct number of input channels for the model
    2, # Pass the correct number of output classes
    allow_init=True,
    deep_supervision=False)

checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
torch.cuda.empty_cache()

model = model.to(device)
model.load_state_dict(checkpoint["network_weights"])
model.eval()
'''
'''
#assemble arguments for dask-parallelized inference
args = []
for block in blocklist:
    args.append((model, input_zarr_location, block, output_inference_values_location, output_count_map_location, lower_bound, upper_bound))

#infer per block, parallelized with dask
#downsample_tasks = [delayed(_preprocess_and_infer)(*arg) for arg in args]
#compute(*downsample_tasks)
'''

## == Actual inference == ## 

#create list of blocks 
blocklist = []
for block in generate_slice_blocks(input_data.shape,block_size=(200,200,200),overlap=0.5):
    blocklist.append(block)


#predictor from NNunet's example prediction script
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device('cuda', 0),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=False
)

#find nnunet's checkpoint
results_dir = '/home/wirrbel/MN_8/2026-06-01_nnunet_test/training_data/wfa/nnUNet_results/Dataset020_wfa_moritz_ilastik/nnUNetTrainer_50epochs__nnUNetResEncUNetLPlans__3d_fullres/'
dataset_json = load_json(os.path.join(results_dir, 'dataset.json'))
model_path = results_dir+'/fold_0/checkpoint_best.pth'

# Initialize with your specific model path
predictor.initialize_from_trained_model_folder(
    results_dir,
    use_folds=(0,),
    checkpoint_name='checkpoint_best.pth',
)
  
### Inference
for block in tqdm(blocklist, desc="inferring blocks"):
    _preprocess_and_infer(predictor, input_zarr_location, block, output_inference_values_location, output_count_map_location, lower_bound, upper_bound)

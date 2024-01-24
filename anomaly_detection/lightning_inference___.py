"""Inference Entrypoint script."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#from argparse import ArgumentParser, Namespace
#from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from anomalib.config import get_configurable_parameters
from anomalib.data.inference import InferenceDataset
from anomalib.data.utils import InputNormalizationMethod, get_transforms
from anomalib.models import get_model
from anomalib.utils.callbacks import get_callbacks

import os
import time

model_name = 'padim' #'dfm'

def infer(infer_img_path):
    config_path = 'C:/Users/user/Desktop/anomaly_detection/results/{}/raphas_v2/run/config.yaml'.format(model_name)
    model_path = 'C:/Users/user/Desktop/anomaly_detection/results/{}/raphas_v2/run/weights/lightning/model-v4.ckpt'.format(model_name)
       
    infer_resutls_path = "C:/Users/user/Desktop/anomaly_detection/infer_results_defect_{}".format(model_name)
    os.makedirs(infer_resutls_path, exist_ok=True)       

    config = get_configurable_parameters(config_path=config_path)
    config.trainer.resume_from_checkpoint = str(model_path)
    if infer_resutls_path:  # overwrite save path
        config.visualization.save_images = True
        config.visualization.image_save_path = infer_resutls_path
    else:
        config.visualization.save_images = False

    # create model and trainer
    model = get_model(config)
    callbacks = get_callbacks(config)
    trainer = Trainer(callbacks=callbacks, **config.trainer)

    # get the transforms
    transform_config = config.dataset.transform_config.eval if "transform_config" in config.dataset.keys() else None
    image_size = (config.dataset.image_size[0], config.dataset.image_size[1])
    center_crop = config.dataset.get("center_crop")
    if center_crop is not None:
        center_crop = tuple(center_crop)
    normalization = InputNormalizationMethod(config.dataset.normalization)
    transform = get_transforms(
        config=transform_config, image_size=image_size, center_crop=center_crop, normalization=normalization
    )
   
    # create the dataset
    dataset = InferenceDataset(
        infer_img_path, image_size=tuple(config.dataset.image_size), transform=transform  # type: ignore
    )
    dataloader = DataLoader(dataset)

    # generate predictions
    trainer.predict(model=model, dataloaders=[dataloader])


if __name__ == "__main__":
    
    folder_path = 'C:/Users/user/Desktop/anomaly_detection/test_img/' 
    processed_files = set()

    while True:
        img_files = os.listdir(folder_path)
        for file in img_files:
            if file not in processed_files:
                print(f"find new image !!!! : {file}")
                
                img_file = folder_path + file
                infer(img_file)
                
                processed_files.add(file)
        #time.sleep(1)

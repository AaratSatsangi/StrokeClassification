
from Classifiers import TransNets, ConvNets
from Utils import *
import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2


MODELS = [
    # (name, type, fold)
    ("ResNet_S", "conv", 9),
    ("SWIN_S", "trans", 10),
    ("CvT_S", "trans", 6)
]
MODEL_FOLDER_DICT = {
    # type: folder-name
    "conv": "Convolutional Networks",
    "trans" : "Transformer Networks",
    "hybrid": "Hybrid Networks"
}

PATH_FOLDER_INFERENCE = "Inference/"
USING_CAM = "Score-CAM"


def save_imgs(outputs_dict:dict):
    """
    Parameters:
        outputs_dict = {
            'model_name': {
                'imgs': [original_img, annotated, output],
                'class': predicted_class_name,
                'prob': classification_probability,
                'mask': generated_mask
            }
        } 
    """
    col_headers = ["Original", "Annotated", USING_CAM]
    row_headers = [model_name for model_name in outputs_dict.keys()] # Model Names
    row_texts = [f"{data['class']}\nP={data['prob']: 0.3f}" for model_name, data in outputs_dict.items()]

    rows = len(row_headers)
    cols = len(col_headers)
    
    # Some Sanity Checks
    for model_name, data in outputs_dict.items():
        if len(data['imgs']) != cols:
            raise ValueError(f"Number of output images for {model_name} is {len(data['imgs'])} which is less than {cols}")


    fig = plt.figure(figsize=(10, 10))
    spec = gridspec.GridSpec(rows + 1, 4, figure=fig, wspace=0.5, hspace=0.5)

    # Setting Column Names
    for col_num in range(cols):
        ax = fig.add_subplot(spec[0, col_num + 1])
        ax.text(0.5, 0.5, col_headers[col_num], ha='center', va='center', fontsize=12)
        ax.axis('off')

    # Setting Row Names
    for row_num in range(rows):
        ax = fig.add_subplot(spec[row_num + 1, 0])
        ax.text(0.5, 0.5, row_headers[row_num], ha='center', va='center', fontsize=12)
        ax.axis('off')
        
        # Add row-specific text on the right
        ax = fig.add_subplot(spec[row_num + 1, 4])
        ax.text(0, 0.5, row_texts[row_num], ha='left', va='center', fontsize=10, wrap=True)
        ax.axis('off')

    # Adding Images to Plot
    masks = []
    for row_num, (model_name, data) in enumerate(outputs_dict.items()):
        imgs = data['imgs']
        masks.append((model_name, data['mask']))
        for col_num in range(cols):
            ax = fig.add_subplot(spec[row_num + 1, col_num + 1])
            ax.imshow(imgs)
            ax.axis('off')

    # Save the plot to the specified file
    if not os.path.exists(PATH_FOLDER_INFERENCE):
        os.makedirs(PATH_FOLDER_INFERENCE)
    infer_num = len(os.listdir(PATH_FOLDER_INFERENCE)) + 1
    path = PATH_FOLDER_INFERENCE + "Image " + str(infer_num) + "/"
    plt.savefig(path + "final_output.png", bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory
    
    # Save Masks
    for model_name, mask in masks:
        cv2.imwrite(path + model_name + "_mask.png", mask.astype(np.uint8))

    return

def get_target_layer(model_name, model: nn.Module):
    model_target_layers = {
        "ResNet": ["layer4"],
        "SWIN": ["norm1"],
        "CvT": ["norm1"]
    }
    target_layer = []

    if "ResNet" in model_name:
        for module_name, module in model.named_children():
            if model_target_layers["ResNet"] in module_name:
                target_layer.append
    
    elif "SWIN" in model_name:
        for module_name, module in model.named_children():
            if model_target_layers["SWIN"] in module_name:
                target_layer.append(sub_module)
            
            for sub_module_name, sub_module in module.named_children():
                if model_target_layers["SWIN"] in sub_module_name:
                    target_layer.append(sub_module)




if __name__ == "__main__":
    pass
    # import picture to run inference on
    # load each model one by one
    #   extract mask, save mask in list, save overlay in list2
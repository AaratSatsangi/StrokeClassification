
from Classifiers import TransNets, ConvNets
from torchvision import transforms
from Preprocessor import CTPreprocessor
from Utils import *
import torch
from torch import nn
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import json
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.spatial.distance import jensenshannon
# from Utils.Helpers import _binarizeUsingMax
from PIL import Image


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

PATH_FOLDER_INFERENCE = "Inference Images/"
USING_CAM = "Score-CAM"
TARGET_MAP = {
    0: "Hemorrhagic",
    1: "Ischemic",
    2: "Normal"
}
MODEL_TYPE_DICT: dict = {"conv": "Convolutional Networks", "trans" : "Transformer Networks"}
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (224,224)


def save_imgs(outputs_dict:dict, save_folder:str):
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
    col_headers = ["Original", "Annotated", USING_CAM] if len(outputs_dict[list(outputs_dict.keys())[0]]['imgs']) == 3 else ["Original", USING_CAM]
    row_headers = [model_name for model_name in outputs_dict.keys()] # Model Names
    row_texts = [f"{TARGET_MAP[data['class']]}\nP={data['prob']: 0.3f}" for model_name, data in outputs_dict.items()]

    rows = len(row_headers)
    cols = len(col_headers)
    
    # Some Sanity Checks
    for model_name, data in outputs_dict.items():
        if len(data['imgs']) != cols:
            raise ValueError(f"Number of output images for {model_name} is {len(data['imgs'])} which is less than {cols}")


    fig = plt.figure(figsize=(8.5,7) if cols == 3 else (5.6,7))
    spec = gridspec.GridSpec(
        nrows=rows + 1,
        ncols=cols + 2, 
        figure=fig, 
        wspace=0.05, 
        hspace=0.05, 
        width_ratios=[0.1, 0.26, 0.26, 0.26, 0.1] if cols == 3 else [0.1, 0.4, 0.4, 0.1], 
        height_ratios=[0.04, 0.32, 0.32, 0.32]
    )

    # Setting Column Names
    for col_num in range(cols):
        ax = fig.add_subplot(spec[0, col_num + 1])
        ax.text(0.5, 0, col_headers[col_num], ha='center', va='bottom', fontsize=12)
        ax.axis('off')

    # Setting Row Names
    for row_num in range(rows):
        ax = fig.add_subplot(spec[row_num + 1, 0])
        ax.text(1, 0.5, row_headers[row_num], ha='right', va='center', fontsize=12, wrap=True)
        ax.axis('off')
        
        # Add row-specific text on the right
        ax = fig.add_subplot(spec[row_num + 1, cols  + 2- 1])
        ax.text(0, 0.5, row_texts[row_num], ha='left', va='center', fontsize=12, wrap=True)
        ax.axis('off')

    # Adding Images to Plot
    masks = []
    for row_num, (model_name, data) in enumerate(outputs_dict.items()):
        imgs = data['imgs']
        masks.append((model_name, data['mask']))
        for col_num in range(cols):
            ax = fig.add_subplot(spec[row_num + 1, col_num + 1])
            ax.imshow(imgs[col_num])
            ax.axis('off')

    # Save the plot to the specified file
    plt.savefig(save_folder + "final_output.png", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # Close the figure to free up memory
    
    mask_save_folder = save_folder + "masks/"
    if not os.path.exists(mask_save_folder):
        os.mkdir(save_folder + "masks/")
    # Save Masks
    for model_name, mask in masks:
        cv2.imwrite(save_folder + "masks/" + model_name + ".png", mask.astype(np.uint8))

    return

def get_target_layer(model_name, model: nn.Module):
    model_target_layers = {
        "ResNet": "layer4",
        "SWIN": "norm1",
        "CvT": "norm1"
    }
    target_layer = []

    if "ResNet" in model_name:
        for module_name, module in model.named_children():
            # print(module_name)
            if model_target_layers["ResNet"] in module_name:
                target_layer.append(module)
    
    elif "SWIN" in model_name:
        for module_name, module in model.named_children():
            if module_name == "features":
                for sub_module_name, sub_module in module[-1][1].named_children():
                    # print("==>" + sub_module_name)
                    if model_target_layers["SWIN"] in sub_module_name:
                        target_layer.append(sub_module)
    
    elif "CvT" in model_name:
        for module_name, module in model.named_children():
            if module_name == "blocks":
                for sub_module_name, sub_module in module[-1].named_children():
                    # print("==>" + sub_module_name)
                    if model_target_layers["CvT"] in sub_module_name:
                        target_layer.append(sub_module)

    assert len(target_layer) > 0, "Error: Target Layers not assigned"
    return target_layer

def _reshape_transform(tensor):
    # Bring the channels to the first dimension,
    # like in CNNs.
    if "CvT" in MODEL_NAME:
        result = tensor[:, 1 :  , :] if tensor.size(1) == 197 else tensor
        patch_size = int(np.sqrt(result.size(1) + 1))
        # print(f"Tensor: {result.size()}, Patch: {patch_size}")
        result = result.reshape(tensor.size(0), patch_size, patch_size, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
    elif "SWIN" in MODEL_NAME:
        result = tensor.transpose(2, 3).transpose(1, 2)
    elif "ResNet" in MODEL_NAME:
        return tensor
    else:
        print("ERROR")
        exit()
    # print(result.shape)
    # exit()
    return result

def infer(model:nn.Module, img:torch.Tensor):
    y_pred = model(img).detach()
    outputs = torch.nn.functional.softmax(y_pred, dim=1).cpu().squeeze()
    pred_class = np.argmax(outputs.numpy())
    pred_prob = outputs[pred_class]
    print(f"|\t|--- Predicted: {TARGET_MAP[pred_class]}")
    print(f"|\t|--- Probability: {pred_prob: 0.4f}")
    print("|")
    return pred_class, pred_prob

def get_score_cam(model_name:str, model:nn.Module, image:dict, pred_class):
    """
    Input:
        image = {
            'input_tensor': transformed for model input,
            'norm_img': required for overlay,
            'original_img': original for saving,
            'annotated_img': original overlay mask
        }
    Output:
        outputs_dict = {
            'imgs': [original_img, annotated, output],
            'class': predicted_class_name,
            'prob': classification_probability,
            'mask': generated_mask
        } 
    """
    target_layers = get_target_layer(model_name=model_name, model=model)
    targets = [ClassifierOutputSoftmaxTarget(pred_class)]

    outputs_dict = {}
    input_tensor = image['input_tensor']
    with ScoreCAM(model=model, target_layers=target_layers, reshape_transform=_reshape_transform) as cam:
        cam_img = cam(input_tensor=input_tensor, targets=targets)
        overlay_img = show_cam_on_image(image["norm_img"], cam_img[0, :], use_rgb=True, image_weight=0.5)
        
        outputs_dict['imgs'] = [image['original_img'], image['annotated_img'], overlay_img] if pred_class != 2 else [image['original_img'], overlay_img]
        outputs_dict['class'] = pred_class
        outputs_dict['prob'] = torch.nn.functional.softmax(cam.outputs, dim=1).squeeze()[pred_class]
        outputs_dict['mask'] = cam_img[0, :] * 255
    return outputs_dict

def load_model(model_config:tuple):
    model_name = model_config[0]
    model_type = model_config[1]
    fold = model_config[2]
    path = f"Classifiers/{MODEL_TYPE_DICT[model_type]}/{model_name}/F{fold}_Checkpoint.pth"
    if "SWIN" in model_name:
        model = TransNets.SWIN(model_size="s")
    elif "CvT" in model_name:
        model = TransNets.CvT(model_size="s")
    elif "MaxViT" in model_name:
        model = TransNets.MaxViT(model_size="s")
    elif "ResNet" in model_name:
        model = ConvNets.ResNet(model_size="s")
    else:
        print(f"Error: {model_name} not recognized!")
        exit(1)

    checkpoint = torch.load(f=path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.to(DEVICE)
    return model

def load_images(path = PATH_FOLDER_INFERENCE, img_size = IMG_SIZE):
    img_transforms = CTPreprocessor(
        img_size=img_size,
        transformations=[
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],std=[0.229],inplace=True),
        ],
        use_mask=False
    )
    resize = transforms.Resize(img_size) 
    images = []
    for image_folder in os.listdir(path):
        img_path = f"{path}{image_folder}/img.png"
        annot_path = f"{path}{image_folder}/overlay.png"
        img = Image.open(img_path).convert("RGB")
        annot_img = Image.open(annot_path).convert("RGB") if os.path.exists(annot_path) else Image.new(mode="RGB", size=img_size)
        # with open(f"{path}{image_folder}/class.txt") as f:
        #     class_ =f.readline()
        #     assert class_.isdigit(), "The class must contain only numeric characters"
        #     class_ = int(class_)
        #     assert class_ == 0 or class_ == 1 or class_ == 2, "Invalid Class"
        with open(f"{path}{image_folder}/result.json", 'r') as f:
            class_ = json.load(f)['class']

        images.append(
            {
                "input_tensor": img_transforms(img).unsqueeze(0).to(DEVICE),
                "norm_img": np.float32(resize(img)) / 255,
                "original_img": np.array(resize(img)),
                "annotated_img": np.uint8(resize(annot_img)),
                "class": class_,
                "path": f"{path}{image_folder}/"
            }
        )

    return images

def get_pred_class(input_tensor, model_configs):
    pred_classes = []
    for config in model_configs:
        model = load_model(config)
        print(f"|\tModel: {config[0]}")
        pred_class, pred_prob = infer(model, input_tensor)
        pred_classes.append(pred_class)
        del model
    pred_classes = np.array(pred_classes)
    unique_elements, counts = np.unique(pred_classes, return_counts=True)
    max_count = np.max(counts)
    most_frequent_elements = unique_elements[counts == max_count]
    assert len(most_frequent_elements) == 1, "Error: There is more than one most frequent element." 
    print(f"|---> Final Prediction: {TARGET_MAP[most_frequent_elements[0]]}")
    print(f"|", end="")
    return most_frequent_elements[0]

def get_JSC(masks:list):
    def normalize_img(img):
        total = np.sum(img)
        return img/total
    
    normalized_imgs = [normalize_img(img) for img in masks]
    avg_dist = np.mean(normalized_imgs, axis=0)
    js_divergence = sum(
        1/len(normalized_imgs) * jensenshannon(img.flatten(), avg_dist.flatten(), base=2)**2 for img in normalized_imgs
    )
    return (1-js_divergence)*100

if __name__ == "__main__":
    MODEL_CONFIGS = [
        ("ResNet_S", "conv", 9),
        ("SWIN_S", "trans", 10),
        ("CvT_S", "trans", 6)
    ]
    MODEL_NAME = ""
    images = load_images()
    
    for i in range(len(images)):
        print(f"="*40)
        print(f"Image {i+1}: {TARGET_MAP[images[i]['class']]}")
        print(f"-"*40)
        image = images[i]
        pred_class = get_pred_class(image['input_tensor'], model_configs=MODEL_CONFIGS)
        
        outputs = {}
        for config in MODEL_CONFIGS:
            model = load_model(config)
            _, model = next(model.named_children())
            MODEL_NAME = model_name = config[0]
            outputs[model_name] = get_score_cam(model_name=config[0], model=model, image=image, pred_class=pred_class)
            del model

        score = get_JSC([output_dict['mask'] for _, output_dict in outputs.items()])
        print(f"|---> JSC: {score: 0.3f}%")
        save_imgs(outputs_dict=outputs, save_folder=image['path'])
        print(f"="*40)
        with open(f"{image['path']}/result.json", 'r+') as file:
            data = json.load(file)
            data['pred'] = int(pred_class)
            data['JSC'] = round(score, 3)
            file.seek(0)
            json.dump(data, file, indent=4)
        
    # import picture to run inference on
    # load each model one by one
    #   extract mask, save mask in list, save overlay in list2
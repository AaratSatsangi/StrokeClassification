import torch
import numpy as np
import os
import sys
import cv2
import random
from PIL import Image
from Preprocessor import CTPreprocessor
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ========= DO NOT TOUCH CONSTANTS ===========
random.seed(26)
IMG_SIZE = (1, 224, 224)
MODEL_TYPE_DICT: dict = {"conv": "Convolutional Networks", "trans" : "Transformer Networks"}
PATH_MODEL_FOLDER: str = ""
PATH_SAVE_IMG:str = ""
PATH_TEST_IMGS: dict = {} 
# {
#     "Hemorrhagic": ['Data/Compiled/PNG/Hemorrhagic/15826.png', 'Data/Compiled/PNG/Hemorrhagic/11015.png', 'Data/Compiled/PNG/Hemorrhagic/15597.png', 'Data/Compiled/PNG/Hemorrhagic/11183.png', 'Data/Compiled/PNG/Hemorrhagic/15015.png'],
#     "Ischemic": ['Data/Compiled/PNG/Ischemic/11626.png', 'Data/Compiled/PNG/Ischemic/15921.png', 'Data/Compiled/PNG/Ischemic/10164.png', 'Data/Compiled/PNG/Ischemic/12608.png', 'Data/Compiled/PNG/Ischemic/16241.png'],
#     "Normal": ['Data/Compiled/PNG/Normal/13831.png', 'Data/Compiled/PNG/Normal/12456.png', 'Data/Compiled/PNG/Normal/10681.png', 'Data/Compiled/PNG/Normal/13170.png', 'Data/Compiled/PNG/Normal/12024.png']
# }
TEST_IMGS: dict = {}
OVERLAY_IMGS: dict = {}
IMG_TRANSFORMS_TEST = CTPreprocessor(
    img_size=IMG_SIZE[1:],
    transformations=[
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],std=[0.229],inplace=True),
    ]
)
UN_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[-0.485/0.229], std=[1/0.229], inplace=True)
])
TARGET_MAP = {
    "Hemorrhagic": 0,
    "Ischemic": 1,
    "Normal": 2
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================================
model:torch.nn.Module = None

def setup(model_name, model_type, fine_tuned:bool):
    global model, PATH_MODEL_FOLDER, PATH_SAVE_IMG, TEST_IMGS, PATH_TEST_IMGS
    
    # Load Model
    PATH_MODEL_FOLDER = "Classifiers/" + MODEL_TYPE_DICT[model_type] + "/" + model_name + "/"
    model_name += "_FT.pt" if fine_tuned else ".pt"
    path = PATH_MODEL_FOLDER + model_name
    assert os.path.exists(path) , f"Model does not exist: {path}"
    try:
        print(f"Loading Model: {model_name}")
        _, model = next(torch.load(path, map_location="cuda:0").named_children())
        model.to(DEVICE).eval()
    except Exception as e:
        print(f"Exception occured while loading model: {e}")
        sys.exit(1)

    # Check Directory exists to save heatmaps
    PATH_SAVE_IMG = PATH_MODEL_FOLDER + "Heatmaps/"
    if not os.path.exists(PATH_SAVE_IMG):
        os.mkdir(PATH_SAVE_IMG)

    # get random image paths
    main_path = "Data/Compiled/PNG/"
    main_overlay_path = "Data/Compiled/Overlay/"
    for class_name in os.listdir(main_path):
        imgs_main_path = main_path + class_name + "/"
        imgs_overlay_path = main_overlay_path + class_name + "/"
        imgs_paths = []
        overlay_paths = []
        for img_path in random.sample(os.listdir(imgs_main_path), 10):
            imgs_paths.append(os.path.join(imgs_main_path, img_path))
            if class_name != "Normal": 
                overlay_paths.append(os.path.join(imgs_overlay_path, img_path))
        PATH_TEST_IMGS[class_name] = (imgs_paths, overlay_paths)

    # Load Images
    resize = transforms.Resize(IMG_SIZE[1:])
    for class_name, img_paths in PATH_TEST_IMGS.items():
        TEST_IMGS[class_name] = []
        OVERLAY_IMGS[class_name] = []
        for i in range(len(img_paths[0])): 
            img_path = img_paths[0][i]
            img = Image.open(img_path).convert("RGB")
            TEST_IMGS[class_name].append((IMG_TRANSFORMS_TEST(img), np.float32(resize(img))/255, np.array(resize(img))))
            
            if class_name != "Normal":
                overlay_path = img_paths[1][i]
                overlay = Image.open(overlay_path).convert("RGB")
                OVERLAY_IMGS[class_name].append(np.uint8(resize(overlay)))


def _get_target_layer(model_name):
    target_layer = []
    if "ResNet" in model_name: 
        for name, module in model.named_children():
            # print(name)
            if(name == "layer2" or name == "layer3" or name == "layer4" ):
                target_layer.append(module[-1])
            elif (name == "avgpool"):
                target_layer.append(module)
    
    
    elif "SWIN" in model_name:
        for name, module in model.named_children():
            print(f"Layer Name: {name}")
            if(name == "features"):
                for module_part in module[-1:]:
                    try:
                        for n, sub in module_part[0].named_children():
                            print(f"\tLast Module: LayerName ==> {n}")
                            if n == "norm1" or n =="norm1":
                                print(f"\t\t Appended: ==> {sub}")
                                target_layer.append(sub)
                    except:
                        print()
                        
                # target_layer.append(module)
    
    
    elif "CvT" in model_name:
        for name, module in model.named_children():
            print(f"Layer Name: {name}")
            if(name == "blocks"):
                for module_part in module[-1:]:
                    try:
                        for n, m in module_part.named_children():
                            print(f"\tLast Module: LayerName ==> {n}")
                            if n == "norm1":
                                # pass
                                print(f"\t\t Appended: ==> {m}")
                                target_layer.append(m)
                    except:
                        print()

            # elif(name == "norm"):
            #     target_layer.append(module)
                
    print(f"===> Using {len(target_layer)} Layers <===")
    return target_layer

def reshape_transform(tensor):
    # result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
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
    else:
        print("ERROR")
        exit()
    # print(result.shape)
    # exit()
    return result


# attentionMaps = []

# def attentionHook(m ,i ,o):
#     attention_map = o.squeeze(0)  # Remove batch dimension
#     attention_map = attention_map.reshape(32, attention_map.shape[0], attention_map.shape[1], attention_map.shape[2] / 32)
#     attention_map = attention_map / attention_map.sum(axis=(0, 1), keepdims=True)
#     attentionMaps.append(o)

# def getAttentionMap(model_name):
#     target_layer = []
#     if "SWIN" in model_name:
#         for name, module in model.named_children():
#             print(f"Layer Name: {name}")
#             if(name == "features"):
#                 for n, sub in module[-1][0].named_children():
#                     print(f"\tLast Module: LayerName ==> {n}")
#                     if n == "attn":
#                         sub.register_forward_hook(attentionHook)
#     for class_name, imgs in TEST_IMGS.items():
        
#         overlay_imgs = OVERLAY_IMGS[class_name]
#         i = 0
#         for img in imgs:
#             output = model(img[0].unsqueeze(0).to(DEVICE))
#             # Normalize attention weights (optional, but recommended):
#             attention_map = attention_map / attention_map.sum(axis=(0, 1), keepdims=True)
#             exit()
#             final_img = cv2.hconcat([overlay_imgs[i], cam_image]) if class_name != "Normal" else cam_image
#             cv2.imwrite(PATH_SAVE_IMG + class_name[0] + "_cam_" + str(i) + f"_P[{output_prob: 0.4f}]" + ".jpg", final_img)
#     exit()

MODEL_NAME = "SWIN_S"
MODEL_TYPE = "trans"
    
from torchinfo import summary    

if __name__ == "__main__":
    setup(MODEL_NAME, MODEL_TYPE, True)
    summary(model, input_size=(1,1,224,224), depth = 8, col_names=["input_size", "output_size"])
    # exit()
    # getAttentionMap(model_name=MODEL_NAME)
    target_layer = _get_target_layer(MODEL_NAME)
    
    for class_name, imgs in TEST_IMGS.items():
        targets = [ClassifierOutputSoftmaxTarget(TARGET_MAP[class_name])]
        # print(targets)
        overlay_imgs = OVERLAY_IMGS[class_name]
        i = 0
        for img in imgs:
            input_tensor = img[0].unsqueeze(0)

            with GradCAMPlusPlus(model=model, target_layers=target_layer, reshape_transform=reshape_transform) as cam:
                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                # In this example grayscale_cam has only one image in the batch:
                grayscale_cam = grayscale_cam[0, :]
                # print(grayscale_cam.shape)
                cam_image = show_cam_on_image(img[1], grayscale_cam, use_rgb=True, image_weight=0.5)
                # You can also get the model outputs without having to redo inference
                model_outputs = cam.outputs
                output_prob = torch.nn.functional.softmax(model_outputs, dim=1)[0][TARGET_MAP[class_name]]
                # Save grayscale cam
                # 
                #
            gb_model = GuidedBackpropReLUModel(model=model,device=DEVICE)
            gb = gb_model(input_tensor, target_category=None)
            gb = cv2.merge([gb, gb, gb])
            print(f"GB SHAPE: {gb.shape}")

            cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            cam_gb = UN_NORMALIZE(cam_mask * gb)
            gb = UN_NORMALIZE(gb)
            
            final_img = cv2.hconcat([img[2], overlay_imgs[i], cam_image, np.array(gb), np.array(cam_gb)]) if class_name != "Normal" else cam_image
            save_dir = PATH_SAVE_IMG + class_name + "/"
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            cv2.imwrite(save_dir + "cam" + str(i) + f"_P[{output_prob: 0.4f}]" + ".jpg", final_img)
            i+=1

    
    
    
    
    
    
    
    
    
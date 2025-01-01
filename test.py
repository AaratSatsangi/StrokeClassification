# Take image, calculate three cams, 
# store (main, overlay, cam+main) and cam for each model
# Take cams and calculate jsc

from Utils.Helpers import *
from Classifiers import TransNets, ConvNets
from torchvision import transforms
from Preprocessor import CTPreprocessor
from PIL import Image
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import torch

MODEL_NAME:str = ""

def load_model(model_name, path):
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

    checkpoint = torch.load(path, "cuda:0")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def load_images(path="Test Images/", img_size = (224,224)):
    img_transforms = CTPreprocessor(
        img_size=img_size,
        transformations=[
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],std=[0.229],inplace=True),
        ],
        use_mask=False
    )
    main_img_path = path + "PNG/"
    main_overlay_path = path + "Overlay/"

    imgs = {}
    resize = transforms.Resize(img_size)
    for class_name in os.listdir(main_img_path):
        imgs[class_name] = []
        for img_name in os.listdir(main_img_path + class_name):
            
            img_path = main_img_path + class_name + f"/{img_name}"
            overlay_path = main_overlay_path + class_name + f"/{img_name}"
            
            img = Image.open(img_path).convert("RGB")
            overlay_img = Image.open(overlay_path).convert("RGB")
            imgs[class_name].append(
                {
                    "input_tensor": img_transforms(img), # Transformed Image for model
                    "norm_img": np.float32(resize(img)) / 255, # required for overlay image
                    "main_img": np.array(resize(img)), # the main image
                    "overlay_img": np.uint8(resize(overlay_img)) # overlay image
                }
            )
    
    return imgs

def save_images(img_dict:dict, class_name:str, name:str, model_name:str, path="Test Images/Output/"):
    main_img = img_dict["main_img"].astype(np.uint8)
    overlay_img = img_dict["overlay_img"].astype(np.uint8)
    main_and_cam_img = img_dict["main_and_cam_img"].astype(np.uint8)
    cam_img = img_dict["cam_img"].astype(np.uint8)
    
    spacer_width = 10
    _ = np.ones((main_img.shape[0], spacer_width, main_img.shape[2]), dtype=main_img.dtype)*255

    final_img = cv2.hconcat([_, main_img, _, overlay_img, _, main_and_cam_img, _])
    path_folder = path + model_name + "/"
    if not os.path.exists(path_folder):
        os.mkdir(path_folder)
    cv2.imwrite(path_folder + class_name[0] + name + "_combined.png", final_img)
    cv2.imwrite(path_folder + class_name[0] + name + "_cam.png", cam_img)

def _get_target_layer(model_name, model: torch.nn.Module):
    target_layer = []
    if "ResNet" in model_name: 
        for name, module in model.named_children():
            # print(name)
            if(name == "layer4"):
                target_layer.append(module[-1])
            # elif (name == "avgpool"):
            #     target_layer.append(module)
    
    
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
            # if name == "permute":
            #     print(f"\t\t Appended: ==> {sub}")
            #     target_layer.append(module)
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

def get_cam_images(model:torch.nn.Module, model_name, imgs_dict:dict):
    _, model = next(model.named_children())
    target_layers = _get_target_layer(model_name=model_name, model=model)
    TARGET_MAP = {
        "Hemorrhagic": 0,
        "Ischemic": 1,
        "Normal": 2
    }
    i=0
    cams = {}
    for class_name, img_list in imgs_dict.items():
        targets = [ClassifierOutputSoftmaxTarget(TARGET_MAP[class_name])]
        final_images = {}
        cams[class_name] = []
        for img in img_list:
            input_tensor = img["input_tensor"].unsqueeze(0).to("cuda:0")
            final_images["main_img"] = img["main_img"]
            final_images["overlay_img"] = img["overlay_img"]

            with ScoreCAM(model=model, target_layers=target_layers, reshape_transform=_reshape_transform) as cam:
                # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
                cam_img = cam(input_tensor=input_tensor, targets=targets)
                # In this example grayscale_cam has only one image in the batch:
                cam_img = cam_img[0, :]
                final_images["cam_img"] = cam_img*255
                cams[class_name].append(cam_img)
                main_and_cam_img = show_cam_on_image(img["norm_img"], cam_img, use_rgb=True, image_weight=0.5)
                final_images["main_and_cam_img"] = main_and_cam_img
                output_prob = torch.nn.functional.softmax(cam.outputs, dim=1)[0][TARGET_MAP[class_name]]
            
            save_images(final_images, class_name,f"{str(i)}_P[{output_prob: 0.4f}]", MODEL_NAME)
            i+=1

def normalize_img(img):
    total = np.sum(img)
    return img/total

def get_confidence_score(cam: list):
    normalized_imgs = [normalize_img(img) for img in cam]
    avg_dist = np.mean(normalized_imgs, axis=0)
    js_divergence = sum(
        1/len(normalized_imgs) * jensenshannon(img.flatten(), avg_dist.flatten(), base=2)**2 for img in normalized_imgs
    )
    return (1-js_divergence)*100

cam_paths = {
    "Hemorrhagic": [
        [
            "Test Images/Output/CvT_S/H0_P[ 1.0000]_cam.png",
            "Test Images/Output/ResNet_S/H0_P[ 1.0000]_cam.png",
            "Test Images/Output/SWIN_S/H0_P[ 1.0000]_cam.png"
        ],
        [
            "Test Images/Output/CvT_S/H1_P[ 0.9999]_cam.png",
            "Test Images/Output/ResNet_S/H1_P[ 1.0000]_cam.png",
            "Test Images/Output/SWIN_S/H1_P[ 1.0000]_cam.png"
        ]
    ],
    "Ischemic": [
        [

            "Test Images/Output/CvT_S/I2_P[ 0.9984]_cam.png",
            "Test Images/Output/ResNet_S/I2_P[ 1.0000]_cam.png",
            "Test Images/Output/SWIN_S/I2_P[ 1.0000]_cam.png"
        ],
        [
            "Test Images/Output/CvT_S/I3_P[ 0.8963]_cam.png",
            "Test Images/Output/ResNet_S/I3_P[ 0.8303]_cam.png",
            "Test Images/Output/SWIN_S/I3_P[ 0.9953]_cam.png"
        ]
    ],
    "Normal": [
        [
            "Classifiers/Convolutional Networks/ResNet_S/Heatmaps/Normal/CAM_0.jpg",
            "Classifiers/Transformer Networks/CvT_S/Heatmaps/Normal/CAM_0.jpg",
            "Classifiers/Transformer Networks/SWIN_S/Heatmaps/Normal/CAM_0.png"
        ],
        [
            "Classifiers/Convolutional Networks/ResNet_S/Heatmaps/Normal/CAM_1.jpg",
            "Classifiers/Transformer Networks/CvT_S/Heatmaps/Normal/CAM_1.jpg",
            "Classifiers/Transformer Networks/SWIN_S/Heatmaps/Normal/CAM_1.png"
        ]
        
    ]
    
}
calc_cam = False
if __name__ == "__main__":
    if calc_cam:
        MODEL_TYPE_DICT: dict = {"conv": "Convolutional Networks", "trans" : "Transformer Networks"}
        model_names = ["ResNet_S", "SWIN_S", "CvT_S"]
        model_types = ["conv", "trans", "trans"]
        fold = [9, 10, 6]
        for i in range(len(model_names)):
            MODEL_NAME = model_names[i]
            MODEL_TYPE = model_types[i]
            FOLD = fold[i]
            path_model_folder = "Classifiers/" + MODEL_TYPE_DICT[MODEL_TYPE] + "/" + MODEL_NAME + "/"
            path_checkpoint = path_model_folder + f"F{FOLD}_Checkpoint.pth"
            model = load_model(MODEL_NAME, path=path_checkpoint)
            get_cam_images(model, MODEL_NAME, load_images())

    for _class, path_list in cam_paths.items():
        i=0
        for paths in path_list:
            imgs = []
            for img_path in paths:
                imgs.append(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32))
            print(f"For {_class}: Confidence for image {str(i)} ==> {get_confidence_score(imgs)}")
            i+=1
            
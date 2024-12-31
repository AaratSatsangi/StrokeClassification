import os
import json
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
import torch
from Preprocessor import CTPreprocessor

class Config:
    def __init__(self, path_init_file="init.json"):
        
        model_type_dict = {
            "conv": "Convolutional Networks",
            "trans": "Transformer Networks",
            "hybrid": "Hybrid Networks"
        }
        print("Initializing Training Variables...")
        try:
            # Training Variables: Mostly Constant
            with open(path_init_file, "r") as file:
                train_vars:dict = json.load(file)

        except Exception as e:
            print(f"Error decoding JSON from the file: {path_init_file} ==> {e}")
            exit(1)
        
        
        # ============= TRAINING VARIABLES ==================
        self.CURRENT_FOLD = 0
        self.MODEL_NAME = train_vars["MODEL_NAME"]
        self.MODEL_TYPE = train_vars["MODEL_TYPE"]
        self.MODEL_SIZE = (self.MODEL_NAME[-1]).lower()
        self.START_FOLD = train_vars["START_FOLD"]
        self.K_FOLD = train_vars["K_FOLD"]
        self.TRAIN_EPOCHS = train_vars["TRAIN_EPOCHS"]
        self.FINE_TUNE_EPOCHS = train_vars["FINE_TUNE_EPOCHS"]
        self.PERSIST = train_vars["PERSISTENCE"]
        self.IMG_SIZE = tuple(train_vars["IMG_SIZE"])
        self.AUTO_BREAK = train_vars["AUTO_BREAK"]
        self.LRS_PATIENCE = train_vars["LRS_PATIENCE"]
        self.LRS_FACTOR = train_vars["LRS_FACTOR"]

        self.SERVER_USERNAME = train_vars["SERVER_USERNAME"]
        self.SERVER_FOLDER = train_vars["SERVER_FOLDER"]
        self.SERVER_URL = train_vars["SERVER_URL"] if  train_vars["SERVER_URL"] != "" else None
        self.PATH_DATASET_TRAIN = train_vars["PATH_DATASET_TRAIN"]
        self.PATH_DATASET_TEST = train_vars["PATH_DATASET_TEST"]
        
        self.RANDOM_STATE = 26
        self.WORKERS = os.cpu_count()
        self.GENERATOR = torch.Generator().manual_seed(26)
        self.TRANSFORMS_TRAIN = CTPreprocessor(
                img_size=self.IMG_SIZE[1:],
                transformations=[
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485],std=[0.229],inplace=True),
                    transforms.RandomRotation(90),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip()
                ],
                use_mask=False
        )
        self.TRANSFORMS_TEST = IMG_TRANSFORMS_TEST = CTPreprocessor(
                img_size=self.IMG_SIZE[1:],
                transformations=[
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485],std=[0.229],inplace=True),
                ],
                use_mask=False
        )
        self.CRITERION_TRAIN = torch.nn.CrossEntropyLoss()
        self.CRITERION_VAL:torch.nn.CrossEntropyLoss = None
        self.TRAIN_DATA = ImageFolder(self.PATH_DATASET_TRAIN, self.TRANSFORMS_TRAIN)
        self.TEST_DATA = None
        self.CLASS_NAMES = self.TRAIN_DATA.classes

        self.PATH_MODEL_FOLDER = f"Classifiers/{model_type_dict[self.MODEL_TYPE]}/{self.MODEL_NAME}/"
        self.PATH_MODEL_LOG_FOLDER = f"{self.PATH_MODEL_FOLDER}Logs/"
        self.EXPERIMENT_NUMBER = str(sum(1 for file_name in os.listdir(self.PATH_MODEL_FOLDER) if "architecture" in file_name))
        self.PATH_MODEL_LOG_FILE = f"{self.PATH_MODEL_LOG_FOLDER}/architecture_{self.EXPERIMENT_NUMBER}.txt"
        self.PATH_LOSSPLOT_FOLDER = f"{self.PATH_MODEL_FOLDER}Plots/"
        self.PATH_PERFORMANCE_FOLDER = f"{self.PATH_MODEL_FOLDER}Performance/"
        self.PATH_MODEL_SAVE = None 
        self.PATH_LOSSES_SAVE = None
        self.PATH_LOSSPLOT_SAVE = None
        self.PATH_PERFORMANCE_SAVE = None
        if(not os.path.exists(self.PATH_MODEL_LOG_FOLDER)): os.makedirs(self.PATH_MODEL_LOG_FOLDER)
        if(not os.path.exists(self.PATH_LOSSPLOT_FOLDER)): os.makedirs(self.PATH_LOSSPLOT_FOLDER)
        if(not os.path.exists(self.PATH_PERFORMANCE_FOLDER)): os.makedirs(self.PATH_PERFORMANCE_FOLDER)
        
        # Model Specific Variables
        print("Initializing Model Specific Variables...")    
        try:
            with open(self.PATH_MODEL_FOLDER + "init.json", "r") as file:
                model_vars:dict = json.load(file)
                model_vars["COMPLETED_FOLD"] = self.CURRENT_FOLD+1
        except Exception as e:
            print(f"Error decoding JSON from the file: {self.PATH_MODEL_FOLDER + 'init.json'} ==> {e}")
            exit(1)

        # =========== MODEL SPECIFIC VARIABLES =================
        self.DEVICE = train_vars["DEVICE"]
        self.BATCH_SIZE = model_vars["BATCH_SIZE"]
        self.BATCH_LOAD = model_vars["BATCH_LOAD"]
        self.LEARNING_RATE = model_vars["LEARNING_RATE"]
        self.FREEZE_TO_LAYER = model_vars["FREEZE_TO_LAYER"]

    def updateFold(self, fold:int):
        self.CURRENT_FOLD = fold
        self.PATH_MODEL_SAVE = f"{self.PATH_MODEL_FOLDER}F{self.CURRENT_FOLD+1}_Checkpoint.pth"
        self.PATH_LOSSES_SAVE = f"{self.PATH_MODEL_FOLDER}F{self.CURRENT_FOLD+1}_Losses.txt"
        self.PATH_LOSSPLOT_SAVE = f"{self.PATH_LOSSPLOT_FOLDER}F{self.CURRENT_FOLD+1}_lossplot.png"
        self.PATH_PERFORMANCE_SAVE = f"{self.PATH_PERFORMANCE_FOLDER}F{self.CURRENT_FOLD+1}_performance.json"
        with open(self.PATH_MODEL_FOLDER + "init.json", "r") as file:
            model_vars:dict = json.load(file)
            model_vars["COMPLETED_FOLD"] = self.CURRENT_FOLD
        with open(self.PATH_MODEL_FOLDER + "init.json", "w") as file:
            json.dump(model_vars, file, indent=4)


if __name__ == "__main__":
    consts = Config()
    print(json.dumps(str(vars(consts)), indent=4))

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchinfo import summary
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from collections import Counter

torch.manual_seed(26)

MODEL_NAME = "CNN_3"
BATCH_SIZE = 32
IMG_SIZE = (1, 256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATH_DATASET = "./Data/Compiled/PNG/" 
PATH_MODEL_FOLDER = PATH_MODEL = None
IMG_TRANSFORMS = transforms.Compose([
    transforms.Resize(IMG_SIZE[1:]),
    transforms.Grayscale(),
    transforms.ToTensor()
])
CLASS_NAMES = None
TOTAL_POINTS = 2000
data_X: np.array
data_Y: np.array
data_loader: DataLoader


def setup():
    global CLASS_NAMES, data_loader
    dataset = ImageFolder(PATH_DATASET, IMG_TRANSFORMS)
    data_loader = DataLoader(dataset = dataset, batch_size = BATCH_SIZE, shuffle=True)
    CLASS_NAMES = dataset.classes

def setup_data(model: nn.Module):
    global data_X, data_Y, data_loader
    d_x = []
    d_y = []
    for XY in data_loader:
        if(len(d_x) >= TOTAL_POINTS): break
        imgs = XY[0].to(DEVICE)
        labels = XY[1]
        model_output = model(imgs)
        for i in range(labels.shape[0]):
            if(len(d_x) >= TOTAL_POINTS): break
            d_x.append(model_output[i].cpu().detach().numpy())  # Move to CPU and convert to NumPy
            d_y.append(labels[i].cpu().detach().numpy())  # Move to CPU and convert to NumPy
            

    scaler = StandardScaler()
    data_X = np.array(d_x)
    data_X = scaler.fit_transform(d_x)
    data_Y = np.array(d_y)
    print(f"Total Loaded Data: {data_X.shape}")

def setup_paths(model_name: str):
    global PATH_MODEL_FOLDER, PATH_MODEL
    PATH_MODEL_FOLDER = "./Classifiers/" + model_name + "/"
    PATH_MODEL = PATH_MODEL_FOLDER + model_name + ".pt"

def getIdx_lastLinearLayer(model: nn.Module):

    module = list(model.children())[0]
    model_layers = list(module.named_children())[::-1]
    prediction_layer_found = False
    for idx, (name, layer) in enumerate(model_layers):
        if(isinstance(layer, nn.Linear) and not prediction_layer_found):
            prediction_layer_found = True
            continue
        elif((isinstance(layer, nn.Linear) or isinstance(layer, nn.Flatten)) and prediction_layer_found):
            print(f">>>Found Break Layer: {type(layer).__name__} at index: {len(model_layers) - idx - 1}")
            if(isinstance(layer, nn.Linear)): return len(model_layers) - idx # return the activation layer
            else: return len(model_layers) - idx - 1 #Else if flatten layer is found then return that layer
    raise Exception("Did not find any Linear Layers")
        
def break_model(model: nn.Module, breakLayer: int, print_summary: bool = True):
    
    module = list(model.children())[0]
    totalLayers: int = len(list(module.children()))
    
    if(breakLayer < 0 or breakLayer >= totalLayers):
        raise(f"Invalid Layer Index: {breakLayer}\n Maximum Allowed: {totalLayers}")
    layers = list(module.children())[:breakLayer + 1]
    sub_model = nn.Sequential(*layers)

    if(print_summary):
        summary(sub_model,(1, ) + IMG_SIZE, col_names=["input_size","output_size","num_params"])
    return sub_model

def save_plot(path):
    plt.savefig(path)
    plt.close()

def plot_tSNE(data_X, data_Y):
    perplexity_values = [5, 10, 15, 25, 35, 50, 100]
    for p in tqdm(perplexity_values):
        tsne = TSNE(n_components=2, perplexity=p, max_iter=2000)
        tsne_X = tsne.fit_transform(data_X)

        plt.figure(figsize=(8, 6))
        for label in np.unique(data_Y):
            plt.scatter(tsne_X[data_Y == label, 0], tsne_X[data_Y == label, 1], label = CLASS_NAMES[label], alpha=0.7)

        # Set plot titles and labels
        plt.title(f't-SNE Visualization (Perplexity={p})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title="Classes")

        # Save the plot to a file
        plot_name = f'tsne_perplexity_{p}.png'
        plot_saveFolder_path = PATH_MODEL_FOLDER + "Plots/tSNE_Plots/"
        if(not os.path.exists(plot_saveFolder_path)):
            os.makedirs(plot_saveFolder_path, exist_ok=True)
    
        save_plot(plot_saveFolder_path + plot_name)
        print(".", end="")
    
    print("\nt-SNE plots saved successfully.")

def plot_PCA(data_X, data_Y):

    pca = PCA(n_components=2)
    pca_X = pca.fit_transform(data_X)

    plt.figure(figsize=(8, 6))
    for label in np.unique(data_Y):
        plt.scatter(pca_X[data_Y == label, 0], pca_X[data_Y == label, 1], label = CLASS_NAMES[label], alpha=0.7)

    
    # Set plot titles and labels
    plt.title('PCA Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title="Classes")

    # Save the plot to a file 
    plot_saveFolder_path = PATH_MODEL_FOLDER + "Plots/"
    if(not os.path.exists(plot_saveFolder_path)):
        os.makedirs(plot_saveFolder_path, exist_ok=True)
    save_plot(plot_saveFolder_path + 'pca_plot.png')
    print("PCA plot saved successfully.")
# def make_video():
#     pass
# def save_video():
#     pass


if __name__ == "__main__":
    model: nn.Module
    sub_model: nn.Module
    setup()
    
    if(MODEL_NAME != ""):
        print("\n", "#"*70, sep="")
        print("\t\tMODEL NAME:", MODEL_NAME)
        print("#"*70)
        setup_paths(MODEL_NAME)

        # break model at layer L from the end     
        model = torch.load(PATH_MODEL)
        breakIdx = getIdx_lastLinearLayer(model)
        sub_model = break_model(model=model, breakLayer=breakIdx, print_summary=False)
        sub_model.eval()
        sub_model.to(DEVICE)
        setup_data(sub_model)
        plot_tSNE(data_X=data_X, data_Y=data_Y)
        plot_PCA(data_X=data_X, data_Y=data_Y)

    else:
        # For each model in the Classifier list
        for model_name in os.listdir("./Classifiers"):
            print("\n", "#"*70, sep="")
            print("\t\tMODEL NAME:", model_name)
            print("#"*70)
            setup_paths(model_name)

            # break model at layer L from the end     
            model = torch.load(PATH_MODEL)
            breakIdx = getIdx_lastLinearLayer(model)
            sub_model = break_model(model=model, breakLayer=breakIdx, print_summary=False)
            sub_model.eval()
            sub_model.to(DEVICE)
            setup_data(sub_model)
            plot_tSNE(data_X=data_X, data_Y=data_Y)
            plot_PCA(data_X=data_X, data_Y=data_Y)
            
        
        # plot 2d PCA to have a global structure
        # plot tSNE to have a neighborhood structure
        # make video of the plots

    del data_X
    del data_Y
    del model
    del sub_model
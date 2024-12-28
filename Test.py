import os
from sklearn.metrics import classification_report
import pandas as pd
from Utils.Helpers import *
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import torch
from Logger import MyLogger

def _binarizeUsingMax(t:torch.tensor):
        max_values, _ = t.max(dim=1, keepdim=True)
        return torch.where(t == max_values, torch.tensor(1.0), torch.tensor(0.0)).numpy()

def _calcPerformMetrics(y_pred, y_true, class_names, path_saveDict, logger:MyLogger):
    y_pred = _binarizeUsingMax(y_pred)
    y_true = _binarizeUsingMax(y_true)
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names, output_dict=True, zero_division=0)
    with open(path_saveDict, 'w') as f:
        json.dump(report, f, indent=4)
        logger.log("Results Written in: " + str(path_saveDict))

    saveAsTable(path_saveDict)
    return

def saveAsTable(json_file_path: str, path_model_folder, logger:MyLogger):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Convert JSON data into a DataFrame
    df: pd.DataFrame
    df = pd.DataFrame(data).T  # Transpose to get categories as rows
    df = df.applymap(lambda x: round(x, 3) if isinstance(x, float) else x)
    df = df.iloc[:3,:3]

    # Set up a Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, len(df) * 0.8))  # Adjust figure height based on rows

    # Hide axes
    ax.axis('tight')
    ax.axis('off')

    # Render the DataFrame as a table
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     cellLoc='center',
                     loc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust table scale

    # Save the table as an image
    count = 0
    for file_name in os.listdir(path_model_folder):
        if("performance" in file_name and ".png" in file_name): 
            count +=1
    output_path = path_model_folder + "performance_" + str(count) + ".png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.log(f"Table image saved to {output_path}")

def testModel(t_model: torch.nn.Module):
    y_trueTensor = torch.empty(0,3)
    y_predTensor = torch.empty(0,3)
    test_class_weights, _ = get_sample_weights(TEST_DATA, None, "Test")
    CRITERION_TEST = nn.CrossEntropyLoss(weight=test_class_weights.to(DEVICE))
    with torch.no_grad():
        test_loss = 0.0
        for test_XY in test_loader:
            x = test_XY[0].to(DEVICE)
            y = test_XY[1].to(DEVICE)

            y_pred =  t_model(x)
            test_loss += CRITERION_TEST(y_pred, y).item()

            y_true = torch.zeros(y.shape[0],3)
            for row in range(y.shape[0]):
                y_true[row, y[row]] = 1
            y_trueTensor = torch.vstack([y_trueTensor, y_true.cpu()])
            y_predTensor = torch.vstack([y_predTensor, torch.nn.functional.softmax(y_pred, dim=1).cpu()])

    test_loss /= len(test_loader)
    count = 0
    for file_name in os.listdir(PATH_MODEL_SAVE):
        if("performance" in file_name and ".json" in file_name): 
            count +=1
    path_perfom_save = PATH_MODEL_SAVE + "performance_" + str(count) + ".json"
    _calcPerformMetrics(y_pred=y_predTensor, y_true=y_trueTensor,class_names=CLASS_NAMES, path_saveDict=path_perfom_save)
    LOGGER.log(f"Test Loss:{round(test_loss,5)}")
    return

if __name__ == "__main__":
    LOGGER.log("Testing model: " + PATH_MODEL_SAVE + MODEL_NAME + ".pt")
    test_loader = DataLoader(dataset = TEST_DATA, batch_size = BATCH_LOAD, num_workers=WORKERS)
    testModel(torch.load(PATH_MODEL_SAVE + MODEL_NAME + ".pt"))
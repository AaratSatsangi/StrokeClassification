import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import json
from sklearn.metrics import classification_report
from Classifiers_CNN import CNN_Classifier
torch.manual_seed(26)

TEST_MODEL_NAME = "CNN_4"


PATH_DATASET = "./Data/Compiled/PNG/"
PATH_MODEL_SAVE = "./Classifiers/" + TEST_MODEL_NAME + "/"
PATH_TEST_MODEL = PATH_MODEL_SAVE + TEST_MODEL_NAME + ".pt"
BATCH_SIZE = 32
IMG_SIZE = (1, 256, 256)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"
IMG_TRANSFORMS = transforms.Compose([
    transforms.Resize(IMG_SIZE[1:]),
    transforms.Grayscale(),
    transforms.ToTensor()
])
LOSS = torch.nn.CrossEntropyLoss()
torch.manual_seed(26)

CLASS_NAMES = None
test_loader : DataLoader = None
model: CNN_Classifier = None

def setup():
    global test_loader, model, CLASS_NAMES
    
    dataset = ImageFolder(PATH_DATASET, IMG_TRANSFORMS)
    _, _, test_dataset = random_split(dataset, [0.6, 0.2, 0.2])
    test_loader = DataLoader(dataset = test_dataset, batch_size = BATCH_SIZE)
    CLASS_NAMES = dataset.classes

    # torch.cuda.empty_cache()
    model = torch.load(PATH_TEST_MODEL).to(DEVICE)
    # model.to(DEVICE)
    model.getSummary()

# Implement for Independent Model Testing
def _binarizeUsingMax(t:torch.tensor):
        max_values, _ = t.max(dim=1, keepdim=True)
        return torch.where(t == max_values, torch.tensor(1.0), torch.tensor(0.0)).numpy()

def _calcPerformMetrics(y_pred, y_true, class_names, path_saveDict):
    y_pred = _binarizeUsingMax(y_pred)
    y_true = _binarizeUsingMax(y_true)
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names, output_dict=True, zero_division=0)
    with open(path_saveDict, 'w') as f:
        json.dump(report, f, indent=4)
        print("Results Written in:", path_saveDict)

    print("\nResults:")
    print(json.dumps(report, indent=4))
    return


def testModel(t_model: CNN_Classifier):
    y_trueTensor = torch.empty(0,3)
    y_predTensor = torch.empty(0,3)
    with torch.no_grad():
        test_loss = 0.0
        for test_XY in test_loader:
            x = test_XY[0].to(DEVICE)
            y = test_XY[1].to(DEVICE)

            y_pred = t_model(x)
            error = LOSS(y_pred, y)

            y_true = torch.zeros(y.shape[0],3)
            for row in range(y.shape[0]):
                y_true[row, y[row]] = 1
            y_trueTensor = torch.vstack([y_trueTensor, y_true.cpu()])
            y_predTensor = torch.vstack([y_predTensor, torch.nn.functional.softmax(y_pred, dim=1).cpu()])

    test_loss /= len(test_loader)
    print("\t" +"Test Loss:", round(test_loss,5))
    path_perfom_save = PATH_MODEL_SAVE + "performance.json"
    _calcPerformMetrics(y_pred=y_predTensor, y_true=y_trueTensor,class_names=CLASS_NAMES, path_saveDict=path_perfom_save)
    return

if __name__ == "__main__":
    setup()
    testModel(model)

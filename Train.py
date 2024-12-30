import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from Utils.Helpers import *
from Classifiers import ConvNets, TransNets
from Logger import MyLogger
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import numpy as np
import os
from sklearn.model_selection import KFold
from torchinfo import summary
import time

def load_model(fold:int=0, load_best=False, fineTune = False):
    torch.cuda.empty_cache()
    CONFIG.updateFold(fold)
    
    if "SWIN" in CONFIG.MODEL_NAME:
        model = TransNets.SWIN(model_size=CONFIG.MODEL_SIZE, freezeToLayer=CONFIG.FREEZE_TO_LAYER, pretrained = not fineTune)
    elif "CvT" in CONFIG.MODEL_NAME:
        model = TransNets.CvT(model_size=CONFIG.MODEL_SIZE, freezeToLayer=CONFIG.FREEZE_TO_LAYER, pretrained = not fineTune)
    elif "MaxViT" in CONFIG.MODEL_NAME:
        model = TransNets.MaxViT(model_size=CONFIG.MODEL_SIZE, freezeToLayer=CONFIG.FREEZE_TO_LAYER, pretrained = not fineTune)
    elif "ResNet" in CONFIG.MODEL_NAME:
        model = ConvNets.ResNet(model_size=CONFIG.MODEL_SIZE, freezeToLayer=CONFIG.FREEZE_TO_LAYER, pretrained = not fineTune)
    else:
        print(f"Error: {CONFIG.MODEL_NAME} not recognized!")
        exit(1)
    
    model.to(CONFIG.DEVICE)
    optim = torch.optim.SGD(params=model.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=0.0005, dampening=0, momentum=0.9, nesterov=True)      
    lr_schedular = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)
    if load_best:
        # Load the best model
        checkpoint = torch.load(CONFIG.PATH_MODEL_SAVE)
        model.load_state_dict(checkpoint["model_state_dict"])
        if fineTune:
            optim.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_schedular.load_state_dict(checkpoint["lr_schedular_state_dict"])
            LOGGER.log("\t" + "+"*100)
            LOGGER.log("\t"*6 + "STARTING FINE TUNING")
            LOGGER.log("\t" + "+"*100)
            LOGGER.log(f"\tMin Train Loss: [{checkpoint['train_loss']: 0.5f}] at Epoch {checkpoint['epoch']}")
            LOGGER.log(f"\tLoading Best {CONFIG.MODEL_NAME} Model for Fold: [{fold+1}/{CONFIG.K_FOLD}]")
        
            # Open all Layers
            for _, param in model.named_parameters():
                param.requires_grad = True
        
    else:
        LOGGER.log(f"\n\tNew {CONFIG.MODEL_NAME} loaded successfully")

    save_arch(model=model, fineTune=fineTune)  
    return model, optim, lr_schedular

def save_model(model:nn.Module, optim:torch.optim.SGD, lr_schedular:CosineAnnealingWarmRestarts, epoch, train_loss):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "lr_schedular_state_dict": lr_schedular.state_dict(),
        "train_loss": train_loss,
        "epoch": epoch
    }, CONFIG.PATH_MODEL_SAVE)

def save_arch(model:nn.Module, fineTune=False):
    # Check Architecture Folder Exists or not
    path = f"{CONFIG.PATH_MODEL_FOLDER}Architecture/"
    if not os.path.exists(path):
        os.mkdir(path)
    
    # Write Architecture
    path += f"arch{'_FT' if fineTune else ''}_{CONFIG.EXPERIMENT_NUMBER}.txt" 
    last_freezed_layer = CONFIG.FREEZE_TO_LAYER if not fineTune else ""
    with open(path, "w") as f:
        f.write("="*25 + "Layer Names" + "="*25 + "\n")
        for i, (name, param) in enumerate(model.named_parameters()):
            if last_freezed_layer in name and last_freezed_layer != "":
                f.write(str(i) + ": " + name + "\t\t(freezed till here)\n")
            else:
                f.write(str(i) + ": " + name + "\n")
        f.write("="*61 + "\n")
        f.write("\n\n")
        f.write(str(summary(model, (1,) + CONFIG.IMG_SIZE, depth=8 , col_names=["input_size","output_size","num_params"], verbose=0)))

def scheduler_step(schedular, lr, **kwargs):
    if(isinstance(schedular, CosineAnnealingWarmRestarts)): 
        schedular.step()
        if(lr > schedular.get_last_lr()[-1]):
            LOGGER.log(f"\t\t(-) Learning Rate Decreased: [{lr: 0.2e}] --> [{schedular.get_last_lr()[-1]: 0.2e}]")
            lr = schedular.get_last_lr()[-1]
        elif(lr < schedular.get_last_lr()[-1]):
            LOGGER.log(f"\t\t(+) Learning Rate Increased: [{lr: 0.2e}] --> [{schedular.get_last_lr()[-1]: 0.2e}]")
            lr = schedular.get_last_lr()[-1]

    elif(isinstance(schedular, ReduceLROnPlateau)): 
        schedular.step(kwargs["val_loss"])
        if(lr > schedular.get_last_lr()[-1]):
            LOGGER.log(f"\t\t(-) Learning Rate Decreased: [{lr: 0.2e}] --> [{schedular.get_last_lr()[-1]: 0.2e}]")
            lr = schedular.get_last_lr()[-1]
        elif(lr < schedular.get_last_lr()[-1]):
            LOGGER.log(f"\t\t(+) Learning Rate Increased: [{lr: 0.2e}] --> [{schedular.get_last_lr()[-1]: 0.2e}]")
            lr = schedular.get_last_lr()[-1]
    else: raise Exception("LR Schedular not recognized!\nType: " + type(schedular))
    return lr

def early_stop(p_counter, training_losses):
    if(p_counter-1 >= CONFIG.PERSIST):
        LOGGER.log("\t" + f"\tValidation Loss not decreasing for {CONFIG.PERSIST}")
        if(is_decreasing_order(training_losses[-CONFIG.PERSIST:])):
            LOGGER.log("\t" + f"\tStopping Training: Overfitting Detected")
            # Break out of Training Loop
            if(CONFIG.AUTO_BREAK): 
                p_counter = 1
                return True, p_counter
        else:
            LOGGER.log("\t" + "\tTraining Loss Fluctuating")
        
        # Unsure about Overfitting, ask the user to continue
        while(True):
            if(CONFIG.AUTO_BREAK):
                flag = "n"
                break
            flag = input("\t" + "Keep Training? (y/n) : ")
            if(flag == "y" or flag == "n"):
                break
            else:
                LOGGER.log("\t" + "Wrong Input!!\n")
        
        p_counter = 1
        if(flag == "n"):    
            return True, p_counter
        else:
            return False, p_counter
    else:
        return False, p_counter

def train_KCV():
    LOGGER.log("\n" + "#"*115 + "\n")
    LOGGER.log("\t\t\t\t\tTraining: " + CONFIG.MODEL_NAME)
    LOGGER.log("\n" + "#"*115)

    # fold_min_val_loss = []
    precision_values = {key: [] for key in CONFIG.CLASS_NAMES}
    recall_values = {key: [] for key in CONFIG.CLASS_NAMES}
    f1_values = {key: [] for key in CONFIG.CLASS_NAMES}
    try:
        for fold, (train_idx, val_idx) in enumerate(KF.split(CONFIG.TRAIN_DATA)):
            LOGGER.log("\t" + "="*100)
            LOGGER.log(f"\tFold {fold+1}/{CONFIG.K_FOLD}")
            LOGGER.log("\t" + "="*100)

            _train = Subset(CONFIG.TRAIN_DATA, train_idx)
            _val = Subset(CONFIG.TRAIN_DATA, val_idx)

            _, sample_weights_train = get_sample_weights(CONFIG.TRAIN_DATA, train_idx, "Train", logger = LOGGER)
            val_class_weights, sample_weights_val = get_sample_weights(CONFIG.TRAIN_DATA, val_idx, "Val", logger = LOGGER)

            
            CONFIG.CRITERION_VAL = nn.CrossEntropyLoss(weight=val_class_weights.to(CONFIG.DEVICE))
            SAMPLER_TRAIN = WeightedRandomSampler(weights=sample_weights_train , num_samples=len(sample_weights_train), replacement=True, generator=CONFIG.GENERATOR)
            
            train_loader = DataLoader(dataset = _train, batch_size = CONFIG.BATCH_LOAD, num_workers=CONFIG.WORKERS//2, pin_memory=True, sampler=SAMPLER_TRAIN, generator=CONFIG.GENERATOR, persistent_workers=True, prefetch_factor=4)
            val_loader = DataLoader(dataset = _val, batch_size = CONFIG.BATCH_LOAD, num_workers=CONFIG.WORKERS//2, pin_memory=True, generator=CONFIG.GENERATOR, persistent_workers=True, prefetch_factor=4)
            
            #Initialize New Model for current fold 
            MODEL, OPTIMIZER, LR_SCHEDULER = load_model(fold=fold)
            training_losses = []
            validation_losses = []
            p_counter = 1
            # min_val_loss = float('inf')
            min_train_loss = float('inf')
            lr = CONFIG.LEARNING_RATE

            epoch = 0
            fine_tuning = False
            total_epochs = CONFIG.TRAIN_EPOCHS + CONFIG.FINE_TUNE_EPOCHS
            for epoch in range(total_epochs):
                start_time = time.time()
                LOGGER.log("\t" + f"{('--' if not fine_tuning else '++')}" + "-"*98)
                LOGGER.log("\t" + f"FOLD: [{fold+1}/{CONFIG.K_FOLD}]")
                LOGGER.log("\t" + f"EPOCH: [{epoch+1}/{total_epochs}]" + "\t"*8 + f"PERSISTENCE: [{p_counter}/{CONFIG.PERSIST}]")
                LOGGER.log("\t" + f"{('--' if not fine_tuning else '++')}" + "-"*98)

                train_loss = 0.0
                accum_loss = 0.0
                count = 0
                # Training 1 Epoch
                MODEL.train()
                for step, train_XY in enumerate(train_loader, 0):
                    
                    # Extract X and Y
                    imgs = train_XY[0].to(CONFIG.DEVICE)
                    labels = train_XY[1].to(CONFIG.DEVICE)
                    
                    # Predict labels 
                    y_pred = MODEL(imgs)

                    # Calculate Error
                    error = CONFIG.CRITERION_TRAIN(y_pred, labels)
                    error.backward()
                    accum_loss += error.item()
                    
                    print("\t" +"\tSTEP: [%d/%d]" % (step+1,len(train_loader)), end= "\r")
                    if(count*CONFIG.BATCH_LOAD >= CONFIG.BATCH_SIZE):    
                        OPTIMIZER.step()
                        OPTIMIZER.zero_grad()
                        train_loss += accum_loss
                        print("\t" +"\tSTEP: [%d/%d]\t\t\t\t\t\t>>>>>Batch Loss: [%0.5f]" % (step+1,len(train_loader),accum_loss/(count)), end = "\r") # Print avg batch loss instead of total accum loss
                        accum_loss = 0.0
                        count = 0
                    count += 1
                # avg epoch loss
                train_loss /= len(train_loader)
                training_losses.append(train_loss)
                LOGGER.log("\n\n\t" +"\tTraining Loss: [%0.5f]" % (training_losses[-1]))

                # Validation
                MODEL.eval()
                with torch.no_grad():
                    val_loss = 0
                    for val_XY in val_loader:
                        imgs = val_XY[0].to(CONFIG.DEVICE)
                        labels = val_XY[1].to(CONFIG.DEVICE)

                        y_pred = MODEL(imgs)
                        val_loss += CONFIG.CRITERION_VAL(y_pred, labels).item()
                val_loss /= len(val_loader)
                validation_losses.append(val_loss)
                LOGGER.log("\t" +"\tWeighted Val Loss: [%0.5f]" % (val_loss))

                # Save Best Model with minimum training loss
                p_counter += 1
                if(train_loss < min_train_loss):
                    min_train_loss = train_loss
                    save_model(model=MODEL, optim=OPTIMIZER, lr_schedular=LR_SCHEDULER, epoch=epoch, train_loss=train_loss)
                    p_counter = 1
                LOGGER.log("\t" +"\tMinimum Training Loss: [%0.5f]" % (min_train_loss))

                # Learning Rate Schedular Step
                lr = scheduler_step(LR_SCHEDULER, lr, val_loss = val_loss)

                # Early Stopping for Overfitting Stopping
                stop, p_counter = early_stop(p_counter=p_counter, training_losses=training_losses)
                LOGGER.log("") # Add New Line
                end_time = time.time()
                logTime(start_time, end_time, logger=LOGGER)
                epoch += 1

                if epoch == CONFIG.TRAIN_EPOCHS-1:
                    LOGGER.log("\t" + "-"*100)
                    LOGGER.log("\t" + "\t\t\tFine Tuning")
                    LOGGER.log("\t" + "-"*100)
                    # Open all layers
                    for _, param in MODEL.named_parameters():
                        param.requires_grad = True
                    fine_tuning = True
                    p_counter = 1
                
                elif epoch == total_epochs-1:
                    del MODEL, OPTIMIZER, LR_SCHEDULER
                    LOGGER.log("\t" + "-"*100)
                    LOGGER.log("\t" + f"For Fold [{fold+1}] Testing Model: {CONFIG.PATH_MODEL_SAVE}")
                    # Calculate Performance Metrics
                    MODEL, OPTIMIZER, LR_SCHEDULER = load_model(fold=fold, load_best=True, fineTune=False)
                    MODEL.eval()
                    report = test_model(
                        t_model=MODEL,
                        test_loader = val_loader,
                        test_class_weights = val_class_weights,
                        device = CONFIG.DEVICE,
                        path_save=CONFIG.PATH_PERFORMANCE_SAVE,
                        class_names=CONFIG.CLASS_NAMES,
                        logger = LOGGER
                    )

                    for _class in CONFIG.CLASS_NAMES:
                        metrics = report[_class]
                        precision_values[_class].append(metrics["precision"])
                        recall_values[_class].append(metrics["recall"])
                        f1_values[_class].append(metrics["f1-score"])
                        LOGGER.log(f"\t\tClass: {_class}")
                        LOGGER.log(f"\t\t|--- Precision: {metrics["precision"]}")
                        LOGGER.log(f"\t\t|--- Recall: {metrics["recall"]}")
                        LOGGER.log(f"\t\t|--- F1-Score: {metrics["f1-score"]}")

                    del train_loader, val_loader
                    break
                        
            np.savetxt(CONFIG.PATH_LOSSES_SAVE, verify_lengths(training_losses, validation_losses), fmt="%0.5f", delimiter=",")
            
            plot_losses(
                fold=fold,
                training_losses=training_losses, 
                validation_losses=validation_losses,
                save_path=CONFIG.PATH_LOSSPLOT_SAVE, 
                logger=LOGGER
            )
            # fold_min_val_loss.append(min_val_loss)

    except KeyboardInterrupt:
        # Exit Loop code
        LOGGER.log("\t" + "Keyboard Interrupt: Exiting Loop...")
    finally:
        final_values = {}
        for _class in CONFIG.CLASS_NAMES:
            p = torch.tensor(precision_values[_class])
            r = torch.tensor(recall_values[_class])
            f1 = torch.tensor(f1_values[_class])
            final_values[_class] = {
                "precision": precision_values[_class],
                "recall": recall_values[_class],
                "f1-score": f1_values[_class]
            }
            LOGGER.log(f"\tClass: {_class}")
            LOGGER.log(f"\t|--- Precision: mean={p.mean()}, std={p.std()}, median={p.median}")
            LOGGER.log(f"\t|--- Recall: mean={r.mean()}, std={r.std()}, median={r.median}")
            LOGGER.log(f"\t|--- F1-Score: mean={f1.mean()}, std={f1.std()}, median={f1.median}")
        
        with open(CONFIG.PATH_PERFORMANCE_FOLDER + "final_performance.json", "w") as json_file:
            json.dump(final_values, json_file, indent=4)

   



if __name__ == "__main__":
    CONFIG = Config()
    KF = KFold(n_splits=CONFIG.K_FOLD, shuffle=True, random_state=CONFIG.RANDOM_STATE)
    LOGGER = MyLogger(
        server_url = CONFIG.SERVER_URL,
        server_username = CONFIG.SERVER_USERNAME,
        server_folder = CONFIG.SERVER_FOLDER,
        model_name = CONFIG.MODEL_NAME,
        path_localFile = CONFIG.PATH_MODEL_LOG_FILE
    )

    OPTIMIZER: torch.optim.SGD = None
    LR_SCHEDULER: CosineAnnealingWarmRestarts = None
    SAMPLER: WeightedRandomSampler = None
    MODEL: torch.nn.Module = None

    LOGGER.log("\n\n" + "="*54 + " START " + "="*54)
    LOGGER.log(f"Training Epochs: {CONFIG.TRAIN_EPOCHS}")
    LOGGER.log(f"Fine Tuning Epochs: {CONFIG.FINE_TUNE_EPOCHS}")
    LOGGER.log(f"Using GPU: {CONFIG.DEVICE}")
    LOGGER.log(f"Batch Size: {CONFIG.BATCH_SIZE}")
    LOGGER.log(f"Learning Rate: {CONFIG.LEARNING_RATE}")
    LOGGER.log(f"Early Stopping with Persistence: {CONFIG.PERSIST}")
    LOGGER.log(f"LR Schedular: CosineAnnealingWarmRestarts")
    # Add if statement
    if isinstance(LR_SCHEDULER, ReduceLROnPlateau):
        LOGGER.log(f"|---Patience: {CONFIG.LRS_PATIENCE}")
        LOGGER.log(f"|---Factor: {CONFIG.LRS_FACTOR}")

    train_KCV()
    LOGGER.log("\n\n" + "="*55 + " END " + "="*55 + "\n\n")
    

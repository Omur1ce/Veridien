import math
import torch
import csv
import torch.optim as optim
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import os
import segmentation_models_pytorch as smp



def train_model_loop(model, train_loader, val_loader, loss_fn, optimizer, num_epochs, save_dir=None, 
                     use_AMP=True, pinned_mem=True, scheduler=None, start_epoch=0, device=None):
    """
    Trains model, saves model checkpoints and records stats
     - `device` should be set to gpu automatically
     - `save_dir` auto creates folder at cwd, and checkpoints folder within
     - `return` (train loss, train acc, val loss, val acc)
    """
    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []
    
    best_model_path = "guh"
    best_vloss = 1000000
    best_vacc = 0

    # make checkpoints folder and move to device
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), f"Results_{datetime.now().strftime('%H_%M_%S')}")

    model_save_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(model_save_dir, exist_ok=True)
    
    csv_path = os.path.join(save_dir, "results.csv")
    write_header = not os.path.exists(csv_path)
    
    with open(csv_path, mode="a", newline="") as csv_file:
        writer = csv.writer(csv_file)
        if write_header:
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    # move to device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    print(f"{device=}")

    # automatic mixed prec
    if use_AMP:
        scaler = torch.GradScaler("cuda")
        print("using AMP")
    
    # actual loop
    for epoch in tqdm(range(start_epoch, num_epochs), desc="Epochs"):
        if epoch % 2 == 0:
            torch.cuda.empty_cache()

        # training
        model.train()
        train_loss_epoch = 0.0
        correct_train = 0
        total_train = 0

        if not use_AMP: # no amp :sad:
            for x, y in tqdm(train_loader, desc="Batch training"):
                x, y = x.to(device, non_blocking=pinned_mem), y.to(device, non_blocking=pinned_mem)
                
                optimizer.zero_grad()
                
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()

                train_loss_epoch += loss.detach()
                correct_predictions_train = (torch.argmax(y_pred, dim=1) == y).sum().item()
                total_train += y.size(0)
                correct_train += correct_predictions_train

        else: # AMP :fire:
            for x, y in tqdm(train_loader, desc="Batch training"):
                x, y = x.to(device, non_blocking=pinned_mem), y.to(device, non_blocking=pinned_mem)

                optimizer.zero_grad()

                with torch.autocast("cuda"):
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)

                scaler.scale(loss).backward()  # Scale loss for stability
                scaler.step(optimizer)
                scaler.update()

                train_loss_epoch += loss.detach()
                correct_predictions_train = (torch.argmax(y_pred, dim=1) == y).sum().item()
                total_train += y.size(0)
                correct_train += correct_predictions_train

        # validation
        model.eval()
        val_loss_epoch = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():  
            for x, y in tqdm(val_loader, desc="Batch validation"):
                x, y = x.to(device, non_blocking=pinned_mem), y.to(device, non_blocking=pinned_mem)
                y_pred = model(x)
                val_loss_epoch += loss_fn(y_pred, y).item()

                correct_predictions = (torch.argmax(y_pred, dim=1) == y).sum().item()
                
                total_val += y.size(0)
                correct_val += correct_predictions

        # calculate stats
        train_loss_epoch /= len(train_loader)
        train_acc_epoch = correct_train / total_train
        val_loss_epoch /= len(val_loader)
        val_acc_epoch = correct_val / total_val
        
        train_loss_epoch = float(train_loss_epoch)
        train_acc_epoch = float(train_acc_epoch)
        val_loss_epoch = float(val_loss_epoch)
        val_acc_epoch = float(val_acc_epoch)

        train_loss_arr.append(train_loss_epoch)
        train_acc_arr.append(train_acc_epoch)
        val_loss_arr.append(val_loss_epoch)
        val_acc_arr.append(val_acc_epoch)

        # save checkpoints
        save_filename = f'model_{epoch+1}_{datetime.now().strftime("%H%M%S")}'
        
        if val_loss_epoch < best_vloss:
            best_vloss = val_loss_epoch
            
            if os.path.exists(best_model_path):
                print(f"removing previous {best_model_path}...")
                os.remove(best_model_path)
                
            best_model_path = os.path.join(model_save_dir, save_filename)
            print(f"New best val loss, saving model_{epoch+1} @ {best_model_path}...")
            torch.save(model.state_dict(), best_model_path)
            
        elif val_acc_epoch > best_vacc:
            best_vacc = val_acc_epoch
            
            if os.path.exists(best_model_path):
                print(f"removing previous {best_model_path}...")
                os.remove(best_model_path)
                
            best_model_path = os.path.join(model_save_dir, save_filename)
            print(f"New best val acc, saving model_{epoch+1} @ {best_model_path}...")
            torch.save(model.state_dict(), best_model_path)

        # log stats to csv so its bing chilling if it crashes mid way again
        with open(csv_path, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch + 1, train_loss_epoch, train_acc_epoch, val_loss_epoch, val_acc_epoch])

        print(f"Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss_epoch:.4f}, "
              f"Train Acc: {train_acc_epoch*100:.3f}, "
              f"Val Loss: {val_loss_epoch:.4f}, "
              f"Val Acc: {val_acc_epoch*100:.3f}\n")

        if scheduler:
            scheduler.step()                                                                
    
    try:
        train_loss_arr = train_loss_arr.cpu()
        train_acc_arr = train_acc_arr.cpu()
        val_loss_arr = val_loss_arr.cpu()
        val_acc_arr = val_acc_arr.cpu()
    except:
        pass
        
    return train_loss_arr, train_acc_arr, val_loss_arr, val_acc_arr


def test_model(model, test_loader, device):
    """
    Just runs model on (test) set and returns accuracy
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            predicted = torch.argmax(y_pred, dim=1)
            correct_predictions = sum(predicted == y).item()

            total += y.size(0)
            correct += correct_predictions

    test_acc = correct / total
    print(f"Test acc: {test_acc}")

    return test_acc


def train_model(model, 
                train_loader, val_loader, test_loader, 
                results_base_dir=None, 
                epochs_total = 20,
                lr1=1e-4,
                use_adam=True,
                start_epoch=0,
                load_weights=None):
    
    # create subfolder within results folder
    if results_base_dir:
        results_folder = results_base_dir
    else:
        results_folder = os.path.join("results", datetime.now().strftime("%H_%M_%S"))
    os.makedirs(results_folder, exist_ok=True)
    print(f"{results_folder=}")

    # load from a check point
    if load_weights:
        model.load_state_dict(torch.load(load_weights, weights_only=True))
        print(f"Loaded weights from {load_weights}")
        
    for param in model.parameters(): 
        param.requires_grad = True # set all params to learnable

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} available.")
    model.to(device)
    
    # loss fn
    loss_fn = smp.losses.DiceLoss(mode='multiclass') + nn.CrossEntropyLoss()
    
    # optimizer
    if use_adam:
        def warmup_cos(epoch):
            warmups = 5
            totals = epochs_total
            if epoch < warmups:
                return epoch / warmups  # Linear warmup
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmups) / (totals - warmups)))

        beta1, beta2 = 0.9, 0.999
        optimizer = optim.AdamW(model.parameters(), lr=lr1, betas=(beta1, beta2), weight_decay=0.05)    
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cos)

    
    print(f"Training all layers @ {lr1=} for {epochs_total=}")
    phase_results = train_model_loop(model=model, 
                                     train_loade=train_loader, 
                                     val_loader=val_loader, 
                                     loss_fn=loss_fn, 
                                     optimizer=optimizer, 
                                     num_epochs=epochs_total, 
                                     save_dir=results_folder,
                                     scheduler=scheduler,
                                     start_epoch=start_epoch,
                                     device=device)
    
    print("Training done, testing model...")
    test_acc = test_model(model, test_loader, device)
    print(f"Test acc: {test_acc:.4f}")
    
    # save test stats
    with open(os.path.join(results_folder, "test_acc.txt"), 'w') as f:
        f.write(str(test_acc))

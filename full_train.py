import os, gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from full_scale_classifier import *
from dataset import VideoDataset, custom_collate_fn
import time

def clear_gpu_memory():
    """Function to thoroughly clear GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    torch.backends.cuda.max_memory_allocated = 0
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    clear_gpu_memory()
    if torch.cuda.is_available():
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"Initial GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    BATCH_SIZE = 1
    EPOCHS = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("Loading model")
    latentEncoder = FullLatentEncoder().to(device)
    patchEncoder = FullPatchEncoder().to(device)
    classifier = FullClassifier().to(device)
    vclf = FullVideoClassifier(latentEncoder,patchEncoder,classifier).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.Adam(vclf.parameters(), lr=0.0001,eps=1e-8)
    # Load the preprocessed videos
    real_videos_path = 'data\\many\\real'
    fake_videos_path = 'data\\many\\fake'
    print("Loading dataset")
    dataset = VideoDataset(real_videos_path, fake_videos_path)
    print(f"Dataset length: {len(dataset)}")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    #creating train and test datasets, using a fixed seed for reproducibility
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(314159))
    print("Creating data loaders")
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True, collate_fn=custom_collate_fn,pin_memory=False,num_workers=1)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False, collate_fn=custom_collate_fn,pin_memory=False,num_workers=1)
    # must store test video paths to use in interpret.py and maintain integrity of train/test split
    with open('data\\test_paths.txt', 'w') as f:
        for video in test_dataset:
            f.write(video[2] + '\n')
    print("Begin Training")
    #track time it takes to complete each batch and each epoch
    torch.autograd.set_detect_anomaly(True)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.empty_cache()

    clear_gpu_memory()

    if torch.cuda.is_available():
        print(f"GPU memory before training: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"GPU memory cached before training: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    for epoch in range(EPOCHS):
        start_epoch_time = time.time()
        classes = ['fake', 'real']
        for batch_idx, (data, labels, paths) in enumerate(train_loader):
            #torch.cuda.empty_cache()
            data = data.to(device)
            print("Video dimensions: ",data.shape)
            labels = labels.to(device)
            #print(labels)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = vclf(data)
                pred = torch.argmax(outputs, dim=1)
                #print(pred)
                loss = criterion(outputs, labels)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"NaN or Inf in loss at Epoch {epoch}, Batch {batch_idx}")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vclf.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx} Loss: {loss.item():.2f} prediction: {classes[pred.item()]} target: {classes[labels.item()]} GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB")
            #torch.cuda.empty_cache()
            del data, labels, outputs, loss
            torch.cuda.empty_cache()
        vclf.eval()
        accuracy = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, labels, paths) in enumerate(test_loader):
                data = data.to(device)
                labels = labels.to(device)
                outputs = vclf(data)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
            del data, labels, outputs, loss
            torch.cuda.empty_cache()
        print(f"Test accuracy: {(accuracy/total)*100:.3f}%")
        torch.save(vclf.state_dict(), 'model\\videoClassifier_full.pth')
        end_epoch_time = time.time()
        elapsed_time = end_epoch_time - start_epoch_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Epoch {epoch} took {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")

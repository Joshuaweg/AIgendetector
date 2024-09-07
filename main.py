import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from classifier import *
from dataset import VideoDataset, custom_collate_fn
import time

if __name__ == "__main__":
    torch.cuda.empty_cache()
    BATCH_SIZE = 1
    EPOCHS = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("Loading model")
    latentEncoder = LatentEncoder().to(device)
    patchEncoder = PatchEncoder().to(device)
    classifier = Classifier().to(device)
    vclf = VideoClassifier(latentEncoder,patchEncoder,classifier).to(device)
    vclf.load_state_dict(torch.load('model\\videoClassifier.pth'))
    vclf.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(vclf.parameters(), lr=0.0001)
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
    train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False, collate_fn=custom_collate_fn)
    # must store test video paths to use in interpret.py and maintain integrity of train/test split
    with open('data\\test_paths.txt', 'w') as f:
        for video in test_dataset:
            f.write(video[2] + '\n')
    print("Begin Training")
    #track time it takes to complete each batch and each epoch
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(EPOCHS):
        start_epoch_time = time.time()
        classes = ['fake', 'real']
        for batch_idx, (data, labels, paths) in enumerate(train_loader):
            #torch.cuda.empty_cache()
            data = data.to(device)
            labels = labels.to(device)
            #print(labels)
            optimizer.zero_grad()
            outputs = vclf(data)
            pred = torch.argmax(outputs, dim=1)
            #print(pred)
            loss = criterion(outputs, labels)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"NaN or Inf in loss at Epoch {epoch}, Batch {batch_idx}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vclf.parameters(), max_norm=1.0)
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx} Loss: {loss.item():.2f} prediction: {classes[pred.item()]} target: {classes[labels.item()]} GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB")
            #torch.cuda.empty_cache()
        with torch.no_grad():
            accuracy = 0
            total = 0
            for batch_idx, (data, labels, paths) in enumerate(test_loader):
                data = data.to(device)
                labels = labels.to(device)
                outputs = vclf(data)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
            print(f"Test accuracy: {(accuracy/total)*100:.3f}%")
            torch.save(vclf.state_dict(), 'model\\videoClassifier.pth')
        end_epoch_time = time.time()
        elapsed_time = end_epoch_time - start_epoch_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Epoch {epoch} took {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")

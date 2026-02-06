import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from classifier import *
from dataset import VideoDataset, custom_collate_fn
import time
import gc
import os
import math
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
def enable_memory_efficient_training(model, device):
    """Configure model for memory-efficient training"""
    model = model
    # Enable gradient checkpointing
    if hasattr(model, 'encoder_layer'):
        model.encoder_layer.checkpoint = True
    return model
def load_checkpoint(checkpoint_path, device):
    # Initialize model and optimizer
    latentEncoder = LatentEncoder().to(device)
    patchEncoder = PatchEncoder().to(device)
    classifier = Classifier().to(device)
    model = VideoClassifier(latentEncoder, patchEncoder, classifier)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    
    print(f"Loaded checkpoint from epoch {epoch}")
    return model, optimizer, epoch
def optimize_dataloader(dataset, batch_size):
    """Create memory-efficient dataloader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn,
        pin_memory=False,  # Disable pin_memory to reduce memory usage
        num_workers=0,     # Reduce to 0 to prevent worker memory issues
        persistent_workers=False
    )
def calculate_batch_size(available_memory_gb):
    """Dynamically calculate optimal batch size based on available memory"""
    # Force batch size to 1 due to large video sizes
    return 1
def print_memory_stats(prefix=""):
    """Detailed memory tracking"""
    print(f"{prefix} Memory Stats:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    print(f"  Max Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    print(f"  Max Reserved:  {torch.cuda.max_memory_reserved() / 1e9:.2f}GB")

def get_model_size(model):
    """Get model size in GB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_gb = (param_size + buffer_size) / 1024**3
    return size_all_gb
def train_with_memory_optimization(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    classes = ['real', 'fake']
    checkpoint =13
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        for batch_idx, (data, labels, paths) in enumerate(train_loader):
            # Clear cache before each batch
            torch.cuda.empty_cache()
            gc.collect()
            
            try:
                # Move data to GPU
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, labels)
                pred = torch.argmax(outputs,dim=1)
                correct += (pred == labels).sum().item()
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                running_loss += loss.item()
                
                # Print statistics
                if batch_idx % 10 == 0:
                    with torch.no_grad():
                        pred = torch.argmax(outputs, dim=1)
                        print(f"Epoch {epoch}, Batch {batch_idx} "
                              f"Loss: {running_loss/10:.4f} "
                              f"correct: {correct}/10 "
                              f"GPU Memory: {torch.cuda.memory_allocated(device)/1e9:.2f}GB")
                    running_loss = 0.0
                    correct = 0
                
            except torch.cuda.OutOfMemoryError:
                # Handle OOM error
                torch.cuda.empty_cache()
                gc.collect()
                print(f"OOM at batch {batch_idx}. Skipping batch.")
                if 'data' in locals():
                    del data
                if 'outputs' in locals():
                    del outputs
                continue
            
            # Clean up batch tensors
            del data, labels, outputs, loss
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        print("Validation phase")
        with torch.no_grad():
            for data, labels, paths in test_loader:
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    data = data.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                        
                   
                            
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                
                del data, labels, outputs
            
        print(f'Accuracy on test set: {100 * correct / total:.2f}%')
        
        # Save checkpoint
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'model/videoClassifier_epoch_{checkpoint}.pth')
            checkpoint += 1
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
        
        # Clear memory after epoch
        torch.cuda.empty_cache()
        gc.collect()
def process_in_chunks(model, data, chunk_size):
    """Process large inputs in smaller chunks"""
    B, T = data.shape[0], data.shape[1]
    outputs_list = []
    
    # Calculate number of chunks needed
    n_chunks = math.ceil(T / chunk_size)
    
    for i in range(0, T, chunk_size):
        # Clear cache before processing each chunk
        torch.cuda.empty_cache()
        gc.collect()
        
        # Process chunk
        chunk = data[:, i:min(i+chunk_size, T)]
        try:
            with torch.amp.autocast("cuda"):
                chunk_output = model(chunk)
                outputs_list.append(chunk_output)
        except RuntimeError as e:
            if "out of memory" in str(e):
                # If OOM occurs, try with smaller chunk
                print(f"OOM with chunk size {chunk_size}, trying smaller chunk")
                torch.cuda.empty_cache()
                smaller_chunk_size = chunk_size // 2
                if smaller_chunk_size < 1:
                    raise e
                return process_in_chunks(model, data, smaller_chunk_size)
            else:
                raise e
        
        # Clear chunk from memory
        del chunk
        torch.cuda.empty_cache()
    
    # Combine outputs
    if len(outputs_list) > 0:
        return torch.stack(outputs_list).mean(0)
    else:
        raise RuntimeError("No outputs generated from chunks")
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get available GPU memory
    torch.cuda.empty_cache()
    #available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
    
    # Calculate optimal batch size
    BATCH_SIZE = 1
    print(f"Using batch size: {BATCH_SIZE}")
    
    # Load dataset
    print("Loading dataset...")
    real_videos_path = 'data/many/real'
    fake_videos_path = 'data/many/fake'
    dataset = VideoDataset(real_videos_path, fake_videos_path)
    print(f"Dataset size: {len(dataset)}")
    
    # Split dataset
    from sklearn.model_selection import train_test_split
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=314159
    )
    
    # Create Subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Save test paths
    with open('data/test_paths.txt', 'w') as f:
        for _, _, path in test_dataset:
            f.write(f"{path}\n")
    
    # Initialize model and optimize for memory
    print("Initializing model...")
    model, optimizer, epoch = load_checkpoint('model/videoClassifier_epoch_12.pth', device)
    #model = VideoClassifier(LatentEncoder().to(device), PatchEncoder().to(device), Classifier().to(device)).to(device)
    model= enable_memory_efficient_training(model,device)
    model.to(device)
    
    # Initialize optimizer with gradient clipping
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = optimize_dataloader(train_dataset, BATCH_SIZE)
    test_loader = optimize_dataloader(test_dataset, BATCH_SIZE)
    
    # Training
    print("Starting training...")
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    train_with_memory_optimization(
        model, train_loader, test_loader,
        criterion, optimizer, num_epochs=10,
        device=device
    )


if __name__ == "__main__":
    main()
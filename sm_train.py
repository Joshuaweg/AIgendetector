import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from torch.cuda.amp import GradScaler
import boto3
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
import numpy as np
import torchviz
import GPUtil
import seaborn as sns
import matplotlib.pyplot as plt
import tempfile

from dataset import VideoDataset, custom_collate_fn
from model import FullLatentEncoder, FullPatchEncoder, FullClassifier, FullVideoClassifier

def log_gpu_utilization():
    if not torch.cuda.is_available():
        return {}
        
    gpu_stats = {}
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**2)  # Convert to MB
        memory_allocated = torch.cuda.memory_allocated(i) / (1024**2)  # Convert to MB
        memory_reserved = torch.cuda.memory_reserved(i) / (1024**2)  # Convert to MB
        memory_free = total_memory - memory_allocated
        
        gpu_stats[f"GPU_{i}"] = {
            "total_memory_mb": total_memory,
            "allocated_memory_mb": memory_allocated,
            "reserved_memory_mb": memory_reserved,
            "free_memory_mb": memory_free,
            "utilization": GPUtil.getGPUs()[i].load * 100  # Convert to percentage
        }
        
    return gpu_stats

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_model_graph(model, writer):
    try:
        # If using DataParallel, get the underlying model
        if isinstance(model, nn.DataParallel):
            model_to_trace = model.module
        else:
            model_to_trace = model
            
        # Create dummy input on the same device as model
        device = next(model_to_trace.parameters()).device
        dummy_input = torch.randn(1, 24, 512, 512, 3, device=device)
        
        # Try to add graph with error handling
        try:
            writer.add_graph(model_to_trace, dummy_input)
        except Exception as e:
            print(f"Warning: Failed to add model graph to tensorboard: {str(e)}")
            
    except Exception as e:
        print(f"Warning: Could not generate model graph: {str(e)}")

def plot_confusion_matrix(y_true, y_pred, writer, epoch):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    writer.add_figure('Confusion Matrix', plt.gcf(), epoch)
    plt.close()

def plot_roc_curve(y_scores, y_true, writer, epoch):
    """Plot ROC curve and calculate AUC score.
    
    Args:
        y_scores (array-like): Predicted probabilities or scores
        y_true (array-like): True binary labels
        writer: TensorBoard writer
        epoch (int): Current epoch number
    """
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Validate inputs
    if y_true.shape != y_scores.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_scores {y_scores.shape}")
    if not np.all(np.unique(y_true) == np.array([0, 1])):
        raise ValueError("y_true should contain only binary labels (0 and 1)")
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Epoch {epoch}')
    plt.legend(loc="lower right")
    
    # Add to tensorboard
    writer.add_figure('ROC Curve', plt.gcf(), epoch)
    plt.close()
    
    return roc_auc

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Data and model paths
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DIR'))
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=6, help='Batch size per GPU. Total batch size will be batch_size * num_gpus')
    parser.add_argument('--learning-rate', type=float, default=0.0001)  # Reduced from 0.001
    parser.add_argument('--warmup-epochs', type=int, default=2, help='Number of epochs for learning rate warmup')
    parser.add_argument('--target-size', type=int, default=512)
    parser.add_argument('--max-frames', type=int, default=24)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=2, 
                       help='Number of steps to accumulate gradients. Effective batch size = batch_size * num_gpus * gradient_accumulation_steps')
    
    return parser.parse_args()

def get_lr_multiplier(epoch, warmup_epochs):
    """Calculate learning rate multiplier with linear warmup"""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    return 1.0

def save_checkpoint_to_s3(checkpoint_data, s3_path):
    """Save checkpoint to S3 using boto3"""
    s3_client = boto3.client('s3')
    
    # Parse S3 path
    s3_path = s3_path.replace('s3://', '')
    bucket = s3_path.split('/')[0]
    key = '/'.join(s3_path.split('/')[1:])
    
    # Save checkpoint to temporary file first
    with tempfile.NamedTemporaryFile() as tmp_file:
        torch.save(checkpoint_data, tmp_file.name)
        tmp_file.flush()
        
        # Upload to S3
        try:
            s3_client.upload_file(tmp_file.name, bucket, key)
            return True
        except Exception as e:
            print(f"Error saving checkpoint to S3: {str(e)}")
            return False

def verify_s3_path_exists(s3_path):
    """Verify that a file exists in S3"""
    try:
        s3_client = boto3.client('s3')
        s3_path = s3_path.replace('s3://', '')
        bucket = s3_path.split('/')[0]
        key = '/'.join(s3_path.split('/')[1:])
        
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                raise
    except Exception as e:
        print(f"Error checking S3 path {s3_path}: {str(e)}")
        return False

def train(model, train_loader, val_loader, criterion, optimizer, device, args):
    # Create tensorboard directory and writer with S3 path
    tensorboard_s3_path = 's3://genvideo-dataset-complete/tensorboard'
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    
    # Save training configuration to both local and S3
    config = {
        'initial_learning_rate': args.learning_rate,
        'warmup_epochs': args.warmup_epochs,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'target_size': args.target_size,
        'max_frames': args.max_frames,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.batch_size * torch.cuda.device_count() * args.gradient_accumulation_steps
    }
    
    # Save config locally
    local_config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(local_config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Save config to S3
    config_s3_path = 's3://genvideo-dataset-complete/output/training_config.json'
    save_to_s3(local_config_path, config_s3_path)
    
    scaler = GradScaler()
    best_val_acc = 0
    first_batch_completed = False
    
    # Initialize learning rate scheduler with warmup
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                         factor=0.5, patience=2,
                                                         verbose=True)
    
    # Log model graph and parameters
    generate_model_graph(model, writer)
    model_to_count = model.module if isinstance(model, nn.DataParallel) else model
    writer.add_scalar('Model/Parameters', count_parameters(model_to_count), 0)
    sync_tensorboard_to_s3(os.path.join(args.output_dir, 'tensorboard'), tensorboard_s3_path)
    for epoch in range(args.epochs):
        model.train()
        
        # Apply warmup
        lr_multiplier = get_lr_multiplier(epoch, args.warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate * lr_multiplier
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}: Learning Rate = {current_lr:.6f}")
        
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        all_train_preds = []
        all_train_labels = []
        
        # Running totals for 10-batch intervals
        running_loss = 0
        running_correct = 0
        running_total = 0
        last_log_batch = 0
        
        # Log GPU utilization at epoch start
        gpu_stats = log_gpu_utilization()
        for gpu_id, stats in gpu_stats.items():
            writer.add_scalar(f'{gpu_id}/Total_Memory_MB', stats['total_memory_mb'], epoch)
            writer.add_scalar(f'{gpu_id}/Allocated_Memory_MB', stats['allocated_memory_mb'], epoch)
            writer.add_scalar(f'{gpu_id}/Reserved_Memory_MB', stats['reserved_memory_mb'], epoch)
            writer.add_scalar(f'{gpu_id}/Free_Memory_MB', stats['free_memory_mb'], epoch)
            writer.add_scalar(f'{gpu_id}/Utilization_Percent', stats['utilization'], epoch)
        sync_tensorboard_to_s3(os.path.join(args.output_dir, 'tensorboard'), tensorboard_s3_path)
        for batch_idx, (videos, labels, _) in enumerate(train_loader):
            try:
                # Skip batch if no valid videos
                if videos is None or labels is None:
                    print(f"Skipping invalid batch {batch_idx}")
                    continue
                
                # Clear cache at start of batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                    loss = loss / args.gradient_accumulation_steps  # Scale loss
                
                # Backward pass with gradient accumulation
                scaler.scale(loss).backward()
                
                # Step optimizer every gradient_accumulation_steps
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    # Clip gradients
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                # Update metrics
                with torch.no_grad():
                    _, predicted = outputs.max(1)
                    batch_size = labels.size(0)
                    
                    # Update running totals (10-batch intervals)
                    running_total += batch_size
                    running_correct += predicted.eq(labels).sum().item()
                    running_loss += loss.item() * args.gradient_accumulation_steps
                    
                    # Update epoch totals
                    epoch_total += batch_size
                    epoch_correct += predicted.eq(labels).sum().item()
                    epoch_loss += loss.item() * args.gradient_accumulation_steps
                    
                    all_train_preds.extend(predicted.cpu().numpy())
                    all_train_labels.extend(labels.cpu().numpy())
                
                # Log progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    # Calculate metrics for last 10 batches
                    avg_running_loss = running_loss / (batch_idx + 1 - last_log_batch)
                    avg_running_accuracy = 100. * running_correct / running_total
                    
                    # Log learning rate
                    writer.add_scalar('Train/LearningRate', current_lr, 
                                    epoch * len(train_loader) + batch_idx)
                    
                    print(f"Epoch: {epoch+1}/{args.epochs} | Batch: {batch_idx+1}/{len(train_loader)} | "
                          f"Loss: {avg_running_loss:.4f} | Accuracy: {avg_running_accuracy:.2f}% | "
                          f"LR: {current_lr:.6f}")
                    sync_tensorboard_to_s3(os.path.join(args.output_dir, 'tensorboard'), tensorboard_s3_path)
                    # Reset running totals
                    running_loss = 0
                    running_correct = 0
                    running_total = 0
                    last_log_batch = batch_idx + 1
                
                # Clean up
                del outputs, loss, predicted
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # After first batch completes, verify model saving
                if not first_batch_completed and batch_idx == 0:
                    print("\nVerifying model saving locations after first batch:")
                    
                    # Save initial checkpoint
                    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                    checkpoint_data = {
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'batch': batch_idx
                    }
                    
                    # Test paths
                    checkpoint_name = f'initial_checkpoint_epoch_{epoch+1}_batch_1.pt'
                    models_path = f's3://genvideo-dataset-complete/models/{checkpoint_name}'
                    output_path = f's3://genvideo-dataset-complete/output/initial_model.pt'
                    tensorboard_path = 's3://genvideo-dataset-complete/tensorboard'
                    
                    # Save and verify
                    print("\nSaving and verifying model locations:")
                    
                    # Save and check models directory
                    if save_checkpoint_to_s3(checkpoint_data, models_path):
                        print(f"✓ Successfully saved to models directory: {models_path}")
                    else:
                        print(f"✗ Failed to save to models directory: {models_path}")
                    
                    # Save and check output directory
                    if save_checkpoint_to_s3(checkpoint_data, output_path):
                        print(f"✓ Successfully saved to output directory: {output_path}")
                    else:
                        print(f"✗ Failed to save to output directory: {output_path}")
                    
                    # Verify tensorboard
                    if verify_s3_path_exists(tensorboard_path):
                        print(f"✓ Tensorboard directory exists: {tensorboard_path}")
                    else:
                        print(f"✗ Tensorboard directory not found: {tensorboard_path}")
                    
                    print("\nInitial saving verification complete.\n")
                    first_batch_completed = True
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                if "out of memory" in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        all_val_scores = []
        
        with torch.no_grad():
            for videos, labels, _ in val_loader:
                if videos is None or labels is None:
                    continue
                    
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(videos)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                all_val_scores.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
        
        # Calculate epoch metrics
        train_accuracy = 100. * epoch_correct / epoch_total
        val_accuracy = 100. * val_correct / val_total if val_total > 0 else 0
        train_f1 = f1_score(all_train_labels, all_train_preds)
        val_f1 = f1_score(all_val_labels, all_val_preds)
        
        # Log epoch metrics
        epoch_train_loss = epoch_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Train/EpochLoss', epoch_train_loss, epoch)
        writer.add_scalar('Train/EpochAccuracy', train_accuracy, epoch)
        writer.add_scalar('Train/F1Score', train_f1, epoch)
        writer.add_scalar('Val/EpochLoss', epoch_val_loss, epoch)
        writer.add_scalar('Val/EpochAccuracy', val_accuracy, epoch)
        writer.add_scalar('Val/F1Score', val_f1, epoch)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
        print(f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}\n")
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            
            # Prepare checkpoint data
            checkpoint_data = {
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1
            }
            
            # Save checkpoint to S3 models directory
            checkpoint_name = f'model_epoch_{epoch+1}_val_acc_{val_accuracy:.2f}.pt'
            s3_checkpoint_path = f's3://genvideo-dataset-complete/models/{checkpoint_name}'
            
            # Save to S3
            if save_checkpoint_to_s3(checkpoint_data, s3_checkpoint_path):
                print(f"Saved checkpoint to S3: {s3_checkpoint_path}")
            
            # Also save locally
            local_path = os.path.join(args.model_dir, checkpoint_name)
            torch.save(checkpoint_data, local_path)
            print(f"Saved checkpoint locally: {local_path}")
            
            # If this is the best model so far, also save to output directory
            final_model_path = 's3://genvideo-dataset-complete/output/best_model.pt'
            if save_checkpoint_to_s3(checkpoint_data, final_model_path):
                print(f"Saved best model to S3 output: {final_model_path}")
        # Generate and log confusion matrix and ROC curve
        plot_confusion_matrix(all_val_labels, all_val_preds, writer, epoch)
        
        # Update learning rate scheduler based on training loss
        scheduler.step(epoch_train_loss)
        
        # Save best model (handle DataParallel case)
        
    
    # At the end of training, sync tensorboard logs to S3
    sync_tensorboard_to_s3(os.path.join(args.output_dir, 'tensorboard'), tensorboard_s3_path)
    writer.close()

def sync_tensorboard_to_s3(local_dir, s3_path):
    """Sync tensorboard logs to S3"""
    try:
        s3_client = boto3.client('s3')
        s3_path = s3_path.replace('s3://', '')
        bucket = s3_path.split('/')[0]
        prefix = '/'.join(s3_path.split('/')[1:])
        
        # Upload all files in the tensorboard directory
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = os.path.join(prefix, relative_path)
                s3_client.upload_file(local_path, bucket, s3_key)
        print(f"Successfully synced tensorboard logs to {s3_path}")
    except Exception as e:
        print(f"Error syncing tensorboard logs to S3: {str(e)}")

def save_to_s3(local_path, s3_path):
    """Helper function to save a file to S3"""
    try:
        s3_client = boto3.client('s3')
        s3_path = s3_path.replace('s3://', '')
        bucket = s3_path.split('/')[0]
        key = '/'.join(s3_path.split('/')[1:])
        s3_client.upload_file(local_path, bucket, key)
        return True
    except Exception as e:
        print(f"Error saving to S3: {str(e)}")
        return False

def main():
    tensorboard_s3_path = 's3://genvideo-dataset-complete/tensorboard'
    args = parse_args()
    
    # Print hyperparameters to console
    print("\n=== Training Configuration ===")
    print(f"Initial Learning Rate: {args.learning_rate}")
    print(f"Warmup Epochs: {args.warmup_epochs}")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Batch Size (per GPU): {args.batch_size}")
    print(f"Total Batch Size ({num_gpus} GPUs): {args.batch_size * num_gpus}")
    print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
    print(f"Effective Batch Size: {args.batch_size * num_gpus * args.gradient_accumulation_steps}")
    print(f"Epochs: {args.epochs}")
    print(f"Target Size: {args.target_size}")
    print(f"Max Frames: {args.max_frames}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Model Directory: {args.model_dir}")
    print(f"Output Directory: {args.output_dir}")
    print("===========================\n")
    
    # Log hyperparameters to tensorboard
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    writer.add_hparams(
        {'lr': args.learning_rate, 
         'batch_size': args.batch_size,
         'epochs': args.epochs,
         'target_size': args.target_size,
         'max_frames': args.max_frames},
        {'dummy': 0}  # Required placeholder metric
    )
    sync_tensorboard_to_s3(os.path.join(args.output_dir, 'tensorboard'), tensorboard_s3_path)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using CUDA - Available devices: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'Device {i}: {torch.cuda.get_device_name(i)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    
    # Initialize model and training components
    s3_path = 's3://genvideo-dataset-complete/dataset/'
    dataset = VideoDataset(s3_path, target_size=args.target_size, 
                         max_frames=args.max_frames)
    
    train_size = int(0.8* len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Adjust batch size per GPU
    effective_batch_size = args.batch_size * torch.cuda.device_count()
    print(f'Effective batch size with {torch.cuda.device_count()} GPUs: {effective_batch_size}')
    
    train_loader = DataLoader(train_dataset, batch_size=effective_batch_size,
                            shuffle=True, collate_fn=custom_collate_fn,
                            num_workers=4 * torch.cuda.device_count(),
                            pin_memory=True)  # Enable pinned memory for faster GPU transfer
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size,
                          shuffle=False, collate_fn=custom_collate_fn,
                          num_workers=4 * torch.cuda.device_count(),
                          pin_memory=True)  # Enable pinned memory for faster GPU transfer
    
    latent_encoder = FullLatentEncoder()
    patch_encoder = FullPatchEncoder()
    classifier = FullClassifier()
    model = FullVideoClassifier(latent_encoder, patch_encoder, classifier)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        # Set device_ids explicitly to ensure proper GPU utilization
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    
    model = model.to(device)
    
    # Use a larger learning rate since we have multiple GPUs
    base_lr = args.learning_rate
    effective_lr = base_lr * torch.cuda.device_count()
    print(f"Scaling learning rate from {base_lr} to {effective_lr} for {torch.cuda.device_count()} GPUs")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=effective_lr)
    criterion = nn.CrossEntropyLoss()
    
    train(model, train_loader, val_loader, criterion, optimizer, device, args)

if __name__ == '__main__':
    main()
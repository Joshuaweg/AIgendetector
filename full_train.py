import os, gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from full_scale_classifier import *
from dataset import VideoDataset, custom_collate_fn, SizeBatchSampler
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configuration flags
LOCAL_TESTING = True  # Set to False for deployment

# Training configurations based on environment
if LOCAL_TESTING:
    BATCH_SIZE =2
    EPOCHS = 1
    NUM_WORKERS = 0
    PIN_MEMORY = True
    PREFETCH_FACTOR = None
    PERSISTENT_WORKERS = False
    TEST_MODE = True  # New flag for testing one batch
else:
    # AWS optimized settings
    BATCH_SIZE = 64  # Can be increased based on GPU memory
    EPOCHS = 10
    NUM_WORKERS = 4  # Use multiple workers for data loading
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True
    TEST_MODE = False

# Memory management settings
if LOCAL_TESTING:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
else:
    # AWS optimized memory settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024,expandable_segments:True,roundup_power2:True'

def clear_gpu_memory():
    """Function to thoroughly clear GPU memory"""
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

if __name__ == "__main__":
    # Enable anomaly detection only in LOCAL_TESTING
    if LOCAL_TESTING:
        torch.autograd.set_detect_anomaly(True)
    
    clear_gpu_memory()
    if torch.cuda.is_available():
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"Initial GPU memory cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading model")
    latentEncoder = FullLatentEncoder().to(device)
    patchEncoder = FullPatchEncoder().to(device)
    classifier = FullClassifier().to(device)
    vclf = FullVideoClassifier(latentEncoder,patchEncoder,classifier).to(device)
    
    # Enable gradient checkpointing for transformer
    if hasattr(classifier.transformer_encoder, 'layers'):
        for layer in classifier.transformer_encoder.layers:
            layer.checkpoint = True
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    optimizer = torch.optim.Adam(vclf.parameters(), lr=0.0005, eps=1e-8)
    
    # Load the videos with max size of 512
    real_videos_path = 'data\\many\\real'
    fake_videos_path = 'data\\many\\fake'
    print("Loading dataset")
    dataset = VideoDataset(real_videos_path, fake_videos_path, max_frames=24)
    print(f"Dataset length: {len(dataset)}")
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], 
        generator=torch.Generator().manual_seed(314159)
    )
    
    # Use standard DataLoader with random sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=custom_collate_fn,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=custom_collate_fn,
        pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR if NUM_WORKERS > 0 else None,
        persistent_workers=PERSISTENT_WORKERS if NUM_WORKERS > 0 else False
    )
    
    # Save test paths
    with open('data\\test_paths.txt', 'w') as f:
        for video in test_dataset:
            f.write(video[2] + '\n')
    
    print("Begin Training")
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.empty_cache()
    clear_gpu_memory()
    
    if torch.cuda.is_available():
        print(f"GPU memory before training: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"GPU memory cached before training: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    
    # Initialize TensorBoard writer
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'runs/training_{run_name}')
    
    # Create directories for additional visualizations
    os.makedirs('conference_plots', exist_ok=True)
    
    # Log model graph and architecture summary
    dummy_input = torch.zeros((1, 16, 256, 256, 3), device=device)
    writer.add_graph(vclf, dummy_input)
    
    # Log model architecture details
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_stats = {
        'Total Parameters': count_parameters(vclf),
        'Latent Encoder Parameters': count_parameters(latentEncoder),
        'Patch Encoder Parameters': count_parameters(patchEncoder),
        'Classifier Parameters': count_parameters(classifier)
    }
    
    for name, count in model_stats.items():
        writer.add_text('Model Architecture', f'{name}: {count:,}', 0)
    
    # Training metrics storage
    best_accuracy = 0.0
    training_metrics = {
        'losses': [], 'accuracies': [], 
        'val_losses': [], 'val_accuracies': [],
        'learning_rates': [],
        'batch_times': [],
        'epoch_times': []
    }
    
    # Memory management settings
    memory_clear_frequency = 5  # Clear memory every 5 batches
    
    def log_epoch_metrics(epoch, metrics_dict):
        """Create and save detailed epoch metrics plots"""
        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(metrics_dict['losses'], label='Training Loss')
        plt.plot(metrics_dict['val_losses'], label='Validation Loss')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(metrics_dict['accuracies'], label='Training Accuracy')
        plt.plot(metrics_dict['val_accuracies'], label='Validation Accuracy')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(metrics_dict['learning_rates'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        
        plt.subplot(2, 2, 4)
        plt.plot(metrics_dict['epoch_times'])
        plt.title('Training Time per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Time (minutes)')
        
        plt.tight_layout()
        writer.add_figure('Training Metrics', plt.gcf(), epoch)
        plt.savefig(f'conference_plots/training_metrics_epoch_{epoch}.png')
        plt.close()
    
    def analyze_predictions(val_preds, val_labels, epoch):
        """Analyze prediction confidence and error patterns"""
        # Confidence distribution
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(val_preds, bins=20)
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        
        # Error analysis
        errors = [i for i in range(len(val_labels)) if val_preds[i] != val_labels[i]]
        if errors:
            plt.subplot(1, 2, 2)
            error_conf = [val_preds[i] for i in errors]
            sns.histplot(error_conf, bins=20)
            plt.title('Confidence Distribution of Errors')
            plt.xlabel('Confidence Score')
            plt.ylabel('Count')
        
        plt.tight_layout()
        writer.add_figure('Prediction Analysis', plt.gcf(), epoch)
        plt.savefig(f'conference_plots/prediction_analysis_epoch_{epoch}.png')
        plt.close()
    
    # Add per-class metrics tracking
    class_metrics = {
        'fake': {'precision': [], 'recall': [], 'f1': []},
        'real': {'precision': [], 'recall': [], 'f1': []}
    }
    
    def log_class_metrics(val_preds, val_labels, epoch):
        """Calculate and log per-class metrics"""
        from sklearn.metrics import precision_recall_fscore_support
        
        # Ensure we have both classes represented in labels parameter
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, 
            val_preds, 
            labels=[0, 1],  # Explicitly specify both fake (0) and real (1)
            zero_division=0  # Handle cases where a class isn't present
        )
        
        for i, class_name in enumerate(['fake', 'real']):
            class_metrics[class_name]['precision'].append(precision[i])
            class_metrics[class_name]['recall'].append(recall[i])
            class_metrics[class_name]['f1'].append(f1[i])
            
            writer.add_scalar(f'Metrics/{class_name}_precision', precision[i], epoch)
            writer.add_scalar(f'Metrics/{class_name}_recall', recall[i], epoch)
            writer.add_scalar(f'Metrics/{class_name}_f1', f1[i], epoch)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                         factor=0.5, patience=2, 
                                                         verbose=True)
    
    for epoch in range(EPOCHS):
        start_epoch_time = time.time()
        classes = ['fake', 'real']
        vclf.train()  # Explicitly set train mode
        running_loss = 0.0
        batch_times = []
        epoch_predictions = []
        epoch_labels = []
        
        for batch_idx, (data, labels, paths) in enumerate(train_loader):
            batch_start_time = time.time()
            
            # Track memory before batch
            before_mem = torch.cuda.memory_allocated(device)/1e9
            print("batch_size: ", len(data))
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            with torch.amp.autocast("cuda"):
                outputs = vclf(data)
                pred = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels)
                
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"NaN or Inf in loss at Epoch {epoch}, Batch {batch_idx}")
                continue
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(vclf.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            epoch_predictions.extend(pred.cpu().tolist())
            epoch_labels.extend(labels.cpu().tolist())
            
            # Store output shape before cleanup if in test mode
            if TEST_MODE:
                output_shape = outputs.shape
                output_device = outputs.device
            
            # Efficient cleanup after each batch
            del outputs
            torch.cuda.empty_cache()
            
            # Track memory after batch and timing
            after_mem = torch.cuda.memory_allocated(device)/1e9
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Log per-batch metrics to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Training/Batch_Loss', loss.item(), global_step)
            writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('System/GPU_Memory_Usage', after_mem, global_step)
            writer.add_scalar('System/Batch_Processing_Time', batch_time, global_step)
                
            if batch_idx % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                avg_batch_time = sum(batch_times[-50:]) / len(batch_times[-50:]) if batch_times else 0
                
                # Convert predictions and labels to lists for logging
                preds = pred.cpu().tolist()
                true_labels = labels.cpu().tolist()
                pred_classes = [classes[p] for p in preds]
                true_classes = [classes[l] for l in true_labels]
                
                print(f"Epoch {epoch}, Batch {batch_idx} "
                      f"Avg Loss: {avg_loss:.2f} "
                      f"predictions: {pred_classes} targets: {true_classes} "
                      f"GPU Memory: {after_mem:.2f}GB "
                      f"Memory Change: {(after_mem - before_mem):.3f}GB "
                      f"Batch Time: {batch_time:.2f}s "
                      f"Avg Time: {avg_batch_time:.2f}s")
            
            # Break after one batch in test mode
            if TEST_MODE and batch_idx == 0:
                print("\nTest mode: Stopping after one batch")
                break
            
            # Cleanup large tensors
            if batch_idx % memory_clear_frequency == 0:
                del data, labels
                torch.cuda.empty_cache()
                gc.collect()
        
        # Log epoch metrics
        epoch_loss = running_loss / len(train_loader)
        writer.add_scalar('Training/Epoch_Loss', epoch_loss, epoch)
        
        # Evaluation phase
        vclf.eval()
        val_predictions = []
        val_labels = []
        val_confidence_scores = []
        accuracy = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (data, labels, paths) in enumerate(test_loader):
                data = data.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = vclf(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate confidence scores before deletion
                confidence = torch.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
                
                # Store predictions, labels, and confidence scores
                val_predictions.extend(predicted.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())
                val_confidence_scores.extend(confidence.cpu().numpy())
                
                # Print detailed validation information in test mode
                if TEST_MODE:
                    print("\nValidation Batch Information:")
                    print(f"Input shape: {data.shape}")
                    print(f"Labels shape: {labels.shape}")
                    print(f"Output shape: {outputs.shape}")
                    print(f"Predictions: {predicted.cpu().tolist()}")
                    print(f"True labels: {labels.cpu().tolist()}")
                    print(f"Validation loss: {loss.item():.4f}")
                    print(f"Batch accuracy: {(predicted == labels).sum().item() / labels.size(0) * 100:.2f}%")
                
                del data, labels, outputs, confidence
                
                # Break after one batch in test mode
                if TEST_MODE and batch_idx == 0:
                    print("\nTest mode: Stopping validation after one batch")
                    break
        
        # Skip metrics if no validation data
        if len(val_predictions) == 0:
            print("No validation data available")
            continue
            
        # Calculate and log validation metrics
        epoch_val_loss = val_loss / (batch_idx + 1)  # Use actual number of batches
        accuracy_pct = (accuracy/total)*100
        
        # Calculate elapsed time for this epoch
        end_epoch_time = time.time()
        elapsed_time = end_epoch_time - start_epoch_time
        
        # Update metrics dictionary
        training_metrics['losses'].append(epoch_loss)
        training_metrics['val_losses'].append(epoch_val_loss)
        training_metrics['accuracies'].append(accuracy_pct)
        training_metrics['val_accuracies'].append(accuracy_pct)
        training_metrics['learning_rates'].append(optimizer.param_groups[0]['lr'])
        training_metrics['epoch_times'].append(elapsed_time / 60)  # Convert to minutes
        
        # Log metrics to TensorBoard
        writer.add_scalar('Training/Epoch_Loss', epoch_loss, epoch)
        writer.add_scalar('Validation/Loss', epoch_val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', accuracy_pct, epoch)
        
        # Calculate and log per-class metrics
        if len(val_predictions) > 1:  # Need at least 2 samples for metrics
            log_class_metrics(val_predictions, val_labels, epoch)
            
            # Create prediction confidence analysis
            val_confidence_scores = np.array(val_confidence_scores)
            analyze_predictions(val_confidence_scores[:, 1], val_labels, epoch)
        
            # Create and save epoch metrics plots
            log_epoch_metrics(epoch, training_metrics)
            
            # Generate ROC curve for this epoch
            fpr, tpr, _ = roc_curve(val_labels, val_confidence_scores[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(10, 10))
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Epoch {epoch}')
            plt.legend(loc="lower right")
            writer.add_figure('ROC_Curves/Epoch', plt.gcf(), epoch)
            plt.savefig(f'conference_plots/roc_curve_epoch_{epoch}.png')
            plt.close()
            
            # Generate confusion matrix for this epoch
            cm = confusion_matrix(val_labels, val_predictions)
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=classes, yticklabels=classes)
            plt.title(f'Confusion Matrix (Counts) - Epoch {epoch}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            plt.subplot(1, 2, 2)
            sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlBu',
                       xticklabels=classes, yticklabels=classes)
            plt.title(f'Confusion Matrix (Percentages) - Epoch {epoch}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            plt.tight_layout()
            writer.add_figure('Confusion_Matrices/Epoch', plt.gcf(), epoch)
            plt.savefig(f'conference_plots/confusion_matrices_epoch_{epoch}.png')
            plt.close()
            
            # Log AUC score for this epoch
            writer.add_scalar('Validation/AUC', roc_auc, epoch)
        
        # Update learning rate scheduler
        scheduler.step(accuracy_pct)
        
        print(f"Epoch {epoch} - Test accuracy: {accuracy_pct:.3f}% - Avg Val Loss: {epoch_val_loss:.3f}")
        
        # Save model if validation improves
        if accuracy_pct > best_accuracy:
            best_accuracy = accuracy_pct
            torch.save(vclf.state_dict(), f'model/videoClassifier_full_best.pth')
        torch.save(vclf.state_dict(), f'model/videoClassifier_full_epoch_{epoch}.pth')
        
        end_epoch_time = time.time()
        elapsed_time = end_epoch_time - start_epoch_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Epoch {epoch} took {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
    
    writer.close()
    print("Training completed. TensorBoard logs saved in 'runs' directory.")

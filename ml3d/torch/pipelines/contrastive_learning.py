"""
Contrastive learning pipeline for PointContrast-style self-supervised learning.

Implements InfoNCE loss with correspondence-based positive pairs.
Supports both RandLANet and KPConv encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .base_pipeline import BasePipeline
from ...utils import make_dir

log = logging.getLogger(__name__)


class SimpleContrastiveLoss(nn.Module):
    """Simplified contrastive loss for cloud-level embeddings (SimCLR-style).
    
    Each cloud produces one embedding. Positive pairs are two augmented views
    of the same cloud. All other clouds in the batch are negatives.
    """
    
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings1, embeddings2):
        """Compute contrastive loss between two views.
        
        Args:
            embeddings1: (B, D) L2-normalized embeddings from view 1
            embeddings2: (B, D) L2-normalized embeddings from view 2
                            
        Returns:
            loss: Scalar contrastive loss
            metrics: Dict with accuracy
        """
        batch_size = embeddings1.shape[0]
        device = embeddings1.device
        
        # Compute similarity matrix: (B, B)
        # sim[i, j] = cosine similarity between embeddings1[i] and embeddings2[j]
        similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / self.temperature
        
        # Labels: embeddings1[i] should match with embeddings2[i]
        labels = torch.arange(batch_size, device=device)
        
        # Contrastive loss in both directions
        loss_12 = F.cross_entropy(similarity_matrix, labels)
        loss_21 = F.cross_entropy(similarity_matrix.T, labels)
        loss = (loss_12 + loss_21) / 2.0
        
        # Accuracy: how often is the correct pair the highest similarity?
        predictions = similarity_matrix.argmax(dim=1)
        accuracy = (predictions == labels).float().mean()
        
        metrics = {
            'accuracy': accuracy.item(),
            'loss_12': loss_12.item(),
            'loss_21': loss_21.item(),
        }
        
        return loss, metrics


class ContrastiveLearning(BasePipeline):
    """Pipeline for contrastive self-supervised learning on point clouds.
    
    Implements PointContrast-style training with InfoNCE loss.
    Works with HarvardForest3DContrastive dataset and contrastive models.
    """
    
    def __init__(
        self,
        model,
        dataset=None,
        name='ContrastiveLearning',
        main_log_dir='./logs',
        device='cuda',
        # Training parameters
        learning_rate=1e-3,
        weight_decay=1e-4,
        scheduler_gamma=0.99,
        batch_size=8,
        val_batch_size=8,
        test_batch_size=8,
        max_epoch=100,
        save_ckpt_freq=10,
        # Contrastive parameters
        temperature=0.07,
        freeze_encoder_epochs=5,
        # Logging
        train_sum_dir='contrastive_train',
        **kwargs
    ):
        """Initialize contrastive learning pipeline.
        
        Args:
            model: Contrastive model (RandLANetContrast or KPConvContrast)
            dataset: HarvardForest3DContrastive dataset
            temperature: Temperature for InfoNCE loss
            freeze_encoder_epochs: Freeze encoder for first N epochs
            learning_rate: Base learning rate (for projection head)
            scheduler_gamma: Exponential LR decay factor
        """
        super().__init__(
            model=model,
            dataset=dataset,
            name=name,
            main_log_dir=main_log_dir,
            device=device,
            **kwargs
        )
        
        # Add our custom config parameters to the existing self.cfg from BasePipeline
        # Don't overwrite it!
        self.cfg.learning_rate = learning_rate
        self.cfg.weight_decay = weight_decay
        self.cfg.scheduler_gamma = scheduler_gamma
        self.cfg.batch_size = batch_size
        self.cfg.val_batch_size = val_batch_size
        self.cfg.test_batch_size = test_batch_size
        self.cfg.max_epoch = max_epoch
        self.cfg.save_ckpt_freq = save_ckpt_freq
        self.cfg.temperature = temperature
        self.cfg.freeze_encoder_epochs = freeze_encoder_epochs
        self.cfg.train_sum_dir = train_sum_dir
        
        # Loss function
        self.criterion = SimpleContrastiveLoss(temperature=temperature)
        
        # Optimizer (with separate LRs for encoder and projection head)
        self.optimizer = model.get_optimizer(self.cfg)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=scheduler_gamma
        )
        
        # Logging
        self.current_epoch = 0
        self.train_sum_dir = Path(main_log_dir) / train_sum_dir
        make_dir(str(self.train_sum_dir))
        
        # TensorBoard writer
        self.summary_writer = SummaryWriter(log_dir=str(self.train_sum_dir))
        log.info(f"TensorBoard logs will be saved to: {self.train_sum_dir}")
        
        log.info(f"ContrastiveLearning pipeline initialized:")
        log.info(f"  Temperature: {temperature}")
        log.info(f"  Freeze encoder epochs: {freeze_encoder_epochs}")
        log.info(f"  Base LR: {learning_rate}")
        log.info(f"  Max epochs: {max_epoch}")

    def run_train(self):
        """Main training loop."""
        log.info("=" * 50)
        log.info("Starting contrastive learning training")
        log.info("=" * 50)
        
        # Get train and validation dataloaders
        train_dataset = self.dataset.get_split('train')
        val_dataset = self.dataset.get_split('val')
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_fn,
            drop_last=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.cfg.val_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate_fn,
            drop_last=False
        )
        
        log.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.cfg.max_epoch):
            self.current_epoch = epoch
            
            # Unfreeze encoder after warmup
            if epoch == self.cfg.freeze_encoder_epochs and self.model.encoder_frozen:
                self.model.unfreeze_encoder()
                # Recreate optimizer with updated parameters
                self.optimizer = self.model.get_optimizer(self.cfg)
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.cfg.scheduler_gamma
                )
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Learning rate step
            self.scheduler.step()
            
            # Log metrics
            log.info(
                f"Epoch {epoch+1}/{self.cfg.max_epoch} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
            )
            
            # TensorBoard logging
            self.summary_writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            self.summary_writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            self.summary_writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            self.summary_writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            self.summary_writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.cfg.save_ckpt_freq == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, is_best=True)
                log.info(f"  *** New best validation loss: {best_val_loss:.4f} ***")
        
        # Close TensorBoard writer
        self.summary_writer.close()
        
        log.info("=" * 50)
        log.info("Training completed!")
        log.info(f"Best validation loss: {best_val_loss:.4f}")
        log.info("=" * 50)

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            epoch: Current epoch number
            
        Returns:
            metrics: Dict with loss and accuracy
        """
        self.model.train()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")
        
        for batch in pbar:
            # Move batch to device
            batch = self.to_device(batch, self.device)
            
            # Forward pass for both views
            outputs1 = self.model({'point': batch['point_view1']})
            outputs2 = self.model({'point': batch['point_view2']})
            
            embeddings1 = outputs1['embeddings']  # (B, D)
            embeddings2 = outputs2['embeddings']  # (B, D)
            
            # Compute contrastive loss (cloud-level)
            loss, metrics = self.criterion(embeddings1, embeddings2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_acc += metrics['accuracy']
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{metrics['accuracy']:.4f}"
            })
        
        avg_metrics = {
            'loss': total_loss / num_batches,
            'accuracy': total_acc / num_batches,
        }
        
        return avg_metrics

    def validate_epoch(self, dataloader, epoch):
        """Validate for one epoch.
        
        Args:
            dataloader: Validation dataloader
            epoch: Current epoch number
            
        Returns:
            metrics: Dict with loss and accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Val Epoch {epoch+1}")
            
            for batch in pbar:
                # Move batch to device
                batch = self.to_device(batch, self.device)
                
                # Forward pass for both views
                outputs1 = self.model({'point': batch['point_view1']})
                outputs2 = self.model({'point': batch['point_view2']})
                
                embeddings1 = outputs1['embeddings']
                embeddings2 = outputs2['embeddings']
                
                # Compute contrastive loss (cloud-level)
                loss, metrics = self.criterion(embeddings1, embeddings2)
                
                # Accumulate metrics
                total_loss += loss.item()
                total_acc += metrics['accuracy']
                num_batches += 1
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{metrics['accuracy']:.4f}"
                })
        
        avg_metrics = {
            'loss': total_loss / num_batches,
            'accuracy': total_acc / num_batches,
        }
        
        return avg_metrics

    def collate_fn(self, batch):
        """Collate function for dataloader.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Batched dict
        """
        # Stack point clouds
        point_view1 = torch.stack([torch.from_numpy(b['point_view1']) for b in batch])
        point_view2 = torch.stack([torch.from_numpy(b['point_view2']) for b in batch])
        
        result = {
            'point_view1': point_view1,
            'point_view2': point_view2,
            'names': [b['name'] for b in batch],
        }
        
        return result

    def to_device(self, batch, device):
        """Move batch to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                result[key] = [v.to(device) for v in value]
            else:
                result[key] = value
        return result

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: If True, save as best model
        """
        checkpoint_dir = Path(self.cfg.main_log_dir) / 'checkpoint'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }
        
        if is_best:
            save_path = checkpoint_dir / 'ckpt_best.pth'
        else:
            save_path = checkpoint_dir / f'ckpt_epoch_{epoch+1:03d}.pth'
        
        torch.save(checkpoint, save_path)
        log.info(f"Checkpoint saved: {save_path}")

    def run_test(self):
        """Test is not applicable for contrastive learning."""
        log.warning("Test mode is not applicable for contrastive learning.")
        log.info("Use the trained encoder for downstream tasks (segmentation, classification).")

    def run_inference(self, data):
        """Inference is not applicable for contrastive learning."""
        raise NotImplementedError("Inference not applicable for contrastive learning.")

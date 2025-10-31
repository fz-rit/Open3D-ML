import logging
from os.path import join
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from .base_pipeline import BasePipeline
from ..dataloaders import get_sampler, TorchDataloader, DefaultBatcher, ConcatBatcher
from ..utils import latest_torch_ckpt
from ...utils import make_dir, PIPELINE, get_runid
from ...datasets import InferenceDummySplit

log = logging.getLogger(__name__)


class SSLRotation(BasePipeline):
    """Self-Supervised Learning pipeline for rotation prediction task.
    
    This pipeline trains a model to predict discrete rotations applied to point clouds.
    Useful for learning representations without labeled data.
    
    **Example:**
        train_pipeline = SSLRotation(
            model=model,
            dataset=dataset,
            name='HarvardForestSSL',
            batch_size=4,
            max_epoch=50,
            learning_rate=1e-3,
            num_rotation_classes=4,
            rotation_axis='z',
            main_log_dir='./logs/',
            device='cuda'
        )
        train_pipeline.run_train()
    
    **Args:**
        model: The SSL model (e.g., RandLANetSSL).
        dataset: The dataset (e.g., HarvardForest3D).
        num_rotation_classes: Number of discrete rotation angles (e.g., 4 for 0/90/180/270).
        rotation_axis: Axis to rotate around ('z' for vertical in forest scenes).
        freeze_encoder_epochs: Number of epochs to freeze encoder (warmup).
    """

    def __init__(
        self,
        model,
        dataset=None,
        name='SSLRotation',
        batch_size=4,
        val_batch_size=4,
        test_batch_size=3,
        max_epoch=100,
        learning_rate=1e-3,
        encoder_lr=None,  # If None, use learning_rate * 0.1
        lr_decays=0.95,
        save_ckpt_freq=5,
        scheduler_gamma=0.95,
        weight_decay=1e-4,
        main_log_dir='./logs/',
        device='cuda',
        train_sum_dir='train_log',
        num_rotation_classes=4,
        rotation_axis='z',
        freeze_encoder_epochs=0,
        **kwargs
    ):
        super().__init__(
            model=model,
            dataset=dataset,
            name=name,
            batch_size=batch_size,
            val_batch_size=val_batch_size,
            test_batch_size=test_batch_size,
            max_epoch=max_epoch,
            learning_rate=learning_rate,
            lr_decays=lr_decays,
            save_ckpt_freq=save_ckpt_freq,
            scheduler_gamma=scheduler_gamma,
            main_log_dir=main_log_dir,
            device=device,
            train_sum_dir=train_sum_dir,
            **kwargs
        )
        
        self.num_rotation_classes = num_rotation_classes
        self.rotation_axis = rotation_axis.lower()
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.encoder_lr = encoder_lr if encoder_lr is not None else learning_rate * 0.1
        self.weight_decay = weight_decay
        
        # Compute rotation angles in radians
        self.rotation_angles = [
            2 * np.pi * i / num_rotation_classes for i in range(num_rotation_classes)
        ]
        
        log.info(f"SSL Rotation pipeline initialized with {num_rotation_classes} classes")
        log.info(f"Rotation angles (degrees): {[np.degrees(a) for a in self.rotation_angles]}")

    def apply_rotation(self, points, rotation_class):
        """Apply discrete rotation to point cloud.
        
        Args:
            points: numpy array (N, 3) or (N, d) where first 3 dims are XYZ
            rotation_class: int, rotation class index
            
        Returns:
            rotated_points: numpy array with same shape as input
        """
        angle = self.rotation_angles[rotation_class]
        
        if self.rotation_axis == 'z':
            # Rotate around Z axis
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=np.float32)
        elif self.rotation_axis == 'y':
            # Rotate around Y axis
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=np.float32)
        elif self.rotation_axis == 'x':
            # Rotate around X axis
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported rotation axis: {self.rotation_axis}")
        
        # Apply rotation to XYZ coordinates
        rotated_points = points.copy()
        rotated_points[:, :3] = points[:, :3] @ rotation_matrix.T
        
        return rotated_points

    def _collate_fn(self, batch):
        """Custom collate function that handles rotation labels properly.
        
        Args:
            batch: List of {'data': inputs_dict, 'attr': attr_dict}
            
        Returns:
            Batched data with preserved attr list
        """
        from ..dataloaders import DefaultBatcher
        
        # Separate data and attr
        data_list = [item['data'] for item in batch]
        attr_list = [item['attr'] for item in batch]
        
        # Use default batcher for data only
        batcher = DefaultBatcher()
        batched_data = batcher.collate_fn(data_list)
        
        # Return with attr list preserved
        return {'data': batched_data, 'attr': attr_list}

    def get_batcher(self, device, split='training'):
        """Get the batcher to be used based on the device and split."""
        batcher_name = getattr(self.model.cfg, 'batcher', 'DefaultBatcher')

        if batcher_name == 'DefaultBatcher':
            batcher = DefaultBatcher()
        elif batcher_name == 'ConcatBatcher':
            batcher = ConcatBatcher(device, self.model.cfg.name)
        else:
            batcher = DefaultBatcher()  # Fallback
        return batcher

    def run_train(self):
        """Run SSL rotation training."""
        torch.manual_seed(self.rng.integers(np.iinfo(np.int32).max))
        
        model = self.model
        device = self.device
        model.device = device
        dataset = self.dataset
        cfg = self.cfg
        
        model.to(device)
        
        log.info("="*50)
        log.info("SSL ROTATION TRAINING")
        log.info(f"DEVICE: {device}")
        log.info(f"Num rotation classes: {self.num_rotation_classes}")
        log.info(f"Rotation axis: {self.rotation_axis}")
        log.info(f"Freeze encoder epochs: {self.freeze_encoder_epochs}")
        log.info("="*50)
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file_path = join(cfg.logs_dir, 'log_train_' + timestamp + '.txt')
        log.info(f"Logging in file: {log_file_path}")
        log.addHandler(logging.FileHandler(log_file_path))
        
        # Setup loss and metrics
        criterion = nn.CrossEntropyLoss()
        
        # Setup data loaders
        batcher = self.get_batcher(device)
        
        train_dataset = dataset.get_split('train')
        train_sampler = train_dataset.sampler
        train_split = TorchDataloader(
            dataset=train_dataset,
            preprocess=model.preprocess,
            transform=self._ssl_transform,
            sampler=train_sampler,
            use_cache=dataset.cfg.use_cache
        )
        train_loader = DataLoader(
            train_split,
            batch_size=cfg.batch_size,
            sampler=get_sampler(train_sampler),
            collate_fn=self._collate_fn,  # Use custom collate function
            num_workers=cfg.get('num_workers', 0),
            pin_memory=True
        )
        
        val_dataset = dataset.get_split('val')
        val_sampler = val_dataset.sampler
        val_split = TorchDataloader(
            dataset=val_dataset,
            preprocess=model.preprocess,
            transform=self._ssl_transform,
            sampler=val_sampler,
            use_cache=dataset.cfg.use_cache
        )
        val_loader = DataLoader(
            val_split,
            batch_size=cfg.val_batch_size,
            sampler=get_sampler(val_sampler),
            collate_fn=self._collate_fn,  # Use custom collate function
            num_workers=cfg.get('num_workers', 0),
            pin_memory=True
        )
        
        # Setup optimizer and scheduler
        if hasattr(model, 'get_optimizer'):
            optimizer = model.get_optimizer(cfg)
        else:
            optimizer = torch.optim.AdamW([
                {'params': model.parameters(), 'lr': cfg.learning_rate}
            ], weight_decay=self.weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.scheduler_gamma
        )
        
        # Setup tensorboard
        tensorboard_dir = join(
            cfg.train_sum_dir,
            model.__class__.__name__ + '_' + dataset.name + '_torch'
        )
        runid = get_runid(tensorboard_dir)
        tensorboard_dir = join(tensorboard_dir, runid)
        writer = SummaryWriter(tensorboard_dir)
        self.tensorboard_dir = tensorboard_dir
        
        log.info(f"Writing summary to {tensorboard_dir}")
        
        # Load checkpoint if exists
        start_epoch = 0
        best_val_acc = 0.0
        ckpt_dir = join(cfg.logs_dir, 'checkpoint')
        make_dir(ckpt_dir)
        
        latest_ckpt = latest_torch_ckpt(ckpt_dir)
        if latest_ckpt is not None:
            log.info(f"Loading checkpoint from {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            log.info(f"Resuming from epoch {start_epoch}")
        
        # Training loop
        for epoch in range(start_epoch, cfg.max_epoch):
            log.info(f"\n{'='*50}")
            log.info(f"EPOCH {epoch + 1}/{cfg.max_epoch}")
            log.info(f"{'='*50}")
            
            # Update model epoch for conditional freezing
            if hasattr(model, 'set_epoch'):
                model.set_epoch(epoch)
            
            # Train
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # Validate
            val_loss, val_acc = self._validate_epoch(
                model, val_loader, criterion, device, epoch
            )
            
            # Learning rate schedule
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Logging
            log.info(f"\nEpoch {epoch + 1} Summary:")
            log.info(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            log.info(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            log.info(f"  Learning Rate: {current_lr:.6f}")
            
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/accuracy', train_acc, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/accuracy', val_acc, epoch)
            writer.add_scalar('learning_rate', current_lr, epoch)
            
            # Save checkpoint
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                log.info(f"  New best validation accuracy: {best_val_acc:.2f}%")
            
            if (epoch + 1) % cfg.save_ckpt_freq == 0 or is_best:
                ckpt_path = join(ckpt_dir, f'ckpt_epoch_{epoch + 1:03d}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'best_val_acc': best_val_acc,
                }, ckpt_path)
                log.info(f"  Saved checkpoint to {ckpt_path}")
                
                if is_best:
                    best_path = join(ckpt_dir, 'ckpt_best.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'best_val_acc': best_val_acc,
                    }, best_path)
                    log.info(f"  Saved best checkpoint to {best_path}")
        
        writer.close()
        log.info("\n" + "="*50)
        log.info("TRAINING COMPLETED")
        log.info(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        log.info("="*50)

    def _ssl_transform(self, data, attr):
        """Apply SSL rotation augmentation AND point sampling to preprocessed data.
        
        This combines:
        1. Rotation augmentation for SSL task
        2. Point sampling and KNN (from RandLANet transform logic)
        
        Args:
            data: Preprocessed data dict with 'point', 'feat', 'label', 'search_tree'
            attr: Attribute dict
            
        Returns:
            inputs: Dict ready for model forward pass
        """
        from ...datasets.utils import DataProcessing
        import torch
        
        cfg = self.model.cfg
        
        # Sample a random rotation class
        rotation_class = self.rng.integers(0, self.num_rotation_classes)
        
        # Apply rotation to points BEFORE sampling
        pc = data['point'].copy()
        pc = self.apply_rotation(pc, rotation_class)
        
        # Get features
        feat = data['feat'].copy() if data['feat'] is not None else None
        tree = data['search_tree']
        
        # Sample points (from RandLANet transform logic)
        if pc.shape[0] > cfg.num_points:
            selected_idxs = self.rng.choice(pc.shape[0], cfg.num_points, replace=False)
        else:
            selected_idxs = np.arange(pc.shape[0])
        
        pc = pc[selected_idxs]
        if feat is not None:
            feat = feat[selected_idxs]
        
        # Concatenate features with coordinates
        if feat is None:
            feat = pc.copy()
        else:
            feat = np.concatenate([pc, feat], axis=1)
        
        # Build inputs for RandLANet forward pass
        input_points = []
        input_neighbors = []
        input_pools = []
        
        for i in range(cfg.num_layers):
            # KNN search
            neighbour_idx = DataProcessing.knn_search(pc, pc, cfg.num_neighbors)
            
            # Subsampling
            sub_points = pc[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:pc.shape[0] // cfg.sub_sampling_ratio[i], :]
            
            input_points.append(pc)
            input_neighbors.append(neighbour_idx.astype(np.int64))
            input_pools.append(pool_i.astype(np.int64))
            pc = sub_points
        
        # Prepare inputs dict
        inputs = {
            'coords': input_points,
            'neighbor_indices': input_neighbors,
            'sub_idx': input_pools,
            'features': feat
        }
        
        # Store rotation label in attr (preserved through batching)
        attr['rotation_label'] = rotation_class
        
        return inputs

    def _train_epoch(self, model, train_loader, criterion, optimizer, device, epoch):
        """Train for one epoch."""
        model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}")
        
        for batch_idx, inputs in enumerate(pbar):
            # Move data to device
            if hasattr(inputs['data'], 'to'):
                for key in inputs['data']:
                    if torch.is_tensor(inputs['data'][key]):
                        inputs['data'][key] = inputs['data'][key].to(device)
            
            # Extract rotation labels from attr list
            rotation_labels = torch.tensor(
                [attr_item['rotation_label'] for attr_item in inputs['attr']],
                dtype=torch.long,
                device=device
            )
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(inputs['data'])
            loss = criterion(logits, rotation_labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Compute accuracy
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == rotation_labels).sum().item()
            
            total_loss += loss.item() * rotation_labels.size(0)
            total_correct += correct
            total_samples += rotation_labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / rotation_labels.size(0):.2f}%'
            })
        
        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        
        return avg_loss, avg_acc

    def _validate_epoch(self, model, val_loader, criterion, device, epoch):
        """Validate for one epoch."""
        model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(val_loader, desc=f"Val Epoch {epoch + 1}")
        
        with torch.no_grad():
            for batch_idx, inputs in enumerate(pbar):
                # Move data to device
                if hasattr(inputs['data'], 'to'):
                    for key in inputs['data']:
                        if torch.is_tensor(inputs['data'][key]):
                            inputs['data'][key] = inputs['data'][key].to(device)
                
                # Extract rotation labels from attr list
                rotation_labels = torch.tensor(
                    [attr_item['rotation_label'] for attr_item in inputs['attr']],
                    dtype=torch.long,
                    device=device
                )
                
                # Forward pass
                logits = model(inputs['data'])
                loss = criterion(logits, rotation_labels)
                
                # Compute accuracy
                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == rotation_labels).sum().item()
                
                total_loss += loss.item() * rotation_labels.size(0)
                total_correct += correct
                total_samples += rotation_labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.0 * correct / rotation_labels.size(0):.2f}%'
                })
        
        avg_loss = total_loss / total_samples
        avg_acc = 100.0 * total_correct / total_samples
        
        return avg_loss, avg_acc

    def run_test(self):
        """Run testing (validation accuracy)."""
        model = self.model
        device = self.device
        dataset = self.dataset
        cfg = self.cfg
        
        model.to(device)
        model.eval()
        
        log.info("="*50)
        log.info("SSL ROTATION TESTING")
        log.info(f"DEVICE: {device}")
        log.info("="*50)
        
        # Load best checkpoint
        ckpt_dir = join(cfg.logs_dir, 'checkpoint')
        best_ckpt = join(ckpt_dir, 'ckpt_best.pth')
        
        if Path(best_ckpt).exists():
            log.info(f"Loading best checkpoint from {best_ckpt}")
            checkpoint = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            log.warning("No best checkpoint found, using current model weights")
        
        # Setup data loader
        batcher = self.get_batcher(device)
        
        test_dataset = dataset.get_split('val')  # Use val as test
        test_sampler = test_dataset.sampler
        test_split = TorchDataloader(
            dataset=test_dataset,
            preprocess=model.preprocess,
            transform=self._ssl_transform,
            sampler=test_sampler,
            use_cache=dataset.cfg.use_cache
        )
        test_loader = DataLoader(
            test_split,
            batch_size=cfg.test_batch_size,
            sampler=get_sampler(test_sampler),
            collate_fn=batcher.collate_fn
        )
        
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = self._validate_epoch(model, test_loader, criterion, device, 0)
        
        log.info(f"\nTest Results:")
        log.info(f"  Test Loss: {test_loss:.4f}")
        log.info(f"  Test Accuracy: {test_acc:.2f}%")
        log.info("="*50)

    def run_inference(self, data):
        """Run inference on a single data sample.
        
        Args:
            data: A dict with 'point' and optionally 'feat' keys.
            
        Returns:
            Predicted rotation class (0, 1, 2, or 3).
        """
        model = self.model
        device = self.device
        
        model.to(device)
        model.eval()
        
        # Preprocess
        attr = {'split': 'test'}
        processed_data = model.preprocess(data, attr)
        
        # Transform (apply model's transform, not SSL rotation augmentation)
        from ...datasets.utils import DataProcessing
        
        pc = processed_data['point'].copy()
        feat = processed_data['feat'].copy() if processed_data['feat'] is not None else None
        tree = processed_data['search_tree']
        
        # Point sampling
        if pc.shape[0] > self.cfg.num_points:
            selected_idxs = np.random.choice(pc.shape[0], self.cfg.num_points, replace=False)
        else:
            selected_idxs = np.arange(pc.shape[0])
        
        pc = pc[selected_idxs]
        if feat is not None:
            feat = feat[selected_idxs]
        
        # Concatenate features
        if feat is None:
            feat = pc.copy()
        else:
            feat = np.concatenate([pc, feat], axis=1)
        
        # Prepare inputs (similar to transform but without rotation)
        input_points = []
        input_neighbors = []
        input_pools = []
        
        for i in range(model.cfg.num_layers):
            neighbour_idx = DataProcessing.knn_search(pc, pc, model.cfg.num_neighbors)
            sub_points = pc[:pc.shape[0] // model.cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:pc.shape[0] // model.cfg.sub_sampling_ratio[i], :]
            
            input_points.append(pc)
            input_neighbors.append(neighbour_idx.astype(np.int64))
            input_pools.append(pool_i.astype(np.int64))
            pc = sub_points
        
        # Convert to tensors and add batch dimension
        inputs = {
            'coords': [torch.from_numpy(p).unsqueeze(0).float().to(device) for p in input_points],
            'neighbor_indices': [torch.from_numpy(n).unsqueeze(0).long().to(device) for n in input_neighbors],
            'sub_idx': [torch.from_numpy(s).unsqueeze(0).long().to(device) for s in input_pools],
            'features': torch.from_numpy(feat).unsqueeze(0).float().to(device)
        }
        
        # Forward pass
        with torch.no_grad():
            logits = model(inputs)
            pred_class = torch.argmax(logits, dim=1).cpu().item()
        
        return pred_class


PIPELINE._register_module(SSLRotation)

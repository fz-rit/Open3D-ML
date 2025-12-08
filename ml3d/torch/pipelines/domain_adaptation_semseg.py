"""
Domain Adaptation Pipeline for Semantic Segmentation using CORAL.

This pipeline extends SemanticSegmentation to support unsupervised domain adaptation
using Correlation Alignment (CORAL) loss as described in SqueezeSegV2.
"""

import logging
from os.path import exists, join
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from open3d.visualization.tensorboard_plugin import summary
from .semantic_segmentation import SemanticSegmentation
from .domain_adaptation_monitoring import DomainAdaptationMonitor
from .domain_adaptation_trainer import DomainAdaptationTrainer
from ..dataloaders import get_sampler, TorchDataloader, DefaultBatcher, ConcatBatcher
from ..utils import latest_torch_ckpt
from ..modules.losses import SemSegLoss, filter_valid_label, MultiLayerCORALLoss, AdaptiveCORALLoss
from ..modules.metrics import SemSegMetric
from ...utils import make_dir, PIPELINE, get_runid, code2md
from ...datasets import InferenceDummySplit

log = logging.getLogger(__name__)


class DomainAdaptationSemanticSegmentation(SemanticSegmentation):
    """Domain adaptation pipeline for semantic segmentation.
    
    This pipeline trains a model on a labeled source domain while adapting it
    to an unlabeled target domain using CORAL (Correlation Alignment).
    
    Training scheme:
    - Source domain (labeled): supervised segmentation loss
    - Target domain (unlabeled): unsupervised CORAL loss for feature alignment
    - Progressive domain calibration: gradually increase CORAL weight during training
    
    **Example:**
        pipeline = DomainAdaptationSemanticSegmentation(
            model=model,
            source_dataset=semantic3d_dataset,
            target_dataset=forest_dataset,
            coral_weight=0.1,
            progressive_steps=5000,
            alignment_layers=[0, 2, 4],
            layer_weights=[0.3, 0.4, 0.3],
            use_geodesic=True,
            batch_size=8,
            max_epoch=100
        )
        pipeline.run_train()
    
    **Args:**
        model: Model with domain adaptation support (e.g., RandLANetDA, KPFCNNDA)
        source_dataset: Labeled source domain dataset
        target_dataset: Unlabeled target domain dataset (labels only for validation)
        coral_weight: Maximum weight for CORAL loss (default: 0.1)
        progressive_steps: Steps to ramp up CORAL weight from 0 to coral_weight
        alignment_layers: Which model layers to align (default: model's alignment_layers)
        layer_weights: Weights for each alignment layer (default: equal weights)
        use_geodesic: Use geodesic distance for CORAL (default: True)
        coral_loss_type: 'multi_layer' or 'adaptive' (default: 'adaptive')
        Other args same as SemanticSegmentation
    """

    def __init__(
            self,
            model,
            source_dataset=None,
            target_dataset=None,
            dataset=None,  # For compatibility, uses source_dataset if provided
            name='DomainAdaptationSemanticSegmentation',
            batch_size=4,
            val_batch_size=4,
            test_batch_size=3,
            max_epoch=100,
            learning_rate=1e-2,
            lr_decays=0.95,
            save_ckpt_freq=20,
            adam_lr=1e-2,
            scheduler_gamma=0.95,
            momentum=0.98,
            main_log_dir='./logs/',
            device='cuda',
            split='train',
            train_sum_dir='train_log',
            # Domain adaptation specific parameters
            coral_weight=0.1,
            progressive_steps=5000,
            alignment_layers=None,
            layer_weights=None,
            use_geodesic=True,
            coral_loss_type='adaptive',
            **kwargs):

        # Set source dataset as primary dataset for compatibility
        if source_dataset is None and dataset is not None:
            source_dataset = dataset
        
        if source_dataset is None:
            raise ValueError("source_dataset must be provided for domain adaptation")
        if target_dataset is None:
            raise ValueError("target_dataset must be provided for domain adaptation")

        # Initialize parent with source dataset
        super().__init__(model=model,
                         dataset=source_dataset,
                         name=name,
                         batch_size=batch_size,
                         val_batch_size=val_batch_size,
                         test_batch_size=test_batch_size,
                         max_epoch=max_epoch,
                         learning_rate=learning_rate,
                         lr_decays=lr_decays,
                         save_ckpt_freq=save_ckpt_freq,
                         adam_lr=adam_lr,
                         scheduler_gamma=scheduler_gamma,
                         momentum=momentum,
                         main_log_dir=main_log_dir,
                         device=device,
                         split=split,
                         train_sum_dir=train_sum_dir,
                         **kwargs)

        # Store domain adaptation parameters
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.coral_weight = coral_weight
        self.progressive_steps = progressive_steps
        self.alignment_layers = alignment_layers
        self.layer_weights = layer_weights
        self.use_geodesic = use_geodesic
        self.coral_loss_type = coral_loss_type
        # Number of target batches to evaluate during validation (0 disables target eval)
        self.num_target_validate_batch = self.cfg.get('num_target_validate_batch', 0)
        
        # Initialize CORAL loss
        if coral_loss_type == 'adaptive':
            min_weight_ratio = kwargs.get('min_weight_ratio', 0.1)
            self.coral_loss = AdaptiveCORALLoss(
                base_weight=coral_weight,
                ramp_up_steps=progressive_steps,
                use_geodesic=use_geodesic,
                min_weight_ratio=min_weight_ratio
            )
        else:  # multi_layer
            self.coral_loss = MultiLayerCORALLoss(
                layer_weights=layer_weights,
                use_geodesic=use_geodesic
            )
        
        # Initialize monitoring and training modules (created in run_train)
        self.monitor = None
        self.trainer = None
        
        # Ensure model supports feature extraction
        if not hasattr(model, 'forward') or 'return_features' not in str(model.forward.__code__.co_varnames):
            log.warning(
                f"Model {model.__class__.__name__} may not support feature extraction. "
                "Consider using RandLANetDA or KPFCNNDA for domain adaptation."
            )

    def run_train(self):
        """Run domain adaptation training."""
        torch.manual_seed(self.rng.integers(np.iinfo(np.int32).max))
        torch.autograd.set_detect_anomaly(True)
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available! This pipeline requires a CUDA-enabled GPU."
            )
        
        model = self.model
        model.device = self.device
        model.to(self.device)

        log.info(f"Domain Adaptation: Source={self.source_dataset.name}, Target={self.target_dataset.name}")
        log.info(f"CORAL weight: {self.coral_weight}, Progressive steps: {self.progressive_steps}")
        
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file_path = join(self.cfg.logs_dir, 'log_train_da_' + timestamp + '.txt')
        log.info(f"Logging in file: {log_file_path}")
        log.addHandler(logging.FileHandler(log_file_path))

        # Initialize losses and metrics
        Loss = SemSegLoss(self, model, self.source_dataset, self.device)
        self.metric_train = SemSegMetric()
        self.metric_val = SemSegMetric()

        self.batcher = self.get_batcher(self.device)

        # SOURCE DOMAIN (labeled) - for training
        source_train_dataset = self.source_dataset.get_split('train')
        source_train_sampler = source_train_dataset.sampler
        source_train_split = TorchDataloader(
            dataset=source_train_dataset,
            preprocess=model.preprocess,
            transform=model.transform,
            sampler=source_train_sampler,
            use_cache=self.source_dataset.cfg.use_cache,
            steps_per_epoch=self.source_dataset.cfg.get('steps_per_epoch_train', None)
        )

        source_train_loader = DataLoader(
            source_train_split,
            batch_size=self.cfg.batch_size,
            sampler=get_sampler(source_train_sampler),
            num_workers=self.cfg.get('num_workers', 0),
            pin_memory=self.cfg.get('pin_memory', False),
            collate_fn=self.batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed))
        )

        # SOURCE DOMAIN - validation
        source_valid_dataset = self.source_dataset.get_split('validation')
        source_valid_sampler = source_valid_dataset.sampler
        source_valid_split = TorchDataloader(
            dataset=source_valid_dataset,
            preprocess=model.preprocess,
            transform=model.transform,
            sampler=source_valid_sampler,
            use_cache=self.source_dataset.cfg.use_cache,
            steps_per_epoch=self.source_dataset.cfg.get('steps_per_epoch_valid', None)
        )

        source_valid_loader = DataLoader(
            source_valid_split,
            batch_size=self.cfg.val_batch_size,
            sampler=get_sampler(source_valid_sampler),
            num_workers=self.cfg.get('num_workers', 0),
            pin_memory=self.cfg.get('pin_memory', False),
            collate_fn=self.batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed))
        )

        # TARGET DOMAIN (unlabeled for training, but we have labels for validation)
        target_train_dataset = self.target_dataset.get_split('test')  # Use test split as unlabeled
        target_train_sampler = target_train_dataset.sampler
        target_train_split = TorchDataloader(
            dataset=target_train_dataset,
            preprocess=model.preprocess,
            transform=model.transform,
            sampler=target_train_sampler,
            use_cache=self.target_dataset.cfg.get('use_cache', False),
            steps_per_epoch=self.source_dataset.cfg.get('steps_per_epoch_train', None)
        )

        target_train_loader = DataLoader(
            target_train_split,
            batch_size=self.cfg.batch_size,
            sampler=get_sampler(target_train_sampler),
            num_workers=self.cfg.get('num_workers', 0),
            pin_memory=self.cfg.get('pin_memory', False),
            collate_fn=self.batcher.collate_fn,
            worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                torch.utils.data.get_worker_info().seed))
        )

        # TARGET DOMAIN - validation (optional monitoring)
        target_valid_loader = None
        if self.num_target_validate_batch > 0:
            target_valid_dataset = self.target_dataset.get_split('test')
            target_valid_sampler = target_valid_dataset.sampler

            # Limit target validation batches if requested
            target_steps = self.num_target_validate_batch

            target_valid_split = TorchDataloader(
                dataset=target_valid_dataset,
                preprocess=model.preprocess,
                transform=model.transform,
                sampler=target_valid_sampler,
                use_cache=self.target_dataset.cfg.get('use_cache', False),
                steps_per_epoch=target_steps
            )

            target_valid_loader = DataLoader(
                target_valid_split,
                batch_size=self.cfg.val_batch_size,
                sampler=get_sampler(target_valid_sampler),
                num_workers=self.cfg.get('num_workers', 0),
                pin_memory=self.cfg.get('pin_memory', False),
                collate_fn=self.batcher.collate_fn,
                worker_init_fn=lambda x: np.random.seed(x + np.uint32(
                    torch.utils.data.get_worker_info().seed))
            )

        self.optimizer, self.scheduler = model.get_optimizer(self.cfg)

        is_resume = model.cfg.get('is_resume', False)
        start_epoch = self.load_ckpt(model.cfg.ckpt_path, is_resume=is_resume)

        # Sanity check parameters/BN stats after loading checkpoint
        model.check_finite(reset_bn=True, raise_on_nan=True, prefix="post-load")

        # Setup tensorboard
        tensorboard_dir = join(
            self.cfg.train_sum_dir,
            model.__class__.__name__ + '_DA_' + 
            self.source_dataset.name + '_to_' + self.target_dataset.name + '_torch'
        )
        runid = get_runid(tensorboard_dir)
        self.tensorboard_dir = join(self.cfg.train_sum_dir,
                                    runid + '_' + Path(tensorboard_dir).name)
        
        # Setup monitoring directory (parallel to checkpoint dir)
        monitoring_dir = join(self.tensorboard_dir, 'domain_monitoring')
        make_dir(monitoring_dir)
        log.info(f"Domain adaptation monitoring outputs will be saved to: {monitoring_dir}")
        
        # Initialize monitoring and trainer modules
        self.monitor = DomainAdaptationMonitor(
            monitoring_dir=monitoring_dir,
            use_geodesic=self.use_geodesic,
            monitor_freq=self.cfg.get('monitor_freq', 5),
            enable_tsne=self.cfg.get('enable_tsne', True),
            enable_umap=self.cfg.get('enable_umap', True)
        )
        
        self.trainer = DomainAdaptationTrainer(
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=self.device,
            coral_loss=self.coral_loss,
            coral_loss_type=self.coral_loss_type,
            grad_clip_norm=model.cfg.get('grad_clip_norm', 1.0)
        )

        writer = SummaryWriter(self.tensorboard_dir)
        self.save_config(writer)
        log.info(f"Writing summary in {self.tensorboard_dir}")
        record_summary = self.cfg.get('summary', {}).get('record_for', [])

        log.info("Started domain adaptation training")

        for epoch in range(start_epoch, self.cfg.max_epoch + 1):
            log.info(f'=== EPOCH {epoch:d}/{self.cfg.max_epoch:d} ===')
            
            # Set point sampler
            model.trans_point_sampler = source_train_sampler.get_point_sampler()
            
            # --------------------- TRAINING ---------------------
            train_stats = self.trainer.train_epoch(
                source_loader=source_train_loader,
                target_loader=target_train_loader,
                Loss=Loss,
                metric_train=self.metric_train,
                epoch=epoch,
                record_summary=('train' in record_summary),
                summary_callback=self.get_3d_summary if 'train' in record_summary else None
            )
            
            if train_stats['summary']:
                self.summary['train'] = train_stats['summary']

            # --------------------- VALIDATION ---------------------
            model.trans_point_sampler = source_valid_sampler.get_point_sampler()
            
            log.info("Validating on SOURCE domain...")
            val_stats = self.trainer.validate_epoch(
                valid_loader=source_valid_loader,
                Loss=Loss,
                metric_val=self.metric_val,
                record_summary=('valid' in record_summary),
                summary_callback=self.get_3d_summary if 'valid' in record_summary else None
            )
            
            if val_stats['summary']:
                self.summary['valid'] = val_stats['summary']

            # Optional: Validate on target domain
            target_metric = None
            if target_valid_loader is not None:
                log.info("Validating on TARGET domain (monitoring only)...")
                target_metric = SemSegMetric()
                model.eval()
                with torch.no_grad():
                    for step, inputs in enumerate(target_valid_loader):
                        if hasattr(inputs['data'], 'to'):
                            inputs['data'].to(self.device)

                        results = model(inputs['data'], return_intermediate_features=False)
                        if isinstance(results, tuple):
                            results = results[0]
                        
                        loss, gt_labels, predict_scores = model.get_loss(
                            Loss, results, inputs, self.device
                        )

                        if predict_scores.size()[-1] > 0:
                            target_metric.update(predict_scores, gt_labels)

            # Save logs with domain adaptation metrics
            self.save_logs_da(writer, epoch, target_metric, 
                            train_stats['train_loss'], val_stats['val_loss'], 
                            train_stats['coral_loss'])
            
            # Domain adaptation monitoring (periodic)
            if self.monitor.should_monitor(epoch, self.cfg.max_epoch):
                # Avoid crashes when validation metrics are unavailable (all batches skipped)
                val_iou = self.metric_val.iou() if hasattr(self, 'metric_val') else None
                source_val_iou = val_iou[-1] if val_iou is not None else 0.0

                target_val_iou = 0.0
                if target_metric:
                    tgt_iou = target_metric.iou()
                    target_val_iou = tgt_iou[-1] if tgt_iou is not None else 0.0
                
                self.monitor.run_monitoring(
                    epoch=epoch,
                    source_loader=source_valid_loader,
                    target_loader=target_train_loader,
                    model=model,
                    device=self.device,
                    source_val_iou=source_val_iou,
                    target_val_iou=target_val_iou
                )

            # Label histograms
            def _hist_from_conf(conf):
                if conf is None:
                    return None
                counts = conf.sum(axis=1)
                return counts.astype(np.int64).tolist()

            train_hist = _hist_from_conf(self.metric_train.confusion_matrix)
            val_hist = _hist_from_conf(self.metric_val.confusion_matrix)
            if train_hist is not None:
                log.info(f"Label histogram (source train): {train_hist}")
            if val_hist is not None:
                log.info(f"Label histogram (source val): {val_hist}")

            # Save best checkpoint based on source validation IoU
            val_iou = self.metric_val.iou()
            if val_iou is not None:
                current_iou = val_iou[-1]
                if not hasattr(self, 'best_val_iou'):
                    self.best_val_iou = float('-inf')
                
                if current_iou > self.best_val_iou:
                    self.best_val_iou = current_iou
                    log.info(f"New best validation IoU: {current_iou:.4f} - Saving checkpoint")
                    self.save_ckpt(epoch, best=True)
                else:
                    log.info(f"Current IoU: {current_iou:.4f}, Best: {self.best_val_iou:.4f}")
            else:
                log.warning(f"Epoch {epoch}: Skipping best model check - no valid validation metrics")

            # Regular checkpoint saving
            if epoch % self.cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch)

        log.info("Finished domain adaptation training")
        writer.close()

    def save_logs_da(self, writer, epoch, target_metric=None, 
                    train_loss=None, val_loss=None, coral_loss=None):
        """Save logs including domain adaptation metrics."""
        # Use provided losses or fallback to NaN
        train_loss = train_loss if train_loss is not None else float('nan')
        val_loss = val_loss if val_loss is not None else float('nan')
        coral_loss = coral_loss if coral_loss is not None else float('nan')
        
        # Log losses (always available)
        loss_dict = {
            'Training loss': train_loss,
            'Validation loss': val_loss,
            'CORAL loss': coral_loss,
        }
        
        for key, val in loss_dict.items():
            writer.add_scalar(key, val, epoch)
        
        # Log CORAL weight schedule
        if self.coral_loss_type == 'adaptive':
            current_weight = self.coral_loss.get_current_weight()
            writer.add_scalar('CORAL weight', current_weight, epoch)
        
        log.info(f"Loss: {loss_dict}")
        
        # Standard metrics (handle None when all batches skipped)
        train_accs = self.metric_train.acc()
        val_accs = self.metric_val.acc()

        train_ious = self.metric_train.iou()
        val_ious = self.metric_val.iou()
        
        # Handle case where all batches were skipped (returns None)
        if train_accs is None or val_accs is None or train_ious is None or val_ious is None:
            log.warning(f"Epoch {epoch}: Metrics unavailable - all batches had ignored labels")
            log.warning("Loss values are still logged, but accuracy/IoU metrics cannot be computed")
            return
        # Compute accuracy and IoU dicts
        acc_dicts = [{
            'Training accuracy': acc,
            'Validation accuracy': val_acc
        } for acc, val_acc in zip(train_accs, val_accs)]

        iou_dicts = [{
            'Training IoU': iou,
            'Validation IoU': val_iou
        } for iou, val_iou in zip(train_ious, val_ious)]

        # Add target domain metrics if available
        if target_metric is not None:
            target_accs = target_metric.acc()
            target_ious = target_metric.iou()
            
            if target_accs is not None and target_ious is not None:
                for i, (acc_dict, iou_dict) in enumerate(zip(acc_dicts, iou_dicts)):
                    if i < len(target_accs):
                        acc_dict['Target accuracy'] = target_accs[i]
                    if i < len(target_ious):
                        iou_dict['Target IoU'] = target_ious[i]

        for key, val in acc_dicts[-1].items():
            writer.add_scalar("acc/" + key, val, epoch)
        for key, val in iou_dicts[-1].items():
            writer.add_scalar("iou/" + key, val, epoch)

        log.info(f"Mean acc: {acc_dicts[-1]}")
        log.info(f"Mean IoU: {iou_dicts[-1]}")

    def run_test(self):
        """Run testing on target domain."""
        # Override dataset to use target for testing
        original_dataset = self.dataset
        self.dataset = self.target_dataset
        
        log.info(f"Testing on TARGET domain: {self.target_dataset.name}")
        
        # Call parent's run_test
        super().run_test()
        
        # Restore original dataset
        self.dataset = original_dataset


PIPELINE._register_module(DomainAdaptationSemanticSegmentation, 'torch')

"""
Domain Adaptation Training Module.

Handles the core training loop for domain adaptation with CORAL loss.
"""

import logging
import numpy as np
import torch
from tqdm import tqdm

log = logging.getLogger(__name__)


class DomainAdaptationTrainer:
    """Handles domain adaptation training logic."""
    
    def __init__(self, model, optimizer, scheduler, device, 
                 coral_loss, coral_loss_type='adaptive',
                 grad_clip_norm=1.0):
        """
        Initialize domain adaptation trainer.
        
        Args:
            model: Model instance
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler
            device: Device to use
            coral_loss: CORAL loss instance
            coral_loss_type: 'adaptive' or 'multi_layer'
            grad_clip_norm: Gradient clipping norm (0 to disable)
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.coral_loss = coral_loss
        self.coral_loss_type = coral_loss_type
        self.grad_clip_norm = grad_clip_norm
        self.global_step = 0
        self.nan_debug_logs = 0
        
        # Statistics
        self.losses = []
        self.coral_losses = []
        self.skipped_batches = 0
        self.nan_debug_logs = 0
        
        # EMA-based adaptive weighting
        self.seg_loss_ema = None  # Will be initialized on first batch
        self.coral_loss_ema = None
        self.ema_decay = 0.9  # Smoothing factor for moving average
        self.adaptive_weights = []  # Track weight history

        # Track batchnorm layers for temporary freezing during target forward
        self._bn_layers = [m for m in self.model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        
        # Debug: Track label histograms per batch
        self.batch_label_histograms = []
    
    def train_epoch(self, source_loader, target_loader, Loss, metric_train, 
                    epoch, record_summary=False, summary_callback=None):
        """
        Train for one epoch.
        
        Args:
            source_loader: Source domain dataloader (labeled)
            target_loader: Target domain dataloader (unlabeled)
            Loss: Loss function instance
            metric_train: Training metrics tracker
            epoch: Current epoch number
            record_summary: Whether to record summary for tensorboard
            summary_callback: Callback to get 3D summary
        
        Returns:
            dict: Training statistics
        """
        self.model.train()
        metric_train.reset()
        self.losses = []
        self.coral_losses = []
        self.skipped_batches = 0
        self.batch_label_histograms = []
        
        # Create dual iterator for source and target batches
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        # Determine number of iterations per epoch
        num_iterations = max(len(source_loader), len(target_loader))
        
        pbar = tqdm(range(num_iterations), desc=f'Epoch {epoch} training')
        summary = None
        
        for step in pbar:
            source_inputs = next(source_iter)
            target_inputs = next(target_iter)            
            # source_inputs['data'] = source_inputs['data'].to(self.device)
            # target_inputs['data'] = target_inputs['data'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass on SOURCE domain (with labels)
            source_results = self.model(source_inputs['data'], return_intermediate_features=True)
            source_logits, source_features = source_results

            # Compute supervised segmentation loss on source
            seg_loss, gt_labels, predict_scores = self.model.get_loss(
                Loss, source_logits, source_inputs, self.device
            )
            
            # Debug: Record label histogram for this batch
            if gt_labels.numel() > 0:
                unique_labels, label_counts = torch.unique(gt_labels, return_counts=True)
                histogram = {int(label): int(count) for label, count in zip(unique_labels, label_counts)}
                self.batch_label_histograms.append({
                    'batch': step,
                    'histogram': histogram,
                    'total_points': int(gt_labels.numel()),
                    'skipped': torch.isnan(seg_loss) or torch.isinf(seg_loss)
                })
            
            # Skip batch if it contains only ignored labels (NaN/Inf loss)
            if torch.isnan(seg_loss) or torch.isinf(seg_loss):
                # if self.nan_debug_logs < 3:
                #     unique_labels = torch.unique(gt_labels).tolist() if gt_labels.numel() > 0 else []
                #     log.warning(
                #         f"Skipping training batch {step} - all ignored labels; seg_loss={'NaN' if torch.isnan(seg_loss) else 'Inf'}; "
                #         f"unique labels: {unique_labels}"
                #     )
                #     self.nan_debug_logs += 1
                # else:
                #     log.debug(f"Skipping training batch {step} - all ignored labels (seg_loss={'NaN' if torch.isnan(seg_loss) else 'Inf'})")
                # self.skipped_batches += 1
                # continue
                raise RuntimeError(self._build_nan_error_message(
                    epoch, step, seg_loss, torch.tensor(0.0), seg_loss,
                    source_logits, gt_labels, source_inputs
                ))

            if predict_scores.size()[-1] > 0:
                metric_train.update(predict_scores, gt_labels)

            # Forward pass on TARGET domain (no labels, only features)
            # Freeze BN stats for target to avoid polluting running mean/var
            self._set_bn_eval()
            with torch.no_grad():
                target_results = self.model(target_inputs['data'], return_intermediate_features=True)
            
            self._set_bn_train()
            if isinstance(target_results, tuple):
                target_logits, target_features = target_results
            else:
                target_features = []
            # Compute CORAL loss for domain alignment
            coral_loss_value = self._compute_coral_loss(
                source_features, target_features, 
                epoch=epoch, step=step
            )

            # EMA-based adaptive weighting
            # Update moving averages
            seg_magnitude = seg_loss.detach()
            coral_magnitude = coral_loss_value.detach()
            
            if self.seg_loss_ema is None:
                # Initialize EMAs on first batch
                self.seg_loss_ema = seg_magnitude
                self.coral_loss_ema = coral_magnitude
            else:
                # Update EMAs with exponential smoothing
                self.seg_loss_ema = self.ema_decay * self.seg_loss_ema + (1 - self.ema_decay) * seg_magnitude
                self.coral_loss_ema = self.ema_decay * self.coral_loss_ema + (1 - self.ema_decay) * coral_magnitude
            
            # Calculate adaptive weight to match scales
            # Clamp to [1, 100] to prevent extreme values
            adaptive_weight = self.seg_loss_ema / (self.coral_loss_ema + 1e-8)
            adaptive_weight = torch.clamp(adaptive_weight, min=0.001, max=20.0)
            self.adaptive_weights.append(adaptive_weight.cpu().item())
            
            # Combined loss with adaptive weighting
            total_loss = seg_loss + adaptive_weight * coral_loss_value

            # Skip batch if CORAL loss caused NaN/Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                raise RuntimeError(self._build_nan_error_message(
                    epoch, step, seg_loss, coral_loss_value, total_loss,
                    source_logits, gt_labels, source_inputs
                ))

            # Backward and optimize
            total_loss.backward()
            
            # Clip gradients to prevent explosion
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            
            self.optimizer.step()

            # Fail fast if parameters became non-finite after the update
            self.model.check_finite(reset_bn=False, raise_on_nan=True,
                                    prefix=f"after_step_{self.global_step}")

            # Update step counters
            self.global_step += 1
            if self.coral_loss_type == 'adaptive':
                self.coral_loss.step()

            # Record losses
            self.losses.append(seg_loss.cpu().item())
            self.coral_losses.append(coral_loss_value.cpu().item())
            
            # Update progress bar
            pbar.set_postfix({
                'seg_loss': f'{seg_loss.item():.4f}',
                'coral_loss': f'{coral_loss_value.item():.4f}',
                'adaptive_w': f'{adaptive_weight.item():.2f}'
            })

            # Save summary for first batch
            if record_summary and step == 0 and summary_callback:
                summary = summary_callback(source_logits, source_inputs['data'], epoch)
        
        # Log info about skipped batches
        if self.skipped_batches > 0:
            skip_ratio = self.skipped_batches / num_iterations
            log.info(f"Epoch {epoch}: Skipped {self.skipped_batches}/{num_iterations} batches ({skip_ratio*100:.1f}%) with all ignored labels")
            if skip_ratio > 0.5:
                log.warning(f"More than 50% of batches skipped! Consider checking your data for sufficient valid labels.")
        
        self.scheduler.step()
        
        # Calculate average adaptive weight for this epoch
        avg_adaptive_weight = np.mean(self.adaptive_weights) if self.adaptive_weights else float('nan')
        
        return {
            'summary': summary,
            'train_loss': np.mean(self.losses) if self.losses else float('nan'),
            'coral_loss': np.mean(self.coral_losses) if self.coral_losses else float('nan'),
            'adaptive_weight': avg_adaptive_weight,
        }

    def _set_bn_eval(self):
        """Temporarily set all batchnorm layers to eval to freeze running stats."""
        for bn in self._bn_layers:
            bn.training_state_was = bn.training
            bn.eval()

    def _set_bn_train(self):
        """Restore batchnorm layers to their previous training state."""
        for bn in self._bn_layers:
            prev = getattr(bn, 'training_state_was', True)
            if prev:
                bn.train()
    
    def _compute_coral_loss(self, source_features, target_features, epoch=0, step=0):
        """Compute CORAL loss with proper handling of different loss types."""
        coral_loss_value = torch.tensor(0.0, device=self.device)
        
        if len(source_features) == 0 or len(target_features) == 0:
            return coral_loss_value

        # Skip computation entirely if the current CORAL weight is zero
        if self.coral_loss_type == 'adaptive':
            current_weight = self.coral_loss.get_current_weight()
        else:
            current_weight = getattr(self.coral_loss, 'base_weight', 0.0)

        if current_weight == 0.0:
            return coral_loss_value

        # Fail fast on non-finite features before eigen decomposition
        def _assert_finite(feats, name):
            if not torch.isfinite(feats).all():
                raise RuntimeError(
                    f"Non-finite {name} features before CORAL at epoch {epoch}, step {step}"
                )

        for idx, (s_feat, t_feat) in enumerate(zip(source_features, target_features)):
            _assert_finite(s_feat, f"source[{idx}]")
            _assert_finite(t_feat, f"target[{idx}]")
        
        # Enable debug on first iteration of first epoch
        enable_coral_debug = (epoch == 0 and step == 0)
        
        if self.coral_loss_type == 'adaptive':
            # Adaptive CORAL with single or multi-layer features
            if len(source_features) == 1:
                coral_loss_value = self.coral_loss(source_features[0], target_features[0], debug=enable_coral_debug)
            else:
                # Average across layers for adaptive loss
                for i, (s_feat, t_feat) in enumerate(zip(source_features, target_features)):
                    if enable_coral_debug:
                        log.info(f"\n=== Adaptive CORAL Layer {i} ===")
                    layer_loss = self.coral_loss.coral_loss(s_feat, t_feat, debug=enable_coral_debug)
                    coral_loss_value += layer_loss
                coral_loss_value /= len(source_features)
                # Apply adaptive weight (inverted: starts high, ramps down)
                if self.coral_loss.ramp_up_steps > 0:
                    progress = min(1.0, self.global_step / self.coral_loss.ramp_up_steps)
                    adaptive_weight = self.coral_loss.base_weight * (1.0 - progress * (1.0 - self.coral_loss.min_weight_ratio))
                else:
                    adaptive_weight = self.coral_loss.base_weight
                if enable_coral_debug:
                    log.info(f"\nAdaptive CORAL Summary:")
                    log.info(f"  Raw averaged loss: {coral_loss_value:.10f}")
                    log.info(f"  Progress: {progress:.4f}")
                    log.info(f"  Adaptive weight (inverted): {adaptive_weight:.4f}")
                    log.info(f"  Final weighted loss: {(coral_loss_value * adaptive_weight):.10f}")
                coral_loss_value *= adaptive_weight
        else:
            # Multi-layer CORAL
            coral_loss_value = self.coral_loss(source_features, target_features, debug=enable_coral_debug)
        
        return coral_loss_value
    
    def _build_nan_error_message(self, epoch, step, seg_loss, coral_loss, total_loss,
                                 source_logits, gt_labels, source_inputs):
        """Build detailed error message for NaN/Inf detection."""
        error_msg = f"\n{'='*60}\n"
        error_msg += f"NaN/Inf DETECTED - Epoch {epoch}, Step {step}\n"
        error_msg += f"{'='*60}\n"
        error_msg += f"seg_loss: {seg_loss.item() if not torch.isnan(seg_loss) else 'NaN'}\n"
        error_msg += f"coral_loss: {coral_loss.item() if not torch.isnan(coral_loss) else 'NaN'}\n"
        error_msg += f"total_loss: {total_loss.item() if not torch.isnan(total_loss) else 'NaN'}\n"
        error_msg += f"\nSource logits stats:\n"
        error_msg += f"  Shape: {source_logits.shape}\n"
        error_msg += f"  Contains NaN: {torch.isnan(source_logits).any()}\n"
        error_msg += f"  Contains Inf: {torch.isinf(source_logits).any()}\n"
        error_msg += f"  Range: [{source_logits.min():.4f}, {source_logits.max():.4f}]\n"
        error_msg += f"\nLabels stats:\n"
        error_msg += f"  gt_labels numel: {gt_labels.numel()}\n"
        if gt_labels.numel() > 0:
            error_msg += f"  gt_labels unique: {torch.unique(gt_labels).tolist()}\n"
        error_msg += f"\nInput data stats:\n"
        if 'labels' in source_inputs['data']:
            raw_labels = source_inputs['data']['labels']
            error_msg += f"  Raw labels shape: {raw_labels.shape}\n"
            error_msg += f"  Raw labels unique: {torch.unique(raw_labels).tolist()}\n"
        error_msg += f"{'='*60}\n"
        return error_msg
    
    def validate_epoch(self, valid_loader, Loss, metric_val, record_summary=False, 
                      summary_callback=None):
        """
        Run validation for one epoch.
        
        Args:
            valid_loader: Validation dataloader
            Loss: Loss function instance
            metric_val: Validation metrics tracker
            record_summary: Whether to record summary
            summary_callback: Callback to get 3D summary
        
        Returns:
            dict: Validation statistics
        """
        self.model.eval()
        metric_val.reset()
        valid_losses = []
        valid_skipped = 0
        summary = None
        batch_label_histograms = []
        
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(valid_loader, desc='validation')):
                if hasattr(inputs['data'], 'to'):
                    inputs['data'].to(self.device)

                results = self.model(inputs['data'], return_intermediate_features=False)
                if isinstance(results, tuple):
                    results = results[0]
                
                loss, gt_labels, predict_scores = self.model.get_loss(
                    Loss, results, inputs, self.device
                )

                # Debug: Record label histogram for this batch
                if gt_labels.numel() > 0:
                    unique_labels, label_counts = torch.unique(gt_labels, return_counts=True)
                    histogram = {int(label): int(count) for label, count in zip(unique_labels, label_counts)}
                    batch_label_histograms.append({
                        'batch': step,
                        'histogram': histogram,
                        'total_points': int(gt_labels.numel()),
                        'skipped': torch.isnan(loss) or torch.isinf(loss)
                    })

                # Skip validation batches with NaN loss (all ignored labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    log.debug(f"Skipping validation batch {step} - all ignored labels")
                    valid_skipped += 1
                    continue

                if predict_scores.size()[-1] > 0:
                    metric_val.update(predict_scores, gt_labels)

                valid_losses.append(loss.cpu().item())
                
                if record_summary and step == 0 and summary_callback:
                    summary = summary_callback(results, inputs['data'], 0)
        
        # Log validation statistics
        if valid_skipped > 0:
            total_val_batches = len(valid_loader)
            skip_ratio = valid_skipped / total_val_batches if total_val_batches > 0 else 0
            log.warning(f"Validation: Skipped {valid_skipped}/{total_val_batches} batches ({skip_ratio*100:.1f}%) with all ignored labels")
            if skip_ratio > 0.8:
                log.error(f"WARNING: >80% validation batches skipped! Your validation set may have insufficient valid labels.")
        
        # # Log batch label histogram statistics for validation
        # if len(batch_label_histograms) > 0:
        #     log.info(f"\n{'='*60}")
        #     log.info(f"VALIDATION BATCH LABEL HISTOGRAM SUMMARY")
        #     log.info(f"{'='*60}")
            
        #     skipped_count = sum(1 for h in batch_label_histograms if h['skipped'])
        #     valid_count = len(batch_label_histograms) - skipped_count
            
        #     log.info(f"Total batches: {len(batch_label_histograms)} (Valid: {valid_count}, Skipped: {skipped_count})")
        #     log.info(f"\nAll validation batches:")
        #     for hist_data in batch_label_histograms:
        #         status = "SKIPPED" if hist_data['skipped'] else "OK"
        #         hist = hist_data['histogram']
        #         total = hist_data['total_points']
        #         percentages = {label: f"{count/total*100:.1f}%" for label, count in hist.items()}
        #         log.info(f"  Batch {hist_data['batch']:2d} [{status:7s}]: {hist} -> {percentages}")
            
        #     # Compute aggregate statistics
        #     label_sums = {}
        #     for hist_data in batch_label_histograms:
        #         if not hist_data['skipped']:  # Only count valid batches
        #             for label, count in hist_data['histogram'].items():
        #                 label_sums[label] = label_sums.get(label, 0) + count
            
        #     if label_sums:
        #         total_valid_points = sum(label_sums.values())
        #         log.info(f"\nAggregate statistics (valid batches only):")
        #         log.info(f"  Total points: {total_valid_points}")
        #         for label in sorted(label_sums.keys()):
        #             count = label_sums[label]
        #             pct = count / total_valid_points * 100
        #             log.info(f"  Label {label}: {count:8d} ({pct:5.2f}%)")
            
        #     log.info(f"{'='*60}\n")
        
        return {
            'summary': summary,
            'val_loss': np.mean(valid_losses) if valid_losses else float('nan'),
        }

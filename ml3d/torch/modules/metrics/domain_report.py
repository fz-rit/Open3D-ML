"""
HTML Report Generator for Domain Adaptation Monitoring.
Creates comprehensive visual reports of feature alignment quality.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import base64


def generate_html_report(report_data: Dict,
                        output_path: str,
                        epoch: Optional[int] = None) -> str:
    """
    Generate comprehensive HTML report for domain adaptation monitoring.
    
    Args:
        report_data: Dictionary containing:
            - metrics: Dict of scalar metrics
            - layer_metrics: Per-layer alignment metrics
            - plots: Dict of plot paths {name: path}
            - config: Training configuration
        output_path: Path to save HTML report
        epoch: Current epoch (for periodic reports)
    
    Returns:
        Path to saved report
    """
    
    # Extract data
    metrics = report_data.get('metrics', {})
    layer_metrics = report_data.get('layer_metrics', {})
    plots = report_data.get('plots', {})
    config = report_data.get('config', {})
    
    # Encode images as base64
    encoded_images = {}
    for name, path in plots.items():
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode('utf-8')
                encoded_images[name] = f"data:image/png;base64,{encoded}"
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Domain Adaptation Report{f' - Epoch {epoch}' if epoch else ''}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header .subtitle {{
            margin-top: 10px;
            font-size: 1.1em;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 40px;
        }}
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-unit {{
            font-size: 0.6em;
            color: #888;
        }}
        .plot-container {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .plot-title {{
            font-size: 1.3em;
            color: #555;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        .layer-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        .layer-table th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        .layer-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        .layer-table tr:hover {{
            background: #f5f7fa;
        }}
        .config-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .config-item {{
            margin: 8px 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .config-key {{
            color: #667eea;
            font-weight: bold;
        }}
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }}
        .status-good {{
            background: #d4edda;
            color: #155724;
        }}
        .status-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        .status-bad {{
            background: #f8d7da;
            color: #721c24;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            background: #f8f9fa;
            color: #666;
            font-size: 0.9em;
        }}
        .summary-box {{
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .summary-box h2 {{
            margin-top: 0;
            font-size: 1.5em;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Domain Adaptation Monitoring Report</h1>
            <div class="subtitle">
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                {f'| Epoch: {epoch}' if epoch else ''}
            </div>
        </div>
        
        <div class="content">
            <!-- Summary Section -->
            {_generate_summary_section(metrics)}
            
            <!-- Key Metrics Section -->
            <div class="section">
                <h2 class="section-title">📊 Key Alignment Metrics</h2>
                <div class="metrics-grid">
                    {_generate_metric_cards(metrics)}
                </div>
            </div>
            
            <!-- Visualizations Section -->
            {_generate_visualizations_section(encoded_images)}
            
            <!-- Layer-wise Metrics Section -->
            {_generate_layer_metrics_section(layer_metrics)}
            
            <!-- Configuration Section -->
            {_generate_config_section(config)}
        </div>
        
        <div class="footer">
            <p>Domain Adaptation Report | Open3D-ML | CORAL-based Unsupervised DA</p>
        </div>
    </div>
</body>
</html>
"""
    
    # Save HTML
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    return output_path


def _generate_summary_section(metrics: Dict) -> str:
    """Generate executive summary box."""
    mmd = metrics.get('mmd', 0)
    cov_dist = metrics.get('cov_distance', 0)
    source_iou = metrics.get('source_val_iou', 0)
    target_iou = metrics.get('target_val_iou', 0)
    
    # Determine status
    if mmd < 0.05 and cov_dist < 0.1:
        status = "Excellent"
        badge_class = "status-good"
    elif mmd < 0.15 and cov_dist < 0.3:
        status = "Good"
        badge_class = "status-good"
    elif mmd < 0.3:
        status = "Fair"
        badge_class = "status-warning"
    else:
        status = "Needs Improvement"
        badge_class = "status-bad"
    
    return f"""
    <div class="summary-box">
        <h2>📋 Executive Summary <span class="status-badge {badge_class}">{status}</span></h2>
        <div class="grid-2">
            <div>
                <p><strong>Domain Alignment:</strong> MMD = {mmd:.4f}, Cov Distance = {cov_dist:.4f}</p>
                <p><strong>Source Performance:</strong> Validation IoU = {source_iou:.2%}</p>
            </div>
            <div>
                <p><strong>Target Performance:</strong> Validation IoU = {target_iou:.2%}</p>
                <p><strong>Recommendation:</strong> {_get_recommendation(mmd, cov_dist, source_iou, target_iou)}</p>
            </div>
        </div>
    </div>
    """


def _get_recommendation(mmd: float, cov_dist: float, source_iou: float, target_iou: float) -> str:
    """Generate actionable recommendation."""
    if mmd > 0.3:
        return "Increase coral_weight or add more alignment_layers"
    elif source_iou < 0.5:
        return "Reduce coral_weight - source performance degrading"
    elif target_iou < source_iou * 0.6:
        return "Try different alignment_layers or increase progressive_steps"
    else:
        return "Training progressing well - continue monitoring"


def _generate_metric_cards(metrics: Dict) -> str:
    """Generate HTML for metric cards."""
    cards_html = ""
    
    metric_specs = [
        ('mmd', 'MMD (RBF Kernel)', '', 4),
        ('cov_distance', 'Covariance Distance', '', 4),
        ('a_distance', 'A-Distance', '', 4),
        ('source_val_iou', 'Source Val IoU', '%', 2),
        ('target_val_iou', 'Target Val IoU', '%', 2),
        ('gap_reduction', 'Gap Reduction', '%', 1),
    ]
    
    for key, label, unit, decimals in metric_specs:
        if key in metrics:
            value = metrics[key]
            if unit == '%':
                value *= 100
            
            cards_html += f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">
                    {value:.{decimals}f}
                    <span class="metric-unit">{unit}</span>
                </div>
            </div>
            """
    
    return cards_html


def _generate_visualizations_section(encoded_images: Dict) -> str:
    """Generate visualizations section HTML."""
    section_html = """
    <div class="section">
        <h2 class="section-title">📈 Feature Space Visualizations</h2>
    """
    
    plot_order = [
        ('tsne', 't-SNE Projection'),
        ('umap', 'UMAP Projection'),
        ('feature_distributions', 'Feature Distributions'),
        ('covariance_matrices', 'Covariance Matrices'),
        ('layer_alignment', 'Layer-wise Alignment Progress'),
        ('training_metrics', 'Training Metrics Over Time'),
    ]
    
    for key, title in plot_order:
        if key in encoded_images:
            section_html += f"""
            <div class="plot-container">
                <div class="plot-title">{title}</div>
                <img src="{encoded_images[key]}" alt="{title}">
            </div>
            """
    
    section_html += "</div>"
    return section_html


def _generate_layer_metrics_section(layer_metrics: Dict) -> str:
    """Generate layer-wise metrics table."""
    if not layer_metrics:
        return ""
    
    section_html = """
    <div class="section">
        <h2 class="section-title">🔍 Layer-wise Alignment Metrics</h2>
        <table class="layer-table">
            <thead>
                <tr>
                    <th>Layer</th>
                    <th>MMD (RBF)</th>
                    <th>MMD (Linear)</th>
                    <th>Cov Distance</th>
                    <th>Source Mean</th>
                    <th>Target Mean</th>
                    <th>Alignment Quality</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for layer_idx in sorted(layer_metrics.keys()):
        metrics = layer_metrics[layer_idx]
        mmd_rbf = metrics.get('mmd_rbf', 0)
        mmd_linear = metrics.get('mmd_linear', 0)
        cov_dist = metrics.get('cov_distance', 0)
        
        source_stats = metrics.get('source_stats', {})
        target_stats = metrics.get('target_stats', {})
        
        source_mean = source_stats.get('mean', 0)
        target_mean = target_stats.get('mean', 0)
        
        # Quality assessment
        if mmd_rbf < 0.05:
            quality = "✅ Excellent"
        elif mmd_rbf < 0.15:
            quality = "🟢 Good"
        elif mmd_rbf < 0.3:
            quality = "🟡 Fair"
        else:
            quality = "🔴 Poor"
        
        section_html += f"""
            <tr>
                <td><strong>Layer {layer_idx}</strong></td>
                <td>{mmd_rbf:.4f}</td>
                <td>{mmd_linear:.4f}</td>
                <td>{cov_dist:.4f}</td>
                <td>{source_mean:.4f}</td>
                <td>{target_mean:.4f}</td>
                <td>{quality}</td>
            </tr>
        """
    
    section_html += """
            </tbody>
        </table>
    </div>
    """
    
    return section_html


def _generate_config_section(config: Dict) -> str:
    """Generate configuration section."""
    if not config:
        return ""
    
    section_html = """
    <div class="section">
        <h2 class="section-title">⚙️ Training Configuration</h2>
        <div class="config-section">
    """
    
    key_configs = [
        'coral_weight',
        'progressive_steps',
        'alignment_layers',
        'layer_weights',
        'batch_size',
        'learning_rate',
        'max_epoch',
    ]
    
    for key in key_configs:
        if key in config:
            value = config[key]
            if isinstance(value, list):
                value = ', '.join(map(str, value))
            section_html += f"""
            <div class="config-item">
                <span class="config-key">{key}:</span> {value}
            </div>
            """
    
    section_html += """
        </div>
    </div>
    """
    
    return section_html


def save_metrics_json(metrics: Dict, output_path: str):
    """Save metrics as JSON for programmatic access."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

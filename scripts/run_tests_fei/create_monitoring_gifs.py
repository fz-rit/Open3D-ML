#!/usr/bin/env python3
"""
Create GIF animations from domain monitoring images across epochs.
This script generates GIFs for UMAP and t-SNE visualizations to observe
feature evolution during domain adaptation training.
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple
import glob

try:
    from PIL import Image
except ImportError:
    print("PIL (Pillow) is required. Install with: pip install Pillow")
    exit(1)


def natural_sort_key(s: str) -> List:
    """Sort strings containing numbers in natural order."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def collect_images(monitoring_dir: Path, image_name: str) -> List[Tuple[int, Path]]:
    """
    Collect all images with the given name from epoch directories.
    
    Args:
        monitoring_dir: Path to the domain_monitoring directory
        image_name: Name of the image file (e.g., 'umap.png')
    
    Returns:
        List of (epoch_number, image_path) tuples, sorted by epoch
    """
    images = []
    
    # Find all epoch directories
    epoch_dirs = glob.glob(str(monitoring_dir / "epoch_*"))
    
    for epoch_dir in epoch_dirs:
        epoch_dir_path = Path(epoch_dir)
        # Extract epoch number from directory name
        match = re.search(r'epoch_(\d+)', epoch_dir_path.name)
        if match:
            epoch_num = int(match.group(1))
            image_path = epoch_dir_path / image_name
            if image_path.exists():
                images.append((epoch_num, image_path))
    
    # Sort by epoch number
    images.sort(key=lambda x: x[0])
    
    return images


def create_gif(image_paths: List[Path], output_path: Path, duration: int = 500, 
               loop: int = 0, add_epoch_label: bool = True, 
               epoch_numbers: List[int] = None):
    """
    Create a GIF from a list of images.
    
    Args:
        image_paths: List of paths to image files
        output_path: Path where the GIF will be saved
        duration: Duration of each frame in milliseconds
        loop: Number of loops (0 = infinite)
        add_epoch_label: Whether to add epoch number to each frame
        epoch_numbers: List of epoch numbers corresponding to each image
    """
    if not image_paths:
        print(f"No images found for {output_path.name}")
        return
    
    frames = []
    durations = []
    
    for idx, img_path in enumerate(image_paths):
        try:
            img = Image.open(img_path)
            
            # Add epoch label if requested
            if add_epoch_label and epoch_numbers:
                from PIL import ImageDraw, ImageFont
                img = img.copy()
                draw = ImageDraw.Draw(img)
                
                # Try to use a nice font, fall back to default if not available
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
                except:
                    font = ImageFont.load_default()
                
                # Calculate progress (0.0 to 1.0)
                progress = idx / (len(image_paths) - 1) if len(image_paths) > 1 else 1.0
                
                # Color gradient from blue (start) to green (end)
                # Blue: (0, 100, 255), Green: (0, 200, 0)
                r = int(0)
                g = int(100 + (100 * progress))
                b = int(255 * (1 - progress))
                text_color = (r, g, b)
                
                # Add epoch label in top-left corner with background
                text = f"Epoch {epoch_numbers[idx]}"
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Draw semi-transparent background
                padding = 10
                label_x, label_y = 10, 10
                draw.rectangle(
                    [(label_x, label_y), (label_x + text_width + 2*padding, label_y + text_height + 2*padding)],
                    fill=(255, 255, 255, 230)
                )
                draw.text((label_x + padding, label_y + padding), text, fill=text_color, font=font)
                
                # Add progress bar right underneath the epoch label
                bar_height = 12
                bar_spacing = 5
                bar_y = label_y + text_height + 2*padding + bar_spacing
                bar_x = label_x
                bar_width = max(text_width + 2*padding, 200)  # At least as wide as the label or 200px
                
                # Draw progress bar background (gray)
                draw.rectangle(
                    [(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)],
                    fill=(200, 200, 200),
                    outline=(100, 100, 100),
                    width=2
                )
                
                # Draw progress bar fill (gradient color)
                if progress > 0:
                    fill_width = int(bar_width * progress)
                    draw.rectangle(
                        [(bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height)],
                        fill=text_color,
                        outline=None
                    )
            
            frames.append(img)
            
            # Set duration: first frame +2s, last frame +3s, others normal
            if idx == 0:
                durations.append(duration + 2000)  # +2 seconds
            elif idx == len(image_paths) - 1:
                durations.append(duration + 3000)  # +3 seconds
            else:
                durations.append(duration)
                
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    
    if frames:
        # Save as GIF with variable durations
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=loop,
            optimize=False
        )
        print(f"Created GIF: {output_path} ({len(frames)} frames)")
    else:
        print(f"No valid frames for {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Create GIF animations from domain monitoring images"
    )
    parser.add_argument(
        "monitoring_dir",
        type=str,
        help="Path to the domain_monitoring directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for GIF files (default: same as monitoring_dir)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=500,
        help="Duration of each frame in milliseconds (default: 500)"
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't add epoch labels to frames"
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=["umap.png", "tsne.png", "decoder_umap_classes.png", "decoder_tsne_classes.png"],
        help="List of image names to create GIFs for"
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=None,
        help="Maximum epoch to include (default: include all epochs)"
    )
    
    args = parser.parse_args()
    
    monitoring_dir = Path(args.monitoring_dir)
    if not monitoring_dir.exists():
        print(f"Error: Directory not found: {monitoring_dir}")
        return 1
    
    output_dir = Path(args.output_dir) if args.output_dir else monitoring_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing images from: {monitoring_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frame duration: {args.duration}ms")
    print()
    
    # Create GIFs for each requested image type
    for image_name in args.images:
        print(f"Processing {image_name}...")
        
        # Collect images across epochs
        epoch_images = collect_images(monitoring_dir, image_name)
        
        if not epoch_images:
            print(f"  No images found for {image_name}")
            continue
        
        # Filter by max epoch if specified
        if args.max_epoch is not None:
            epoch_images = [(epoch, path) for epoch, path in epoch_images if epoch <= args.max_epoch]
            if not epoch_images:
                print(f"  No images found for {image_name} with epoch <= {args.max_epoch}")
                continue
        
        print(f"  Found {len(epoch_images)} images (epochs {epoch_images[0][0]} to {epoch_images[-1][0]})")
        
        # Extract epoch numbers and paths
        epoch_numbers = [epoch_num for epoch_num, _ in epoch_images]
        image_paths = [img_path for _, img_path in epoch_images]
        
        # Create output filename
        output_name = image_name.replace('.png', '_evolution.gif')
        output_path = output_dir / output_name
        
        # Create GIF
        create_gif(
            image_paths,
            output_path,
            duration=args.duration,
            add_epoch_label=not args.no_labels,
            epoch_numbers=epoch_numbers
        )
        print()
    
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())

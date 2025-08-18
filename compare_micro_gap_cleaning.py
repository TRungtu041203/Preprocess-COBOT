#!/usr/bin/env python3
"""
Compare before/after results of micro-gap cleaning
Shows side-by-side visualization of original vs cleaned skeleton data
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Tuple

# Try to import the advanced visualizer
try:
    from Cobot_preprocess.visualize_skeleton_advanced import plot_skeleton_frame_2d, plot_skeleton_frame_3d
    VIZ_AVAILABLE = True
    print("✅ Advanced visualizer available")
except ImportError:
    VIZ_AVAILABLE = False
    print("⚠️  Advanced visualizer not available, using basic plots")


def plot_frame_basic(ax, data: np.ndarray, frame_idx: int, title: str = ""):
    """Basic 2D plot for comparison with bone connections"""
    ax.clear()
    
    if frame_idx >= data.shape[0]:
        ax.text(0.5, 0.5, f"Frame {frame_idx} OOR", ha='center', va='center', transform=ax.transAxes)
        return
    
    frame = data[frame_idx]
    
    # Define skeleton connections for COBOT dataset (48 joints) - MediaPipe palm structure
    SKELETON_CONNECTIONS = [
        # Arms
        [42, 44], [44, 46], [43, 45], [45, 47], [42, 43],
        
        # Right hand (indices 0-20)
        [0, 1], [1, 2], [2, 3], [3, 4],     # thumb
        [5, 6], [6, 7], [7, 8],              # index
        [9, 10], [10, 11], [11, 12],         # middle
        [13, 14], [14, 15], [15, 16],        # ring
        [17, 18], [18, 19], [19, 20],        # pinky
        
        # Palm chain (right hand)
        [0, 5], [5, 9], [9, 13], [13, 17], [0, 17],
        
        # Left hand (indices 21-41, offset +21)
        [21, 22], [22, 23], [23, 24], [24, 25], # thumb
        [26, 27], [27, 28], [28, 29],            # index
        [30, 31], [31, 32], [32, 33],            # middle
        [34, 35], [35, 36], [36, 37],            # ring
        [38, 39], [39, 40], [40, 41],            # pinky
        
        # Palm chain (left hand)
        [21, 26], [26, 30], [30, 34], [34, 38], [21, 38]
    ]
    
    # Plot bone connections first (background)
    for connection in SKELETON_CONNECTIONS:
        i, j = connection
        if i < len(frame) and j < len(frame):
            # Check if both joints are valid (non-zero)
            if (frame[i] != 0).any() and (frame[j] != 0).any():
                ax.plot([frame[i, 0], frame[j, 0]], [frame[i, 1], frame[j, 1]], 
                       'b-', linewidth=1.5, alpha=0.6)
    
    # Plot joints on top
    valid_mask = (frame != 0).any(axis=1)
    valid_points = frame[valid_mask]
    
    if len(valid_points) > 0:
        ax.scatter(valid_points[:, 0], valid_points[:, 1], s=25, c='red', alpha=0.8, edgecolors='black', linewidth=0.5)
    
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    if len(valid_points) > 0:
        x_min, x_max = valid_points[:, 0].min(), valid_points[:, 0].max()
        y_min, y_max = valid_points[:, 1].min(), valid_points[:, 1].max()
        margin = 0.1
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)


def create_comparison_plot(original_data: np.ndarray, cleaned_data: np.ndarray, 
                          frame_idx: int, title: str = "", save_path: Optional[Path] = None) -> None:
    """Create side-by-side comparison plot"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot original data
    if VIZ_AVAILABLE:
        plot_skeleton_frame_2d(ax1, original_data, frame_idx, f"Original\n{title}")
    else:
        plot_frame_basic(ax1, original_data, frame_idx, f"Original\n{title}")
    
    # Plot cleaned data
    if VIZ_AVAILABLE:
        plot_skeleton_frame_2d(ax2, cleaned_data, frame_idx, f"Cleaned\n{title}")
    else:
        plot_frame_basic(ax2, cleaned_data, frame_idx, f"Cleaned\n{title}")
    
    # Add overall title
    fig.suptitle(f'Micro-Gap Cleaning Comparison - Frame {frame_idx}', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Comparison plot saved to: {save_path}")
    
    plt.show()


def create_comparison_animation(original_data: np.ndarray, cleaned_data: np.ndarray,
                               output_path: Path, title: str = "", fps: int = 15,
                               max_frames: int = 200) -> None:
    """Create side-by-side comparison animation"""
    
    # Limit frames for reasonable file size
    max_frames = min(max_frames, original_data.shape[0], cleaned_data.shape[0])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    def animate(frame_idx):
        ax1.clear()
        ax2.clear()
        
        # Plot original data
        if VIZ_AVAILABLE:
            plot_skeleton_frame_2d(ax1, original_data, frame_idx, "Original")
        else:
            plot_frame_basic(ax1, original_data, frame_idx, "Original")
        
        # Plot cleaned data
        if VIZ_AVAILABLE:
            plot_skeleton_frame_2d(ax2, cleaned_data, frame_idx, "Cleaned")
        else:
            plot_frame_basic(ax2, cleaned_data, frame_idx, "Cleaned")
        
        fig.suptitle(f'Micro-Gap Cleaning Comparison - Frame {frame_idx}', fontsize=14)
        return []
    
    ani = animation.FuncAnimation(fig, animate, frames=max_frames, blit=False, repeat=True, interval=1000//fps)
    
    try:
        # Try MP4 first
        writer = animation.FFMpegWriter(fps=fps)
        ani.save(str(output_path), writer=writer, dpi=120)
        print(f"✅ MP4 animation saved to: {output_path}")
    except Exception as e:
        print(f"⚠️  MP4 export failed: {e}")
        try:
            # Fallback to GIF
            gif_path = output_path.with_suffix('.gif')
            writer = animation.PillowWriter(fps=fps)
            ani.save(str(gif_path), writer=writer, dpi=100)
            print(f"✅ GIF animation saved to: {gif_path}")
        except Exception as e2:
            print(f"❌ Animation export failed: {e2}")
    
    plt.close(fig)


def analyze_data_quality(data: np.ndarray, name: str = "Data") -> dict:
    """Analyze data quality and return statistics using the same mask logic as cleaning"""
    
    # Use the same mask logic as test_micro_gap_cleaning.py
    # Detects frames where ALL coordinates are exactly zero (actual gaps)
    finite_mask = np.isfinite(data).all(axis=2)
    all_zero_mask = (data == 0).all(axis=2)
    valid_mask = finite_mask & ~all_zero_mask  # Valid = finite AND not all-zero
    
    # Calculate valid ratio (same as cleaning script)
    valid_ratio = valid_mask.mean()
    
    # For comparison, we need the missing ratio
    missing_ratio = 1.0 - valid_ratio
    
    # Frame-level analysis
    frame_missing_counts = (~valid_mask).sum(axis=1)
    frame_missing_ratios = frame_missing_counts / valid_mask.shape[1]
    
    # Joint-level analysis
    joint_missing_counts = (~valid_mask).sum(axis=0)
    joint_missing_ratios = joint_missing_counts / valid_mask.shape[0]
    
    stats = {
        'name': name,
        'shape': data.shape,
        'valid_ratio': float(valid_ratio),
        'missing_ratio': float(missing_ratio),
        'frames_perfect': int((frame_missing_ratios == 0).sum()),
        'frames_good': int(((frame_missing_ratios > 0) & (frame_missing_ratios <= 0.25)).sum()),
        'frames_fair': int(((frame_missing_ratios > 0.25) & (frame_missing_ratios <= 0.5)).sum()),
        'frames_poor': int(((frame_missing_ratios > 0.5) & (frame_missing_ratios <= 0.75)).sum()),
        'frames_bad': int((frame_missing_ratios > 0.75).sum()),
        'joints_perfect': int((joint_missing_ratios == 0).sum()),
        'joints_good': int(((joint_missing_ratios > 0) & (joint_missing_ratios <= 0.25)).sum()),
        'joints_fair': int(((joint_missing_ratios > 0.25) & (joint_missing_ratios <= 0.5)).sum()),
        'joints_poor': int(((joint_missing_ratios > 0.5) & (joint_missing_ratios <= 0.75)).sum()),
        'joints_bad': int((joint_missing_ratios > 0.75).sum())
    }
    
    return stats


def print_quality_report(original_stats: dict, cleaned_stats: dict) -> None:
    """Print a detailed quality comparison report"""
    
    print(f"\n{'='*80}")
    print("📊 MICRO-GAP CLEANING QUALITY REPORT")
    print(f"{'='*80}")
    
    # Overall comparison - show valid ratio improvement
    valid_improvement = cleaned_stats['valid_ratio'] - original_stats['valid_ratio']
    missing_improvement = original_stats['missing_ratio'] - cleaned_stats['missing_ratio']
    print(f"📈 OVERALL IMPROVEMENT:")
    print(f"   Valid data: {original_stats['valid_ratio']:.2%} → {cleaned_stats['valid_ratio']:.2%}")
    print(f"   Missing data: {original_stats['missing_ratio']:.2%} → {cleaned_stats['missing_ratio']:.2%}")
    print(f"   Improvement: {valid_improvement:+.2%} (valid) / {missing_improvement:+.2%} (missing)")
    
    # Frame-level analysis
    print(f"\n🎬 FRAME-LEVEL ANALYSIS:")
    print(f"   Original - Perfect: {original_stats['frames_perfect']}, Good: {original_stats['frames_good']}, "
          f"Fair: {original_stats['frames_fair']}, Poor: {original_stats['frames_poor']}, Bad: {original_stats['frames_bad']}")
    print(f"   Cleaned  - Perfect: {cleaned_stats['frames_perfect']}, Good: {cleaned_stats['frames_good']}, "
          f"Fair: {cleaned_stats['frames_fair']}, Poor: {cleaned_stats['frames_poor']}, Bad: {cleaned_stats['frames_bad']}")
    
    # Joint-level analysis
    print(f"\n🦴 JOINT-LEVEL ANALYSIS:")
    print(f"   Original - Perfect: {original_stats['joints_perfect']}, Good: {original_stats['joints_good']}, "
          f"Fair: {original_stats['joints_fair']}, Poor: {original_stats['joints_poor']}, Bad: {original_stats['joints_bad']}")
    print(f"   Cleaned  - Perfect: {cleaned_stats['joints_perfect']}, Good: {cleaned_stats['joints_good']}, "
          f"Fair: {cleaned_stats['joints_fair']}, Poor: {cleaned_stats['joints_poor']}, Bad: {cleaned_stats['joints_bad']}")
    
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Compare before/after micro-gap cleaning results")
    parser.add_argument("--original", type=Path, required=True, help="Path to original .npy file")
    parser.add_argument("--cleaned", type=Path, required=True, help="Path to cleaned .npy file")
    parser.add_argument("--frame", type=int, default=50, help="Frame index to visualize")
    parser.add_argument("--animate", action="store_true", help="Create comparison animation")
    parser.add_argument("--output_dir", type=Path, default=Path("comparison_outputs"), help="Output directory")
    parser.add_argument("--fps", type=int, default=15, help="FPS for animation")
    parser.add_argument("--max_frames", type=int, default=200, help="Max frames for animation")
    
    args = parser.parse_args()
    
    print("🔍 MICRO-GAP CLEANING COMPARISON")
    print("=" * 50)
    print(f"📁 Original file: {args.original}")
    print(f"📁 Cleaned file: {args.cleaned}")
    print(f"🎬 Frame to visualize: {args.frame}")
    print(f"🎬 Create animation: {args.animate}")
    print("=" * 50)
    
    # Load data
    print("📂 Loading data...")
    try:
        original_data = np.load(str(args.original))
        cleaned_data = np.load(str(args.cleaned))
        print(f"✅ Loaded original: {original_data.shape}, cleaned: {cleaned_data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Analyze data quality
    print("\n📊 Analyzing data quality...")
    original_stats = analyze_data_quality(original_data, "Original")
    cleaned_stats = analyze_data_quality(cleaned_data, "Cleaned")
    
    # Print quality report
    print_quality_report(original_stats, cleaned_stats)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison plot
    print(f"\n🎨 Creating comparison plot for frame {args.frame}...")
    plot_path = args.output_dir / f"comparison_frame_{args.frame}.png"
    create_comparison_plot(original_data, cleaned_data, args.frame, 
                          f"{args.original.stem} vs {args.cleaned.stem}", plot_path)
    
    # Create animation if requested
    if args.animate:
        print(f"\n🎬 Creating comparison animation...")
        anim_path = args.output_dir / f"comparison_animation_{args.original.stem}_vs_{args.cleaned.stem}.mp4"
        create_comparison_animation(original_data, cleaned_data, anim_path, 
                                  f"{args.original.stem} vs {args.cleaned.stem}", 
                                  args.fps, args.max_frames)
    
    print(f"\n🎉 Comparison completed!")
    print(f"📁 Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

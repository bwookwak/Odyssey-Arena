"""
Quick plotting script for experiment results.

Reads summary.json files from multiple experiment runs and generates comparison plots.

Usage:
    python -m experiments.plot_quick --output_dirs output/nomem output/naive output/gated
"""

import argparse
import json
from pathlib import Path
import sys


def load_summaries(output_dirs):
    """
    Load summary.json files from output directories.
    
    Args:
        output_dirs: List of output directory paths
        
    Returns:
        List of (label, summary_dict) tuples
    """
    summaries = []
    
    for output_dir in output_dirs:
        output_path = Path(output_dir)
        summary_file = output_path / 'summary.json'
        
        if not summary_file.exists():
            print(f"Warning: {summary_file} not found, skipping...")
            continue
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Use directory name as label
        label = output_path.name
        summaries.append((label, summary))
    
    return summaries


def plot_comparisons(summaries, output_prefix='comparison'):
    """
    Generate comparison plots.
    
    Args:
        summaries: List of (label, summary_dict) tuples
        output_prefix: Prefix for output plot filenames
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Error: matplotlib not installed. Install with: pip install matplotlib")
        sys.exit(1)
    
    if not summaries:
        print("Error: No summaries to plot")
        return
    
    labels = [s[0] for s in summaries]
    
    # Extract metrics
    success_rates = [s[1]['success_rate'] for s in summaries]
    avg_steps = [s[1]['avg_steps_success'] for s in summaries]
    loop_ratios = [s[1]['avg_loop_ratio'] for s in summaries]
    memory_usage = [s[1]['avg_memory_usage_rate'] for s in summaries]
    
    # Set up plot style
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig_width = max(8, len(labels) * 1.5)
    
    # Plot 1: Success Rate
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, success_rates, color='steelblue', alpha=0.8)
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, success_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2%}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_file = f'{output_prefix}_success_rate.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Plot 2: Average Steps to Success
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    bars = ax.bar(x_pos, avg_steps, color='coral', alpha=0.8)
    ax.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Steps (Success)', fontsize=12, fontweight='bold')
    ax.set_title('Steps to Success Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, avg_steps)):
        height = bar.get_height()
        if val > 0:  # Only show if non-zero
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_file = f'{output_prefix}_steps.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    # Plot 3: Loop Ratio and Memory Usage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width*1.5, 6))
    
    # Loop ratio
    bars1 = ax1.bar(x_pos, loop_ratios, color='tomato', alpha=0.8)
    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loop Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Loop Ratio (Repetitive Behavior)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, loop_ratios):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Memory usage
    bars2 = ax2.bar(x_pos, memory_usage, color='mediumseagreen', alpha=0.8)
    ax2.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Memory Usage Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Memory Usage Rate', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, memory_usage):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    output_file = f'{output_prefix}_loop_memory.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    print("\nPlotting complete!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comparison plots from experiment results"
    )
    
    parser.add_argument('--output_dirs', type=str, nargs='+', required=True,
                        help='List of output directories containing summary.json files')
    parser.add_argument('--output_prefix', type=str, default='comparison',
                        help='Prefix for output plot files')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load summaries
    print("Loading summaries...")
    summaries = load_summaries(args.output_dirs)
    
    if not summaries:
        print("Error: No valid summaries found")
        sys.exit(1)
    
    print(f"Loaded {len(summaries)} summaries")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_comparisons(summaries, output_prefix=args.output_prefix)


if __name__ == '__main__':
    main()

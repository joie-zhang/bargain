#!/usr/bin/env python3
"""
Visualize the most recent batch of experiment results.

This script automatically finds the most recent batch and provides options
to generate various visualizations.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Base directory
BASE_DIR = Path(__file__).parent
RESULTS_BASE = BASE_DIR / "experiments" / "results"

def find_most_recent_batch():
    """Find the most recently modified batch directory."""
    if not RESULTS_BASE.exists():
        print(f"Error: Results directory not found: {RESULTS_BASE}")
        return None
    
    # Get all subdirectories
    batches = []
    for item in RESULTS_BASE.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Get modification time
            mtime = item.stat().st_mtime
            batches.append((item, mtime))
    
    if not batches:
        print(f"Error: No batch directories found in {RESULTS_BASE}")
        return None
    
    # Sort by modification time (most recent first)
    batches.sort(key=lambda x: x[1], reverse=True)
    most_recent = batches[0][0]
    
    return most_recent

def run_visualization(script_name, results_dir, output_dir=None):
    """Run a visualization script."""
    script_path = BASE_DIR / "visualization" / script_name
    
    if not script_path.exists():
        print(f"Error: Visualization script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)]
    
    # Add arguments based on script
    if script_name == "create_scaling_regplot.py":
        cmd.extend(["--results-dir", str(results_dir)])
        if output_dir:
            cmd.extend(["--output-dir", str(output_dir)])
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=True, cwd=BASE_DIR)
        print(f"\n✅ Successfully completed: {script_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {script_name}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Visualize the most recent batch of experiment results"
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        help="Specific batch directory to visualize (default: most recent)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualization/figures",
        help="Output directory for figures (default: visualization/figures)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all visualization scripts"
    )
    parser.add_argument(
        "--heatmaps",
        action="store_true",
        help="Generate MMLU-ordered heatmaps"
    )
    parser.add_argument(
        "--regplot",
        action="store_true",
        help="Generate scaling regression plots"
    )
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Analyze convergence rounds"
    )
    parser.add_argument(
        "--tokens",
        action="store_true",
        help="Analyze token usage"
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Streamlit UI viewer"
    )
    
    args = parser.parse_args()
    
    # Find batch directory
    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        if not batch_dir.exists():
            print(f"Error: Batch directory not found: {batch_dir}")
            return 1
    else:
        batch_dir = find_most_recent_batch()
        if not batch_dir:
            return 1
    
    print(f"\n{'='*60}")
    print(f"Visualizing batch: {batch_dir.name}")
    print(f"Full path: {batch_dir}")
    print(f"{'='*60}\n")
    
    # If no specific option is selected, show menu
    if not any([args.all, args.heatmaps, args.regplot, args.convergence, args.tokens, args.ui]):
        print("No visualization option specified. Available options:")
        print("  --heatmaps     Generate MMLU-ordered heatmaps")
        print("  --regplot      Generate scaling regression plots")
        print("  --convergence  Analyze convergence rounds")
        print("  --tokens       Analyze token usage")
        print("  --ui           Launch Streamlit UI viewer")
        print("  --all          Run all visualization scripts")
        print("\nExample: python visualize_latest_batch.py --heatmaps --regplot")
        return 0
    
    # Launch UI if requested
    if args.ui:
        print("\n🚀 Launching Streamlit UI viewer...")
        print("The UI will open in your browser.")
        print("Press Ctrl+C to stop the server.\n")
        ui_path = BASE_DIR / "ui" / "negotiation_viewer.py"
        try:
            subprocess.run(["streamlit", "run", str(ui_path)], cwd=BASE_DIR)
        except KeyboardInterrupt:
            print("\n\nStopped UI server.")
        except FileNotFoundError:
            print("Error: streamlit not found. Install with: pip install streamlit")
            return 1
        return 0
    
    # Run visualizations
    success_count = 0
    total_count = 0
    
    if args.all or args.heatmaps:
        total_count += 1
        # Note: create_mmlu_ordered_heatmaps.py has hardcoded path
        # We'll need to modify it or create a symlink
        print("\n⚠️  Note: create_mmlu_ordered_heatmaps.py uses hardcoded paths.")
        print("   You may need to modify it or create a symlink to your results directory.")
        if run_visualization("create_mmlu_ordered_heatmaps.py", batch_dir, args.output_dir):
            success_count += 1
    
    if args.all or args.regplot:
        total_count += 1
        if run_visualization("create_scaling_regplot.py", batch_dir, args.output_dir):
            success_count += 1
    
    if args.all or args.convergence:
        total_count += 1
        if run_visualization("analyze_convergence_rounds.py", batch_dir, args.output_dir):
            success_count += 1
    
    if args.all or args.tokens:
        total_count += 1
        if run_visualization("analyze_token_usage.py", batch_dir, args.output_dir):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Completed {success_count}/{total_count} visualizations")
    if args.output_dir:
        output_path = Path(args.output_dir)
        if output_path.exists():
            print(f"\nFigures saved to: {output_path.absolute()}")
    print(f"{'='*60}\n")
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main())

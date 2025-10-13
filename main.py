import sys
import os
import argparse

from custom_logger import setup_logging 


# Ensure project and framework paths are on PYTHONPATH
sys.path.append(os.path.abspath('/workspace'))
sys.path.append(os.path.abspath('/workspace/frameworks/CoBEVT/opv2v'))

from attacker import Attacker
from tooling.InferenceRunner import *


from tooling.InferenceRunner import *

# Add the parent directory to Python path
sys.path.append(os.path.abspath('/workspace/CoBEVT/opv2v'))



# Now you can import directly

# Instead of: from CoBEVT.opv2v.opencood.data_utils.datasets import build_dataset

# Set up logging
logger = setup_logging()

def build_arg_parser():
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="CIAK Evaluation Script")
    
    parser.add_argument('--model_dir', type=str, default=None,
                    help='Optional checkpoint directory (overrides default if set).')
    parser.add_argument('--model_type', type=str, default='dynamic',
                    help='Model type: "dynamic" or "static" prediction')
    parser.add_argument('--config', type=str, required=True,
                    help='Path to model/data config yaml')
    parser.add_argument('--explore', action='store_true',
                    help='Enable detailed exploratory mode of the activation function inside of the model')
    parser.add_argument('--limit', type=int, default=0,
                    help='Limit number of batches to process (0 = all).')
    parser.add_argument('--no-cuda', action='store_true',
                    help='Force CPU even if CUDA is available.')
    parser.add_argument('--save-dir', type=str, default="/workspace/CIAK_Results",
                    help='Directory to save visualizations & outputs.')
    parser.add_argument('--debug', action='store_true',
                    help='Print per-batch tensor stats for model outputs.')
    # Add evaluation mode arguments
    parser.add_argument('--evaluate', action='store_true',
                    help='Enable evaluation between benign and attack runs')
    
    return parser


def run_inference(args, is_attack = False):
    """Run inference with given arguments.
    
    Args:
        args: Command line arguments
        is_attack: Whether this is an attack run
        
    Returns:
        Tuple of (metrics, objects_by_batch)
    """
    # Create inference runner
    runner = CoBEVTInferenceRunner(args)    
    runner.run_comparison()

    


def main():
    """Main entry point."""
    logger.info(":: Starting CoBEVT Inference Runner::")
    
    print("""
    ▄▄▄▄    ▄▄▄▄▄▄      ▄▄     ▄▄   ▄▄▄
  ██▀▀▀▀█   ▀▀██▀▀     ████    ██  ██▀
 ██▀          ██       ████    ██▄██
 ██           ██      ██  ██   █████
 ██▄          ██      ██████   ██  ██▄
  ██▄▄▄▄█   ▄▄██▄▄   ▄██  ██▄  ██   ██▄
    ▀▀▀▀    ▀▀▀▀▀▀   ▀▀    ▀▀  ▀▀    ▀▀"""
    )
    
    parser = build_arg_parser()
    args = parser.parse_args()
    
    if args.evaluate:
        try:
            # First run with benign data
        
            logger.info("Running Evaluation")
            _ = run_inference(args, is_attack=False)

            logger.info("Running Attack Evaluation")
            _ = run_inference(args, is_attack=True)

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


if __name__ == "__main__":
    main()
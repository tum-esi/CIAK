
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import statistics
import time
sys.path.append(os.path.abspath('/workspace'))
sys.path.append(os.path.abspath('/workspace/CoBEVT/opv2v/opencood'))

from CoBEVT.opv2v.opencood.utils.seg_utils import extract_class_confidence
from CoBEVT.opv2v.opencood.hypes_yaml import yaml_utils
from CoBEVT.opv2v.opencood.tools import train_utils, infrence_utils
from CoBEVT.opv2v.opencood.data_utils.datasets import build_dataset
from CoBEVT.opv2v.opencood.utils.seg_utils import *
from custom_logger import setup_logging
from tooling.ModelAnalzyer import *
logger = setup_logging()

# Change it based on your folder structure
DEFAULT_STATIC_CHECKPOINT = 'cobevt_static'
DEFAULT_DYNAMIC_CHECKPOINT = 'cobevt_dynamic'

from tooling.OutputManager import OutputManager
from tooling.ModelAnalzyer import save_encoder_activations_per_sample

class CoBEVTInferenceRunner:
    """Main runner class for CoBEVT inference."""
    
    def __init__(self, args) -> None:
        self.args = args
        self.output_manager = OutputManager(Path(args.save_dir))
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.device = None
        self.activation = {}  # For storing activation hooks
        self.hooks = []
        self.name = ""
    
    
    def setup(self, is_attack: bool) -> None:
        """Setup model, dataset, and device."""
        logger.info("Setting up inference...")
        
        # Load configuration
        self.hypes = yaml_utils.load_yaml(self.args.config)
        if is_attack:
            self.name = "attack"
            self.hypes['validate_dir'] = 'dataset_modifications/opv2v/attack/test'
        else:
            self.name = "benign"
            self.hypes['validate_dir'] = 'dataset_modifications/opv2v/benign/test'
           

        # Setup dataset
        dataset_visualize = False # Set to true if you want to use the CoBEVT built-in visualization 
        self.dataset = build_dataset(self.hypes, visualize=dataset_visualize, train=False)
        
        # Log dataset info if in debug mode
        if self.args.debug:
            logger.debug(f"Dataset: {type(self.dataset).__name__} | Samples: {len(self.dataset)}")
            if hasattr(self.dataset, 'dataset_dir'):
                logger.debug(f"  dir: {self.dataset.dataset_dir}")
            if hasattr(self.dataset, 'scenario_folders'):
                logger.debug(f"  scenarios: {len(self.dataset.scenario_folders)} (showing up to 3)")
                logger.debug(f"  first: {self.dataset.scenario_folders[:3]}")
        
        
        
        # Setup dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=4 if not self.args.explore else 10,
            collate_fn=self.dataset.collate_batch,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
        )
        
        if self.args.debug:
            logger.debug(f"DataLoader prepared with {len(self.dataloader)} batches")
        
        # Setup model
        self.model = train_utils.create_model(self.hypes)
        
        # Setup device
        use_cuda = (not self.args.no_cuda) and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        if use_cuda:
            self.model.to(self.device)
        
        # Load checkpoint
        if self.args.model_dir:
            saved_path = self.args.model_dir
        else:
            if self.args.model_type.lower() == 'dynamic':
                saved_path = DEFAULT_DYNAMIC_CHECKPOINT
            else:
                saved_path = DEFAULT_STATIC_CHECKPOINT
                
        logger.info(f"Loading checkpoint from {saved_path}")
        try:
            _, self.model = train_utils.load_saved_model(saved_path, self.model)
            logger.info("Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed loading checkpoint: {e}")
            raise RuntimeError("Failed to load model checkpoint")
            
        self.model.eval()
        
        # Register hooks if in exploration mode
        if self.args.explore:
            register_hooks(logger, self.activation, self.model, self.hooks)
    

    
    def run(self) -> Dict[str, Any]:
        """Run inference on the dataset."""
        start_time = time.time()
        logger.info("Starting inference...")
        
        # Metrics accumulators
        metrics = {
            'static_iou': [],  # Road IoU
            'lane_iou': [],    # Lane IoU
            'dynamic_iou': [],  # Vehicle IoU
            'dynamic_map': [],
            'confidence_maps': {}
        }
        
        if self.args.explore:
                        logger.warning(":: Explore Flag On ::")
        
        batch_limit = self.args.limit if self.args.limit > 0 else len(self.dataloader)
        
        # Progress bar
        iterable = enumerate(self.dataloader)
        if not self.args.explore:
            iterable = tqdm(iterable, total=min(batch_limit, len(self.dataloader)), 
                           desc='Inference', ncols=90)
        
        # Main inference loop
        for batch_idx, batch_data in iterable:
            if batch_idx >= batch_limit:
                break
                
            try:
                with torch.no_grad():
                    # Move data to device
                    batch_data = train_utils.to_device(batch_data, self.device)
                    
                    # Forward pass
                    raw_output = self.model(batch_data['ego'])
                    
                    # Debug raw model output stats if requested
                    if self.args.explore:
                        save_encoder_activations_per_sample(logger, self, batch_idx, batch_data)
                    # Post-process model output
                    output_dict = self.dataset.post_process(batch_data['ego'], raw_output)
                    
                    static_map, dynamic_map = get_maps(output_dict, batch_data)

                    iou_dynamic, iou_static = cal_iou_training(batch_data, output_dict)
                    metrics['static_iou'].append(iou_static[1])  # Road IoU
                    metrics['lane_iou'].append(iou_static[2])    # Lane IoU
                    metrics['dynamic_iou'].append(iou_dynamic[1])  # Vehicle IoU
                    metrics['dynamic_map'].append(dynamic_map) if dynamic_map is not None else []

                    
            except Exception as e:
                logger.warning(f"Skipping batch {batch_idx} due to error: {e}")
                continue
        
        # Clean up hooks
        self._clean_hooks()
        
        # Calculate final metrics
        def safe_mean(vals):
            return statistics.mean(vals) if vals else float('nan')
        
        final_metrics = {
            'road_iou': safe_mean(metrics['static_iou']),
            'lane_iou': safe_mean(metrics['lane_iou']),
            'dynamic_iou': safe_mean(metrics['dynamic_iou']),
            'batches': len(metrics['static_iou']),
            'elapsed': time.time() - start_time,
            'dynamic_map': metrics.get('dynamic_map'),
            
        }
        
        # Log results
        elapsed = final_metrics['elapsed']
        if self.args.explore:
            logger.info(f"Finished exploration in {elapsed:.1f}s | batches: {final_metrics['batches']}")
        else:
            logger.info(f"Done. {final_metrics['batches']} batches in {elapsed:.1f}s")

        logger.success(
            f"Road IoU: {final_metrics['road_iou']:.4f} | "
            f"Lane IoU: {final_metrics['lane_iou']:.4f} | "
            f"Dynamic IoU: {final_metrics['dynamic_iou']:.4f}"
        )
        
        return final_metrics
    
    def run_comparison(self) -> None:
        """Run both benign and attack inference and generate comparisons."""
        logger.info("Starting comparison run...")
        
        # First run benign inference
        logger.info("Running benign inference...")
        self.setup(is_attack=False)
        benign_metrics = self.run()
        
        # Then run attack inference
        logger.info("Running attack inference...")
        self.setup(is_attack=True)
        attack_metrics = self.run()
        
        logger.info("Comparison completed!")
        logger.info(f"Benign Dynamic IoU: {benign_metrics['dynamic_iou']:.4f}")
        logger.info(f"Attack Dynamic IoU: {attack_metrics['dynamic_iou']:.4f}")
        logger.info(f"Dynamic IoU Difference: {benign_metrics['dynamic_iou'] - attack_metrics['dynamic_iou']:.4f}")
        
        # The comparison visualizations are already generated by OutputManager
        # when the second run completes
        
        return
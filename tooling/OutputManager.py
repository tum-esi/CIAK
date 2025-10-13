import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import csv
logger = logging.getLogger(__name__)

class OutputManager:
    """Manages all output files and visualizations."""
    
    def __init__(self, save_dir: Path, name:str='benign'):
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dynamic_masks_dir = None
        self.collab_mask_dir = None
        self.comparison_dir = None
        self.name = name
        # Store metrics from different runs
        self.stored_metrics = {}
        
    def get_dynamic_masks_dir(self) -> Path:
        """Get directory for dynamic mask outputs."""
        if self.dynamic_masks_dir is None:
            self.dynamic_masks_dir = self.save_dir / 'dynamic_masks'
            self.dynamic_masks_dir.mkdir(parents=True, exist_ok=True)
        return self.dynamic_masks_dir
    
    def get_collab_mask_dir(self) -> Path:
        """Get directory for collaborative mask outputs."""
        if self.collab_mask_dir is None:
            self.collab_mask_dir = self.save_dir / 'collab_mask'
            self.collab_mask_dir.mkdir(parents=True, exist_ok=True)
            # Create legend file
            with open(self.collab_mask_dir / 'classes.txt', 'w') as lf:
                lf.write("0 road\n1 lane\n2 vehicle\n")
        return self.collab_mask_dir
    
    def get_comparison_dir(self) -> Path:
        """Get directory for object comparison outputs."""
        if self.comparison_dir is None:
            self.comparison_dir = self.save_dir / 'object_comparison'
            self.comparison_dir.mkdir(parents=True, exist_ok=True)
            # Create legend file for comparison visualization
            with open(self.comparison_dir / 'comparison_legend.txt', 'w') as lf:
                lf.write("Red: New objects (attack)\n")
                lf.write("Green: Missing objects (in benign but not in attack)\n")
                lf.write("Blue: Unchanged objects\n")
                lf.write("Gray: Road\n")
                lf.write("Yellow: Lane\n")
        return self.comparison_dir
    
    def save_dynamic_mask(self, batch_idx: int, dyn_prob: torch.Tensor, threshold: float) -> None:
        """Save dynamic masks with optional ground truth."""
        dyn_dir = self.get_dynamic_masks_dir()
        
        # Convert probability to binary mask
        dyn_np = dyn_prob.detach().cpu().numpy()[0]  # (H,W)
        mask = (dyn_np >= threshold).astype(np.uint8) * 255
        
        # Save binary mask
        cv2.imwrite(str(dyn_dir / f'{batch_idx:04d}_pred.png'), mask)
        
        # Save probability heatmap
        prob_color = (np.clip(dyn_np, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(dyn_dir / f'{batch_idx:04d}_prob.png'), prob_color)
        
    def save_collab_mask(self, batch_idx: int, static_pred: np.ndarray, 
                         dyn_mask: np.ndarray, vehicle_class_id: int,
                         objects: Optional[List[Dict]] = None) -> None:
        """Save collaborative mask combining static and dynamic predictions."""
        cm_dir = self.get_collab_mask_dir()
        
        # Combine static and dynamic predictions
        collab = static_pred.copy()
        collab[dyn_mask] = vehicle_class_id
        
        # Save class IDs
        ids_path = cm_dir / f'{batch_idx:04d}_ids.png'
        cv2.imwrite(str(ids_path), collab)
        
        # Create color visualization (BGR for OpenCV)
        color = np.zeros((collab.shape[0], collab.shape[1], 3), dtype=np.uint8)
        palette = {0: (128, 128, 128), 1: (0, 255, 255), vehicle_class_id: (0, 0, 255)}
        for k_cls, bgr in palette.items():
            color[collab == k_cls] = bgr
            
        color_path = cm_dir / f'{batch_idx:04d}_color.png'
        cv2.imwrite(str(color_path), color)
        
        # Save object information if available
        if objects:
            csv_path = cm_dir / f'{batch_idx:04d}_objects.csv'
            self._save_objects_csv(csv_path, objects, vehicle_class_id)
    
    def save_object_comparison(self, batch_idx: int, static_pred: np.ndarray, 
                              comparison_results: List[Dict], vehicle_class_id: int) -> None:
        """Save comparison visualization between benign and attack runs.
        
        Args:
            batch_idx (int): Batch index
            static_pred (np.ndarray): Static prediction (road/lane)
            comparison_results (List[Dict]): Results from BEVObjectExtractor.compare_runs
            vehicle_class_id (int): Vehicle class ID
        """
        comp_dir = self.get_comparison_dir()
        
        # Base static map (BGR format for OpenCV)
        base_palette = {0: (128, 128, 128), 1: (0, 255, 255)}
        
        # Create base static map
        H, W = static_pred.shape
        color_vis = np.zeros((H, W, 3), dtype=np.uint8)
        for cls_id, bgr in base_palette.items():
            color_vis[static_pred == cls_id] = bgr
        
        # Define colors for different object statuses (BGR format)
        status_colors = {
            'new': (0, 0, 255),      # Red (attack objects)
            'missing': (0, 255, 0),  # Green (missing objects)
            'unchanged': (255, 0, 0)  # Blue (unchanged objects)
        }
        
        # Draw objects with their status colors
        for obj in comparison_results:
            status = obj['status']
            color = status_colors[status]
            
            # Use object mask if available
            if 'mask' in obj and obj['mask'] is not None:
                mask = obj['mask']
                color_vis[mask] = color
            else:
                # Fall back to bounding box
                x, y, w, h = obj['bbox']
                cv2.rectangle(color_vis, (x, y), (x+w, y+h), color, -1)  # -1 means filled
        
        # Save the visualization
        vis_path = comp_dir / f'{batch_idx:04d}_comparison.png'
        cv2.imwrite(str(vis_path), color_vis)
        
        # Save objects to CSV with status information
        csv_path = comp_dir / f'{batch_idx:04d}_comparison.csv'
        self._save_comparison_csv(csv_path, comparison_results, vehicle_class_id)
        
        logger.info(f"Saved object comparison to {vis_path}")
    
    def _save_comparison_csv(self, path: Path, comparison_results: List[Dict], class_id: int) -> None:
        """Save comparison results to CSV file."""
        with open(path, 'w', newline='') as fcsv:
            fieldnames = ['id', 'status', 'class_id', 'area', 'bbox', 
                          'centroid', 'mean_prob', 'max_prob']
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            for o in comparison_results:
                row = {
                    'id': o.get('id', -1),
                    'status': o.get('status', 'unknown'),
                    'class_id': class_id,
                    'area': o.get('area', 0),
                    'bbox': str(o.get('bbox', [])),
                    'centroid': str(o.get('centroid', [])),
                    'mean_prob': o.get('mean_prob', 0.0),
                    'max_prob': o.get('max_prob', 0.0),
                }
                writer.writerow(row)
    
    def _save_objects_csv(self, path: Path, objects: List[Dict], class_id: int) -> None:
        """Save object information to CSV file."""
        with open(path, 'w', newline='') as fcsv:
            fieldnames = ['id', 'class_id', 'class_name', 'area', 'bbox', 
                          'centroid', 'mean_prob', 'max_prob']
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            for o in objects:
                writer.writerow({
                    'id': o['id'],
                    'class_id': class_id,
                    'class_name': 'vehicle',
                    'area': o['area'],
                    'bbox': o['bbox'],
                    'centroid': o['centroid'],
                    'mean_prob': o['mean_prob'],
                    'max_prob': o['max_prob'],
                })
    
    def save_summary_visualizations(self, metrics: Dict[str, float], batch_metrics: Dict[str, List[float]], name:str) -> None:
        """Create and save summary visualizations of metrics."""
        # Store the metrics with the run name
        self.stored_metrics[name] = {
            'final': metrics,
            'batch': batch_metrics
        }
        
        # Mean IoU bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ['Road', 'Lane', 'Dynamic']
        means = [metrics['road_iou'], metrics['lane_iou'], metrics['dynamic_iou']]
        bars = ax.bar(labels, means, color=['steelblue', 'seagreen', 'indianred'])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Mean IoU')
        ax.set_title(f'{name.capitalize()} Inference Mean IoUs')
        
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width()/2, b.get_height()+0.01, 
                   f"{m:.3f}", ha='center', va='bottom', fontsize=9)
                   
        summary_path = self.save_dir / f'{name}_iou_summary.png'
        fig.tight_layout()
        fig.savefig(summary_path)
        plt.close(fig)
        
        # Per-batch curves if multiple batches collected
        if len(batch_metrics['static_iou']) > 1:
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            x = list(range(1, len(batch_metrics['static_iou'])+1))
            ax2.plot(x, batch_metrics['static_iou'], label='Road IoU', color='steelblue')
            ax2.plot(x, batch_metrics['lane_iou'], label='Lane IoU', color='seagreen')
            ax2.plot(x, batch_metrics['dynamic_iou'], label='Dynamic IoU', color='indianred')
            ax2.set_xlabel('Batch')
            ax2.set_ylabel('IoU')
            ax2.set_ylim(0, 1.0)
            ax2.set_title(f'{name.capitalize()} Per-Batch IoU Curves')
            ax2.legend(loc='lower right')
            fig2.tight_layout()
            curves_path = self.save_dir / f'{name}_iou_curves.png'
            fig2.savefig(curves_path)
            plt.close(fig2)
            logger.info(f"Saved IoU curves to {curves_path}")
            
            # Create comparison visualization if we have both benign and attack data
            if 'benign' in self.stored_metrics and 'attack' in self.stored_metrics:
                self.create_comparison_visualization()
                self.create_dynamic_iou_comparison()
                self.create_dynamic_error_comparison()
                self.export_iou_to_csv()
            
        logger.info(f"Saved summary visualization to {summary_path}")
    
    def create_comparison_visualization(self) -> None:
        """Create a comparison visualization between benign and attack runs."""
        # Ensure we have both sets of metrics
        if 'benign' not in self.stored_metrics or 'attack' not in self.stored_metrics:
            logger.warning("Cannot create comparison: missing either benign or attack metrics")
            return
        
        benign_metrics = self.stored_metrics['benign']['batch']
        attack_metrics = self.stored_metrics['attack']['batch']
        
        # Create a figure with three subplots (road, lane, dynamic)
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        metrics_to_plot = [
            ('static_iou', 'Road IoU', 0),
            ('lane_iou', 'Lane IoU', 1),
            ('dynamic_iou', 'Dynamic IoU', 2)
        ]
        
        for metric_key, title, idx in metrics_to_plot:
            ax = axs[idx]
            
            # Get data
            benign_data = benign_metrics.get(metric_key, [])
            attack_data = attack_metrics.get(metric_key, [])
            
            # Find common length if different
            min_len = min(len(benign_data), len(attack_data))
            if min_len == 0:
                continue
                
            x = list(range(1, min_len+1))
            
            # Plot both curves
            ax.plot(x, benign_data[:min_len], label='Benign', color='green', marker='o', markersize=4)
            ax.plot(x, attack_data[:min_len], label='Attack', color='red', marker='x', markersize=4)
            
            # Calculate and display mean values
            benign_mean = sum(benign_data[:min_len]) / min_len if min_len > 0 else 0
            attack_mean = sum(attack_data[:min_len]) / min_len if min_len > 0 else 0
            difference = attack_mean - benign_mean
            
            # Add text with mean values
            ax.text(0.02, 0.95, f"Benign mean: {benign_mean:.3f}", transform=ax.transAxes, 
                   verticalalignment='top', color='green')
            ax.text(0.02, 0.87, f"Attack mean: {attack_mean:.3f}", transform=ax.transAxes, 
                   verticalalignment='top', color='red')
            ax.text(0.02, 0.79, f"Difference: {difference:.3f}", transform=ax.transAxes, 
                   verticalalignment='top', color='black')
            
            # Customize subplot
            ax.set_title(title)
            ax.set_ylabel('IoU')
            ax.set_ylim(0, 1.0)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='lower right')
        
        # Set common labels
        axs[-1].set_xlabel('Batch')
        
        # Add overall title
        fig.suptitle('Benign vs Attack IoU Comparison', fontsize=16)
        
        # Save the figure
        fig.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
        comparison_path = self.save_dir / 'benign_vs_attack_comparison.png'
        fig.savefig(comparison_path)
        plt.close(fig)
        
        logger.info(f"Saved benign vs attack comparison to {comparison_path}")
        
        # Also create a bar chart comparing the means
        self.create_mean_comparison_barchart()
    
    def create_mean_comparison_barchart(self) -> None:
        """Create a bar chart comparing mean IoUs between benign and attack runs."""
        benign_final = self.stored_metrics['benign']['final']
        attack_final = self.stored_metrics['attack']['final']
        
        metrics = ['road_iou', 'lane_iou', 'dynamic_iou']
        labels = ['Road', 'Lane', 'Dynamic']
        
        benign_means = [benign_final.get(m, 0) for m in metrics]
        attack_means = [attack_final.get(m, 0) for m in metrics]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(8, 6))
        rects1 = ax.bar(x - width/2, benign_means, width, label='Benign', color='green')
        rects2 = ax.bar(x + width/2, attack_means, width, label='Attack', color='red')
        
        # Add value labels
        def add_labels(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        add_labels(rects1)
        add_labels(rects2)
        
        # Customize plot
        ax.set_ylabel('Mean IoU')
        ax.set_title('Benign vs Attack Mean IoU Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        fig.tight_layout()
        barchart_path = self.save_dir / 'benign_vs_attack_barchart.png'
        fig.savefig(barchart_path)
        plt.close(fig)
        
        logger.info(f"Saved benign vs attack bar chart to {barchart_path}")
    
    def create_dynamic_iou_comparison(self) -> None:
        """Create a single graph showing only dynamic IoU comparison between benign and attack."""
        if 'benign' not in self.stored_metrics or 'attack' not in self.stored_metrics:
            logger.warning("Cannot create dynamic comparison: missing either benign or attack metrics")
            return
        
        benign_metrics = self.stored_metrics['benign']['batch']
        attack_metrics = self.stored_metrics['attack']['batch']
        
        # Get dynamic IoU data
        benign_data = benign_metrics.get('dynamic_iou', [])
        attack_data = attack_metrics.get('dynamic_iou', [])
        
        # Find common length if different
        min_len = min(len(benign_data), len(attack_data))
        if min_len == 0:
            logger.warning("Cannot create dynamic comparison: no data available")
            return
            
        x = list(range(1, min_len+1))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot both curves
        ax.plot(x, benign_data[:min_len], label='Benign', color='green', marker='o', markersize=4, linewidth=2)
        ax.plot(x, attack_data[:min_len], label='Attack', color='red', marker='x', markersize=4, linewidth=2)
        
        # Calculate and display mean values and distance
        benign_mean = sum(benign_data[:min_len]) / min_len if min_len > 0 else 0
        attack_mean = sum(attack_data[:min_len]) / min_len if min_len > 0 else 0
        difference = benign_mean - attack_mean  # Positive if benign is higher (which is expected)
        
        # Calculate distances per batch
        distances = [benign_data[i] - attack_data[i] for i in range(min_len)]
        avg_distance = sum(distances) / len(distances) if distances else 0
        
        # Add text with mean values
        ax.text(0.02, 0.95, f"Benign mean: {benign_mean:.3f}", transform=ax.transAxes, 
               verticalalignment='top', color='green', fontsize=12)
        ax.text(0.02, 0.90, f"Attack mean: {attack_mean:.3f}", transform=ax.transAxes, 
               verticalalignment='top', color='red', fontsize=12)
        ax.text(0.02, 0.85, f"Mean difference: {difference:.3f}", transform=ax.transAxes, 
               verticalalignment='top', color='black', fontsize=12, weight='bold')
        ax.text(0.02, 0.80, f"Avg per-batch difference: {avg_distance:.3f}", transform=ax.transAxes, 
               verticalalignment='top', color='black', fontsize=12)
        
        # Customize plot
        ax.set_title('Dynamic IoU Comparison: Benign vs Attack', fontsize=14, fontweight='bold')
        ax.set_xlabel('Batch', fontsize=12)
        ax.set_ylabel('Dynamic IoU', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='lower right', fontsize=12)
        
        # Add shaded area between curves to highlight difference
        ax.fill_between(x, benign_data[:min_len], attack_data[:min_len], 
                       color='red', alpha=0.2, label='IoU Reduction')
        
        # Save the figure
        fig.tight_layout()
        comparison_path = self.save_dir / 'dynamic_iou_comparison.png'
        fig.savefig(comparison_path)
        plt.close(fig)
        
        logger.info(f"Saved dynamic IoU comparison to {comparison_path}")
    
    def create_dynamic_error_comparison(self) -> None:
        """Create a graph showing 1-IoU (error rate) for dynamic objects."""
        if 'benign' not in self.stored_metrics or 'attack' not in self.stored_metrics:
            logger.warning("Cannot create error comparison: missing either benign or attack metrics")
            return
        
        benign_metrics = self.stored_metrics['benign']['batch']
        attack_metrics = self.stored_metrics['attack']['batch']
        
        # Get dynamic IoU data and convert to error rate (1-IoU)
        benign_data = [1.0 - iou for iou in benign_metrics.get('dynamic_iou', [])]
        attack_data = [1.0 - iou for iou in attack_metrics.get('dynamic_iou', [])]
        
        # Find common length if different
        min_len = min(len(benign_data), len(attack_data))
        if min_len == 0:
            logger.warning("Cannot create error comparison: no data available")
            return
            
        x = list(range(1, min_len+1))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot both curves
        ax.plot(x, benign_data[:min_len], label='Benign', color='green', marker='o', markersize=4, linewidth=2)
        ax.plot(x, attack_data[:min_len], label='Attack', color='red', marker='x', markersize=4, linewidth=2)
        
        # Calculate and display mean values and distance
        benign_mean = sum(benign_data[:min_len]) / min_len if min_len > 0 else 0
        attack_mean = sum(attack_data[:min_len]) / min_len if min_len > 0 else 0
        difference = attack_mean - benign_mean  # Positive if attack has higher error (which is expected)
        
        # Add text with mean values
        ax.text(0.02, 0.95, f"Benign mean error: {benign_mean:.3f}", transform=ax.transAxes, 
               verticalalignment='top', color='green', fontsize=12)
        ax.text(0.02, 0.90, f"Attack mean error: {attack_mean:.3f}", transform=ax.transAxes, 
               verticalalignment='top', color='red', fontsize=12)
        ax.text(0.02, 0.85, f"Error increase: {difference:.3f}", transform=ax.transAxes, 
               verticalalignment='top', color='black', fontsize=12, weight='bold')
        
        # Customize plot
        ax.set_title('Dynamic Error Rate (1-IoU) Comparison: Benign vs Attack', fontsize=14, fontweight='bold')
        ax.set_xlabel('Batch', fontsize=12)
        ax.set_ylabel('Error Rate (1-IoU)', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper right', fontsize=12)
        
        # Add shaded area between curves to highlight difference
        ax.fill_between(x, attack_data[:min_len], benign_data[:min_len], 
                       color='red', alpha=0.2, label='Error Increase')
        
        # Save the figure
        fig.tight_layout()
        comparison_path = self.save_dir / 'dynamic_error_comparison.png'
        fig.savefig(comparison_path)
        plt.close(fig)
        
        logger.info(f"Saved dynamic error rate comparison to {comparison_path}")
    
    def export_iou_to_csv(self) -> None:
        """Export IoU values per batch to CSV for Excel analysis."""
        if 'benign' not in self.stored_metrics or 'attack' not in self.stored_metrics:
            logger.warning("Cannot export IoU CSV: missing either benign or attack metrics")
            return
        
        benign_metrics = self.stored_metrics['benign']['batch']
        attack_metrics = self.stored_metrics['attack']['batch']
        
        # Get all types of IoU data
        metrics_to_export = [
            ('static_iou', 'Road'),
            ('lane_iou', 'Lane'),
            ('dynamic_iou', 'Dynamic')
        ]
        
        # Find common length for all metrics
        min_lens = []
        for metric_key, _ in metrics_to_export:
            benign_data = benign_metrics.get(metric_key, [])
            attack_data = attack_metrics.get(metric_key, [])
            min_lens.append(min(len(benign_data), len(attack_data)))
        
        min_len = min(min_lens) if min_lens else 0
        if min_len == 0:
            logger.warning("Cannot export IoU CSV: no data available")
            return
        
        # Create CSV file
        csv_path = self.save_dir / 'iou_comparison.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            header = ['Batch']
            for _, label in metrics_to_export:
                header.extend([f'Benign {label} IoU', f'Attack {label} IoU', f'Difference ({label})'])
            writer.writerow(header)
            
            # Write data rows
            for batch_idx in range(min_len):
                row = [batch_idx + 1]  # Batch number (1-indexed)
                
                for metric_key, _ in metrics_to_export:
                    benign_val = benign_metrics.get(metric_key, [])[batch_idx]
                    attack_val = attack_metrics.get(metric_key, [])[batch_idx]
                    diff = benign_val - attack_val
                    
                    row.extend([benign_val, attack_val, diff])
                
                writer.writerow(row)
            
            # Write summary row with averages
            summary_row = ['Average']
            for metric_key, _ in metrics_to_export:
                benign_data = benign_metrics.get(metric_key, [])[:min_len]
                attack_data = attack_metrics.get(metric_key, [])[:min_len]
                
                benign_mean = sum(benign_data) / len(benign_data) if benign_data else 0
                attack_mean = sum(attack_data) / len(attack_data) if attack_data else 0
                diff_mean = benign_mean - attack_mean
                
                summary_row.extend([benign_mean, attack_mean, diff_mean])
            
            writer.writerow([])  # Empty row
            writer.writerow(summary_row)
            
            # Also add 1-IoU (error rates) summary
            error_row = ['Error Rate (1-IoU)']
            for metric_key, _ in metrics_to_export:
                benign_data = benign_metrics.get(metric_key, [])[:min_len]
                attack_data = attack_metrics.get(metric_key, [])[:min_len]
                
                benign_error = 1 - (sum(benign_data) / len(benign_data) if benign_data else 0)
                attack_error = 1 - (sum(attack_data) / len(attack_data) if attack_data else 0)
                error_diff = attack_error - benign_error
                
                error_row.extend([benign_error, attack_error, error_diff])
            
            writer.writerow([])  # Empty row
            writer.writerow(error_row)
        
        logger.info(f"Exported IoU comparison data to {csv_path}")
        
        # Create a detailed CSV just for dynamic IoU
        dynamic_csv_path = self.save_dir / 'dynamic_iou_comparison.csv'
        with open(dynamic_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Batch', 'Benign Dynamic IoU', 'Attack Dynamic IoU', 'Difference', 
                            'Benign Error (1-IoU)', 'Attack Error (1-IoU)', 'Error Increase'])
            
            benign_dynamic = benign_metrics.get('dynamic_iou', [])[:min_len]
            attack_dynamic = attack_metrics.get('dynamic_iou', [])[:min_len]
            
            # Write data rows
            for batch_idx in range(min_len):
                benign_val = benign_dynamic[batch_idx]
                attack_val = attack_dynamic[batch_idx]
                diff = benign_val - attack_val
                
                benign_error = 1 - benign_val
                attack_error = 1 - attack_val
                error_diff = attack_error - benign_error
                
                writer.writerow([
                    batch_idx + 1,  # Batch number (1-indexed)
                    benign_val,
                    attack_val,
                    diff,
                    benign_error,
                    attack_error,
                    error_diff
                ])
            
            # Write summary row
            benign_mean = sum(benign_dynamic) / len(benign_dynamic) if benign_dynamic else 0
            attack_mean = sum(attack_dynamic) / len(attack_dynamic) if attack_dynamic else 0
            diff_mean = benign_mean - attack_mean
            
            benign_error_mean = 1 - benign_mean
            attack_error_mean = 1 - attack_mean
            error_diff_mean = attack_error_mean - benign_error_mean
            
            writer.writerow([])  # Empty row
            writer.writerow([
                'Average',
                benign_mean,
                attack_mean,
                diff_mean,
                benign_error_mean,
                attack_error_mean,
                error_diff_mean
            ])
        
        logger.info(f"Exported detailed dynamic IoU comparison data to {dynamic_csv_path}")
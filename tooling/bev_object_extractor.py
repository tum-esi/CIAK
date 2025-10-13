"""
BEV object extraction utilities.

This module provides utilities to extract object-wise statistics from dynamic
probability maps in BEV space and to compare objects between two runs.

Extracted from InferenceRunner to improve modularity and maintainability.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch


class BEVObjectExtractor:
    """Extract object information from BEV probability maps."""

    @staticmethod
    def from_probability_map(
        dyn_prob: torch.Tensor,
        thresh: float = 0.5,
        min_area: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        From a dynamic probability map [B,H,W], return object-wise stats by
        thresholding and connected components.

        Args:
            dyn_prob: Tensor of shape [B, H, W] (probabilities or logits).
            thresh: Threshold to binarize the probability map.
            min_area: Minimum connected-component area to keep.

        Returns:
            List of objects with fields: id, area, bbox, centroid, mean_prob,
            max_prob, and mask (boolean array HxW).
        """
        # Expect shape [B,H,W]
        prob_np = dyn_prob.detach().float().cpu().numpy()[0]  # (H,W)

        # If values are not in [0,1], assume logits and apply sigmoid
        if prob_np.min() < 0.0 or prob_np.max() > 1.0:
            prob_np = 1.0 / (1.0 + np.exp(-prob_np))

        bin_mask = (prob_np >= thresh).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bin_mask, connectivity=8
        )

        objects: List[Dict[str, Any]] = []
        for lbl in range(1, num_labels):
            area = int(stats[lbl, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            x = int(stats[lbl, cv2.CC_STAT_LEFT])
            y = int(stats[lbl, cv2.CC_STAT_TOP])
            w = int(stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[lbl]

            mask = labels == lbl
            mean_prob = float(prob_np[mask].mean())
            max_prob = float(prob_np[mask].max())

            objects.append(
                {
                    "id": lbl,
                    "area": area,
                    "bbox": [x, y, w, h],
                    "centroid": [float(cx), float(cy)],
                    "mean_prob": mean_prob,
                    "max_prob": max_prob,
                    "mask": mask.copy(),  # Store the actual mask for comparison
                }
            )

        return objects

    @staticmethod
    def compare_runs(
        benignRun: List[Dict[str, Any]],
        attackRun: List[Dict[str, Any]],
        distance_threshold: float = 5.0,
    ) -> List[Dict[str, Any]]:
        """
        Finds the difference between two runs, and marks objects as new, missing, or unchanged.

        Args:
            benignRun: List of benign run objects.
            attackRun: List of attack run objects.
            distance_threshold: Maximum centroid distance to consider objects the same.

        Returns:
            List of objects with a 'status' field ('new', 'missing', 'unchanged').
        """
        comparison_results: List[Dict[str, Any]] = []

        # Mark all benign objects as potentially missing
        benign_objects: List[Dict[str, Any]] = []
        for obj in benignRun:
            obj_copy = obj.copy()
            obj_copy["status"] = "missing"
            benign_objects.append(obj_copy)

        # Process attack objects and compare with benign objects
        matched_benign_indices = set()

        for attack_obj in attackRun:
            attack_obj_copy = attack_obj.copy()
            attack_cx, attack_cy = attack_obj["centroid"]

            # Look for a matching object in benign run
            matched = False

            for i, benign_obj in enumerate(benign_objects):
                benign_cx, benign_cy = benign_obj["centroid"]

                # Euclidean distance between centroids
                distance = float(
                    np.sqrt((attack_cx - benign_cx) ** 2 + (attack_cy - benign_cy) ** 2)
                )

                if distance <= distance_threshold:
                    matched = True
                    matched_benign_indices.add(i)

                    # Update the benign object status to 'unchanged'
                    benign_objects[i]["status"] = "unchanged"
                    break

            # If no match found, this is a new object
            attack_obj_copy["status"] = "unchanged" if matched else "new"
            comparison_results.append(attack_obj_copy)

        # Add all benign objects that weren't matched (still marked as 'missing')
        for i, benign_obj in enumerate(benign_objects):
            if i not in matched_benign_indices:
                comparison_results.append(benign_obj)

        return comparison_results

    @staticmethod
    def create_comparison_mask(
        comparison_results: List[Dict[str, Any]],
        shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Creates a colored mask showing object differences.

        Args:
            comparison_results: Results from compare_runs
            shape: Shape of the mask (H, W)

        Returns:
            RGB mask highlighting object differences (np.uint8, shape HxWx3)
        """
        # Create RGB mask (three channels)
        mask = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)

        # Define colors for different statuses (in RGB format)
        colors = {
            "new": (255, 0, 0),  # Red for attack/new objects
            "missing": (0, 255, 0),  # Green for missing objects
            "unchanged": (0, 0, 255),  # Blue for unchanged objects
        }

        # Fill in the mask based on object status
        for obj in comparison_results:
            status = obj["status"]
            color = colors[status]

            # If object has a stored mask, use it
            if "mask" in obj:
                mask[obj["mask"]] = color
            else:
                # Otherwise use bounding box
                x, y, w, h = obj["bbox"]
                mask[y : y + h, x : x + w] = color

        return mask

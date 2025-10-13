from typing import Dict,Optional, Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt



def register_hooks(logger, activation={}, model=None, hooks=[]) -> None:
    """Register forward hooks to capture model activations."""
    
    logger.debug("Registering model hooks for exploration mode")
    if model is None:
        logger.critical("Model is None, cannot register hooks")
        return
    
    def get_activation(name):
        def hook(module, input_, output):
            if isinstance(output, list):
                activation[name] = [o.detach() if isinstance(o, torch.Tensor) else o for o in output]
            elif isinstance(output, dict):
                activation[name] = {k: v.detach() if isinstance(v, torch.Tensor) else v 
                                        for k, v in output.items()}
            elif isinstance(output, torch.Tensor):
                activation[name] = output.detach()
            else:
                activation[name] = output
                logger.warning(f"{name} output type {type(output)} not tensor/list/dict")
        return hook
    
    try:
        if hasattr(model, 'encoder'):
            hooks.append(model.encoder.register_forward_hook(get_activation('encoder')))
            if hasattr(model.encoder, 'encoder'):
                for layer in [1, 2, 3, 4]:
                    ln = f'layer{layer}'
                    if hasattr(model.encoder.encoder, ln):
                        hooks.append(getattr(model.encoder.encoder, ln).register_forward_hook(
                            get_activation(f'encoder_{ln}')))
    except Exception as e:
        logger.warning(f"Failed to hook encoder: {e}")
        
    try:
        if hasattr(model, 'fusion_net'):
            def fusion_input_hook(module, input_):
                activation['pre_fusion_input'] = [i.detach() if isinstance(i, torch.Tensor) else i for i in input_]
                return None
            hooks.append(model.fusion_net.register_forward_pre_hook(fusion_input_hook))
            hooks.append(model.fusion_net.register_forward_hook(get_activation('fusion_net_output')))
    except Exception as e:
        logger.warning(f"Failed to hook fusion_net: {e}")
        
    try:
        if hasattr(model, 'decoder'):
            hooks.append(model.decoder.register_forward_hook(get_activation('decoder')))
    except Exception as e:
        logger.warning(f"Failed to hook decoder: {e}")

def clean_hooks() -> None:
    """Clean up registered hooks."""
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass
    hooks = []



def extract_static_prediction(output_dict: Dict) -> Tuple[Optional[np.ndarray], Tuple[Optional[int], Optional[int]]]:
    """Extract static prediction and shape from output dictionary."""
    H = W = None
    static_pred = None
    
    # Try to get from static probability tensor
    static_prob = output_dict.get('static_prob')
    if isinstance(static_prob, torch.Tensor) and static_prob.dim() == 4:
        sp = static_prob.detach().cpu().numpy()[0]  # [C,H,W]
        static_pred = sp.argmax(axis=0).astype(np.uint8)
        H, W = static_pred.shape
    
    # Try to get from static map tensor
    elif isinstance(output_dict.get('static_map'), torch.Tensor):
        static_map = output_dict.get('static_map')
        sm = static_map.detach().cpu().numpy()
        if sm.ndim == 4:
            sm = sm[0, 0]
        elif sm.ndim == 3:
            sm = sm[0]
        static_pred = sm.astype(np.uint8)
        H, W = static_pred.shape
    
    # Fallback shape from dynamic map
    dyn_prob = output_dict.get('dynamic_map')
    if isinstance(dyn_prob, torch.Tensor) and (H is None or W is None):
        dnp = dyn_prob.detach().cpu().numpy()[0]
        H, W = dnp.shape
        
    if static_pred is None and H is not None and W is not None:
        static_pred = np.zeros((H, W), dtype=np.uint8)
        
    return static_pred, (H, W)

def extract_dynamic_mask(logger,output_dict: Dict, threshold: float) -> Optional[np.ndarray]:
    """Extract dynamic mask from output dictionary."""
    logger.info(f"Available keys in output_dict: {list(output_dict.keys())}")

    dyn_prob = output_dict.get('dynamic_map')
    if dyn_prob is None:
        logger.warning("'dynamic_map' key not found in output_dict")
        return None
        
    if not isinstance(dyn_prob, torch.Tensor):
        logger.warning(f"'dynamic_map' is not a tensor (type: {type(dyn_prob)})")
        return None
            
    dnp = dyn_prob.detach().cpu().numpy()[0]
    return dnp
def extract_sample_ids(batch_data, B):
    """
    Try hard to build stable per-sample identifiers from batch_data.
    Works with OpenCOOD-style dicts if available, otherwise falls back.
    """
    ids = []
    ego = batch_data.get('ego', {})
    # Common places where identifiers may live (adapt as needed):
    # - file paths per camera
    # - scenario / frame index
    candidate_keys = [
        ('meta', 'paths'), ('meta', 'image_paths'),
        ('meta', 'scenario'), ('meta', 'frame'),
        ('sample_id',), ('idx',), ('index',),
        ('scenario',), ('frame',)
    ]

    def find_in(d, path):
        cur = d
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return None
        return cur

    for i in range(B):
        label = None
        for path in candidate_keys:
            v = find_in(ego, path)
            if v is None:
                v = find_in(batch_data, path)
            if v is not None:
                # If it's a per-sample list, index it; if scalar, just use it
                if isinstance(v, (list, tuple)) and len(v) == B:
                    label = v[i]
                else:
                    label = v
                break

        if isinstance(label, (list, tuple)):
            # e.g., multi-camera paths -> shorten to basename of first cam
            try:
                import os
                label = os.path.basename(label[0])
            except Exception:
                label = str(label)

        if label is None:
            label = f"sample_{i:03d}"
        else:
            # sanitize for filename
            safe = str(label).replace(os.sep, "_").replace(" ", "_")
            label = f"{i:03d}_{safe[:80]}"
        ids.append(label)
    return ids

def save_encoder_activations_per_sample(logger, inference_runner, batch_idx: int, batch_data: Dict) -> None:
    # Create base directory for this batch
    root = inference_runner.output_manager.save_dir / "encoder_acts" / f"batch_{batch_idx:05d}"
    root.mkdir(parents=True, exist_ok=True)
    
    # Get sample IDs
    sample_ids = extract_sample_ids(batch_data, 1)
    
    # Only process sample index 0 (first sample in batch)
    
    i=0
    for sample_id in sample_ids:
        
        sample_dir = root / f"{sample_id}_{inference_runner.name}_{i}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        # Just create one summary image with the 6 requested layers
        try:
            # Collect the 6 specific activations we want
            important_activations = []
            
            # Try to get encoder layers 1-4
            for layer_num in range(1, 5):
                key = f'encoder_layer{layer_num}'
                if key in inference_runner.activation and isinstance(inference_runner.activation[key], torch.Tensor):
                    # Make tensor contiguous before conversion to numpy
                    tensor = inference_runner.activation[key][0].detach().contiguous()
                    act = tensor.cpu().numpy()  # First sample
                    # Handle multi-dimensional tensors by taking first channel until we get a 2D tensor
                    while act.ndim > 2:
                        act = act[0]
                    important_activations.append((f"Layer {layer_num}", act))
            
            # Try to get fusion result
            if 'fusion_net_output' in inference_runner.activation:
                fusion = inference_runner.activation['fusion_net_output']
                if isinstance(fusion, torch.Tensor):
                    torch.save(fusion.contiguous(), sample_dir / "fusion_output.pt")
                    # Make tensor contiguous before conversion to numpy
                    tensor = fusion[0].detach().contiguous()
                    act = tensor.cpu().numpy()  # First sample
                    # Handle multi-dimensional tensors by taking first channel until we get a 2D tensor
                    while act.ndim > 2:
                        act = act[0]
                    important_activations.append(("Fusion", act))
            
            # Try to get decoder
            if 'decoder' in inference_runner.activation:
                decoder = inference_runner.activation['decoder']
                if isinstance(decoder, torch.Tensor):
                    # Make tensor contiguous before conversion to numpy
                    tensor = decoder[0].detach().contiguous()
                    act = tensor.cpu().numpy()  # First sample
                    # Handle multi-dimensional tensors by taking first channel until we get a 2D tensor
                    while act.ndim > 2:
                        act = act[0]
                    important_activations.append(("Decoder", act))
            
            # Create a simple 2x3 grid with these activations
            if important_activations:
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                for i, (title, act) in enumerate(important_activations):
                    if i < 6:  # Only show up to 6 activations
                        ax = axes[i]
                        im = ax.imshow(act, cmap='viridis')
                        ax.set_title(f"{title}\nShape: {act.shape}")
                        fig.colorbar(im, ax=ax)
                
                # Turn off unused subplots
                for i in range(len(important_activations), 6):
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.savefig(sample_dir / "activation_summary.png", dpi=160)
                plt.close(fig)
                i+1
        
        except Exception as e:
            logger.warning(f"Failed to create summary visualization: {e}")
    
    # Clear activations for next batch
    inference_runner.activation.clear()
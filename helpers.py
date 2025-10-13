import torch
import matplotlib.pyplot as plt
import numpy as np

def _to_2d_image(arr: torch.Tensor) -> np.ndarray:
    """Convert arbitrary N-D tensor into a 2-D numpy array (H, W) for imshow."""
    t = arr.detach().cpu()

    # Squeeze leading batch dims of size 1 (but keep spatial dims)
    while t.dim() > 2 and t.shape[0] == 1:
        t = t.squeeze(0)

    if t.dim() == 5:
        # Reduce all non-spatial dims to get (H, W)
        # Assume the last two dims are spatial (common in vision)
        t = t.mean(dim=tuple(range(0, t.dim() - 2)))  # -> (H, W)
    elif t.dim() == 4:
        # (N/C/T, C/N/T, H, W) -> mean non-spatial dims
        t = t.mean(dim=tuple(range(0, t.dim() - 2)))  # -> (H, W)
    elif t.dim() == 3:
        # (C, H, W) or (H, W, C)
        if t.shape[0] in (1, 3, 4):          # CHW
            if t.shape[0] == 1:
                t = t[0]
            else:
                t = t.float().mean(dim=0)    # luminance-ish
        elif t.shape[-1] in (1, 3, 4):       # HWC
            if t.shape[-1] == 1:
                t = t[..., 0]
            else:
                t = t.float().mean(dim=-1)
        else:
            t = t.mean(dim=0)                # unknown 3D -> reduce first axis
    elif t.dim() == 2:
        pass  # already (H, W)
    elif t.dim() == 1:
        n = int(np.sqrt(t.numel()))
        n = max(1, n)
        t = t[: n * n].reshape(n, n)
    else:
        t = t.unsqueeze(0)

    # Normalize to 0..1
    t = t.float()
    t_min, t_max = t.min(), t.max()
    if (t_max - t_min) > 1e-12:
        t = (t - t_min) / (t_max - t_min)
    else:
        t = torch.zeros_like(t)

    return t.cpu().numpy()

def visualize_available_feature_maps(activation, sample_idx):
    """
    Visualize only the specified feature maps: encoder layers 1-4, fusion_net, and decoder
    Also explicitly show what is used as input to the fusion_net
    
    Args:
        activation: Dictionary containing activation outputs
        sample_idx: Current sample index
    """
    if not activation:
        print("No activations captured!")
        return
    
    # Only visualize specified layers
    selected_keys = [
        'encoder_layer1',
        'encoder_layer2', 
        'encoder_layer3',
        'encoder_layer4',
        'fusion_net_output',
        'decoder'
    ]
    
    # Additionally show pre_fusion_input to visualize input to fusion module
    if 'pre_fusion_input' in activation:
        print("Found pre_fusion_input - explicitly showing what is used as input to the fusion_net")
        visualize_input_to_fusion(activation['pre_fusion_input'], sample_idx)
    
    # Create figure for the selected components
    plt.figure(figsize=(20, 20))
    plt.suptitle(f"Selected Feature Maps for Sample {sample_idx}", fontsize=20)
    
    # Count visualizations for layout
    total_vis = 0
    vis_keys = []
    
    # Only count keys that actually exist in the activation dictionary
    for key in selected_keys:
        if key in activation:
            vis_keys.append(key)
            feat = activation[key]
            if isinstance(feat, list):
                total_vis += min(3, len(feat))
            elif isinstance(feat, dict):
                total_vis += min(3, len(feat))
            else:
                total_vis += 1
    
    # If no selected keys were found, inform the user
    if total_vis == 0:
        print("None of the selected components were found in the activations!")
        plt.text(0.5, 0.5, "No selected layers found in activations", 
                ha='center', va='center', fontsize=16)
        plt.tight_layout()
        plt.savefig(f"/workspace/CIAK/feature_maps_{sample_idx}.png")
        plt.show()
        return
    
    rows = max(1, (total_vis + 2) // 3)  # Ceil division to get enough rows
    cols = min(3, total_vis)
    
    # Visualize in the specified order
    plot_idx = 0
    for name in vis_keys:
        feat = activation[name]
        print(f"Visualizing {name} of type {type(feat)}")
        
        try:
            if isinstance(feat, list):
                # Handle list of tensors
                for i, tensor in enumerate(feat[:3]):  # Show at most 3 items
                    if i >= 3: 
                        break
                    plot_idx += 1
                    visualize_single_tensor(tensor, f"{name}-{i}", 
                                          plot_idx, rows, cols)
            elif isinstance(feat, dict):
                # Handle dictionary of tensors
                for i, (key, tensor) in enumerate(list(feat.items())[:3]):  # Show at most 3 items
                    plot_idx += 1
                    visualize_single_tensor(tensor, f"{name}-{key}", 
                                          plot_idx, rows, cols)
            else:
                # Handle single tensor or other type
                plot_idx += 1
                visualize_single_tensor(feat, name, plot_idx, rows, cols)
        except Exception as e:
            print(f"Error visualizing {name}: {e}")
    
    plt.tight_layout()
    plt.savefig(f"/workspace/CIAK/feature_maps_{sample_idx}.png")
    plt.show()
    
def visualize_input_images(inputs, sample_idx):
    """
    Specifically visualize the input images or tensors that are going into the model
    
    Args:
        inputs: The input tensor or list of tensors
        sample_idx: Current sample index
    """
    plt.figure(figsize=(20, 10))
    plt.suptitle(f"Original Inputs for Sample {sample_idx}", fontsize=20)
    
    # Debug info
    print(f"Input type: {type(inputs)}")
    if isinstance(inputs, list):
        print(f"Input list length: {len(inputs)}")
        for i, item in enumerate(inputs):
            print(f"Item {i} type: {type(item)}")
            if isinstance(item, torch.Tensor):
                print(f"Item {i} shape: {item.shape}")
    
    # Try to find camera images or other interpretable inputs
    if isinstance(inputs, list):
        # We might have multiple inputs
        num_inputs = min(len(inputs), 9)  # Limit to 9 inputs
        rows = int(np.ceil(num_inputs / 3))
        cols = min(3, num_inputs)
        
        for i, input_item in enumerate(inputs[:num_inputs]):
            plt.subplot(rows, cols, i+1)
            
            # Try to interpret what this input is
            if isinstance(input_item, torch.Tensor):
                tensor = input_item.detach().cpu()
                
                # Print more debug info about the tensor
                print(f"Processing tensor {i} with shape {tensor.shape}")
                
                try:
                    # Check if this looks like an image
                    if len(tensor.shape) >= 4:  # Batch dimension + channels + H + W
                        # Extract the first item if it's a batch
                        img = tensor[0] if tensor.shape[0] > 0 else tensor
                        print(f"  After batch extraction: shape {img.shape}")
                        
                        # Find the channel dimension - typically dim 0 or 1
                        if img.shape[0] in [1, 3, 4]:  # Channel first
                            img = img.permute(1, 2, 0).numpy()  # CHW -> HWC
                            print(f"  Permuted CHW->HWC: shape {img.shape}")
                        elif img.shape[-1] in [1, 3, 4]:  # Channel last
                            img = img.numpy()
                            print(f"  Channel-last format: shape {img.shape}")
                        else:
                            # Not a standard image format - try different approaches
                            print(f"  Non-standard channels: {img.shape}")
                            # Just take the first few slices or mean across first dimension
                            img = img.mean(dim=0).numpy()
                            print(f"  After averaging: shape {img.shape}")
                    elif len(tensor.shape) == 3:
                        # Could be CHW format without batch
                        if tensor.shape[0] in [1, 3, 4]:
                            img = tensor.permute(1, 2, 0).numpy()
                            print(f"  3D tensor permuted: shape {img.shape}")
                        else:
                            img = tensor.numpy()  # Assume it's already in a displayable format
                            print(f"  3D tensor as is: shape {img.shape}")
                    elif len(tensor.shape) == 2:
                        # 2D tensor can be displayed directly
                        img = tensor.numpy()
                        print(f"  2D tensor: shape {img.shape}")
                    else:
                        # 1D or other - use as plot
                        plt.plot(tensor.numpy().flatten()[:100])
                        plt.title(f"Input {i} (Non-image)\nShape: {tensor.shape}")
                        continue
                    
                    # Normalize for visualization if it's an image
                    if img.max() > 1.0 or img.min() < 0.0:
                        img = (img - img.min()) / (img.max() - img.min() + 1e-10)
                    
                    # If image has too many dimensions, take a slice or projection
                    if len(img.shape) > 3:
                        print(f"  Too many dimensions: {img.shape}, taking mean")
                        img = np.mean(img, axis=0)
                    
                    # For 3D arrays, if the last dimension is too large, it's not an image
                    if len(img.shape) == 3 and img.shape[2] > 4:
                        print(f"  Last dimension too large: {img.shape}, taking first channel")
                        img = img[:, :, 0]
                    
                    plt.imshow(img, cmap='viridis' if len(img.shape) == 2 else None)
                    plt.title(f"Input {i}\nShape: {tensor.shape}")
                    
                except Exception as e:
                    print(f"Error processing tensor {i}: {e}")
                    # Fallback - try to show as much information as possible
                    plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                    if len(tensor.shape) <= 2:
                        try:
                            plt.imshow(tensor.numpy(), cmap='viridis')
                        except:
                            plt.text(0.5, 0.3, f"Shape: {tensor.shape}", ha='center', va='center')
            else:
                plt.text(0.5, 0.5, f"Non-tensor input: {type(input_item)}", 
                        ha='center', va='center')
                plt.title(f"Input {i}")
            
            plt.axis('on')
    
    elif isinstance(inputs, dict):
        # Dict of inputs - show a few key items
        keys = list(inputs.keys())[:9]  # Limit to 9 keys
        rows = int(np.ceil(len(keys) / 3))
        cols = min(3, len(keys))
        
        for i, key in enumerate(keys):
            plt.subplot(rows, cols, i+1)
            # Similar processing as above
            if isinstance(inputs[key], torch.Tensor):
                # Similar tensor visualization as above
                pass
            else:
                plt.text(0.5, 0.5, f"Non-tensor input: {type(inputs[key])}", 
                        ha='center', va='center')
            
            plt.title(f"Input '{key}'")
    
    elif isinstance(inputs, torch.Tensor):
        # Single tensor input - might be a batch of images
        tensor = inputs.detach().cpu()
        
        if len(tensor.shape) >= 4:  # Likely a batch of images
            num_images = min(tensor.shape[0], 9)
            rows = int(np.ceil(num_images / 3))
            cols = min(3, num_images)
            
            for i in range(num_images):
                plt.subplot(rows, cols, i+1)
                img = tensor[i]
                
                # Handle channel dimension as above
                if img.shape[0] in [1, 3, 4]:  # Channel first
                    img = img.permute(1, 2, 0).numpy()
                elif img.shape[-1] in [1, 3, 4]:  # Channel last
                    img = img.numpy()
                else:
                    img = img.mean(0).numpy()  # Take mean across channels
                
                if img.max() > 1.0 or img.min() < 0.0:
                    img = (img - img.min()) / (img.max() - img.min() + 1e-10)
                
                plt.imshow(img)
                plt.title(f"Image {i}, Shape: {img.shape}")
        else:
            # Not image-like
            plt.imshow(tensor.numpy().reshape(tensor.shape[-2:]))
            plt.title(f"Input tensor, Shape: {tensor.shape}")
    
    plt.tight_layout()
    plt.savefig(f"/workspace/CIAK/input_images_{sample_idx}.png")
    plt.show()

def visualize_single_tensor(tensor, name, plot_idx, rows, cols):
    """Helper function to visualize a single tensor"""
    # Create subplot
    plt.subplot(rows, cols, plot_idx)
    
    try:
        print(f"Visualizing tensor {name} with type {type(tensor)}")
        
        # Determine what to visualize based on tensor shape
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu()
            
            print(f"  Tensor shape: {tensor.shape}")
            
            # Check for 1D tensors - these can't be visualized as images
            if len(tensor.shape) == 1 or (len(tensor.shape) == 2 and (tensor.shape[0] == 1 or tensor.shape[1] == 1)):
                # For 1D tensors, use a line plot instead of imshow
                plt.plot(tensor.numpy().flatten())
                plt.title(f"{name}\nShape: {tensor.shape} (1D data)")
                return
            
            # Handle different tensor shapes
            try:
                if len(tensor.shape) == 5:  # B, L/max_cav, C, H, W or similar
                    # Take first batch and first element from second dimension
                    if tensor.shape[1] > 0:
                        feature = tensor[0, 0]
                    else:
                        feature = tensor[0]
                    print(f"  5D tensor reduced to: {feature.shape}")
                elif len(tensor.shape) == 4:  # B, C, H, W
                    feature = tensor[0]  # First batch
                    print(f"  4D tensor reduced to: {feature.shape}")
                elif len(tensor.shape) == 3:  # C, H, W or B, H, W
                    if tensor.shape[0] <= 64:  # Likely channel dimension if small
                        feature = tensor
                        print(f"  3D tensor kept as: {feature.shape}")
                    else:
                        # If first dim is large, it might be batch, take first item
                        feature = tensor[0:1] if tensor.shape[0] > 64 else tensor
                        print(f"  3D tensor (batch?) reduced to: {feature.shape}")
                elif len(tensor.shape) == 2:  # H, W
                    feature = tensor
                    print(f"  2D tensor kept as: {feature.shape}")
                else:
                    # For other shapes, try to reshape or sample
                    print(f"  Complex shape for {name}: {tensor.shape}")
                    if tensor.numel() > 1000000:  # If tensor is very large
                        # Sample some values
                        indices = torch.randint(0, tensor.numel(), (1000,))
                        sampled = tensor.reshape(-1)[indices]
                        plt.plot(sampled.numpy())
                        plt.title(f"{name}\nShape: {tensor.shape}\n(Sampled values)")
                        return
                    else:
                        # Try to reshape to 2D if possible
                        try:
                            h = int(np.sqrt(tensor.numel()))
                            w = tensor.numel() // h
                            feature = tensor.reshape(h, w)
                            print(f"  Reshaped to 2D: {feature.shape}")
                        except:
                            # Otherwise flatten and plot
                            plt.plot(tensor.flatten().numpy()[:1000])
                            plt.title(f"{name}\nShape: {tensor.shape}\n(flattened)")
                            return
                
                # Convert to numpy
                feature = feature.numpy()
                
                # For feature maps with channels, take mean or first few channels
                if len(feature.shape) > 2:
                    if feature.shape[0] == 3:  # Could be RGB
                        # If shape is [3, H, W], try to display as RGB
                        rgb = np.transpose(feature, (1, 2, 0))
                        # Normalize for visualization
                        if rgb.max() > 1.0 or rgb.min() < 0.0:
                            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10)
                        plt.imshow(rgb)
                        plt.title(f"{name}\nShape: {tensor.shape}\n(RGB visualization)")
                        return
                    elif feature.shape[0] <= 16:
                        # For few channels, visualize each channel separately in a grid
                        channel_rows = int(np.ceil(np.sqrt(feature.shape[0])))
                        channel_cols = int(np.ceil(feature.shape[0] / channel_rows))
                        
                        # Create a grid of channel visualizations
                        grid_img = np.zeros((feature.shape[1] * channel_rows, 
                                           feature.shape[2] * channel_cols))
                        
                        for c in range(feature.shape[0]):
                            row = c // channel_cols
                            col = c % channel_cols
                            channel = feature[c]
                            # Normalize channel
                            if channel.max() != channel.min():
                                channel = (channel - channel.min()) / (channel.max() - channel.min())
                            grid_img[row*feature.shape[1]:(row+1)*feature.shape[1], 
                                     col*feature.shape[2]:(col+1)*feature.shape[2]] = channel
                        
                        plt.imshow(grid_img, cmap='viridis')
                        plt.title(f"{name}\nShape: {tensor.shape}\n(channel grid)")
                        return
                    else:
                        # For many channels, take mean across channels
                        feature = np.mean(feature, axis=0)
                
                # Check if resulting feature is 2D (can be visualized as image)
                if len(feature.shape) == 2:
                    # Normalize for better visualization
                    if np.size(feature) > 0:  # Make sure it's not empty
                        if np.max(feature) != np.min(feature):
                            feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
                    
                    plt.imshow(feature, cmap='viridis')
                    plt.title(f"{name}\nShape: {tensor.shape}")
                    plt.colorbar()
                    plt.axis('on')
                else:
                    # Fallback for non-2D data
                    plt.plot(feature.flatten()[:100])  # Plot first 100 values
                    plt.title(f"{name}\nShape: {tensor.shape} (flattened)")
            except Exception as e:
                print(f"  Error processing tensor for visualization: {e}")
                # Fallback to simple visualization
                if tensor.numel() < 10000:
                    plt.plot(tensor.flatten().numpy()[:1000])
                    plt.title(f"{name}\nShape: {tensor.shape}\n(Error: {str(e)})")
                else:
                    indices = torch.randint(0, tensor.numel(), (1000,))
                    sampled = tensor.reshape(-1)[indices]
                    plt.plot(sampled.numpy())
                    plt.title(f"{name}\nShape: {tensor.shape}\n(Sampled, Error: {str(e)})")
        
        elif isinstance(tensor, np.ndarray):
            # Similar handling for numpy arrays
            print(f"  NumPy array shape: {tensor.shape}")
            
            if len(tensor.shape) == 1:
                plt.plot(tensor)
                plt.title(f"{name}\nShape: {tensor.shape} (1D data)")
            else:
                # Try to convert to 2D image
                if len(tensor.shape) > 2:
                    if tensor.shape[0] == 3 and len(tensor.shape) == 3:
                        # Could be RGB
                        rgb = np.transpose(tensor, (1, 2, 0))
                        if rgb.max() > 1.0 or rgb.min() < 0.0:
                            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-10)
                        plt.imshow(rgb)
                        plt.title(f"{name}\nShape: {tensor.shape}\n(RGB visualization)")
                        return
                    feature = np.mean(tensor, axis=0)
                else:
                    feature = tensor
                
                if len(feature.shape) == 2:
                    if np.max(feature) != np.min(feature):
                        feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
                    plt.imshow(feature, cmap='viridis')
                    plt.title(f"{name}\nShape: {tensor.shape}")
                    plt.colorbar()
                else:
                    plt.plot(feature.flatten()[:100])
                    plt.title(f"{name}\nShape: {tensor.shape} (flattened)")
        else:
            plt.text(0.5, 0.5, f"Cannot visualize: {type(tensor)}", 
                    ha='center', va='center')
            plt.title(f"{name}\nType: {type(tensor)}")
    
    except Exception as e:
        print(f"Error in visualize_single_tensor for {name}: {e}")
        plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        plt.title(f"{name}")

def visualize_input_to_fusion(inputs, sample_idx):
    """
    Visualize the input tensors going into fusion_net.
    Handles shapes like (B, T, C, H, W) and (B, H, W, 1, T) explicitly,
    and falls back to a generic reducer for anything else.
    """
    plt.figure(figsize=(20, 15))
    plt.suptitle(f"Input to Fusion Module for Sample {sample_idx}", fontsize=20)

    print(f"Fusion input type: {type(inputs)}")
    if isinstance(inputs, list):
        print(f"Fusion input list length: {len(inputs)}")
        for i, item in enumerate(inputs):
            print(f"Fusion input item {i} type: {type(item)}")
            if isinstance(item, torch.Tensor):
                print(f"Fusion input item {i} shape: {item.shape}")

        num_inputs = min(len(inputs), 9)
        rows = int(np.ceil(num_inputs / 3))
        cols = min(3, num_inputs)

        for i, input_item in enumerate(inputs[:num_inputs]):
            plt.subplot(rows, cols, i + 1)

            if isinstance(input_item, torch.Tensor):
                t = input_item.detach().cpu()
                print(f"Visualizing fusion input tensor {i} with shape {t.shape}")

                try:
                    # Case 1: (B, T, C, H, W) e.g., [1, 5, 128, 32, 32]
                    if t.dim() == 5 and t.shape[0] == 1 and t.shape[-2:] == (32, 32):
                        # pick timestep 0 safely
                        timestep = 0 if t.shape[1] > 0 else 0
                        img2d = _to_2d_image(t[0, timestep])  # (C,H,W)->(H,W)
                        plt.imshow(img2d, cmap='viridis')
                        plt.title(f"Fusion Input {i} (B,T,C,H,W)\nT={timestep}  {tuple(t.shape)}")
                        plt.colorbar()

                    # Case 2: (B, H, W, 1, T) e.g., [1, 32, 32, 1, 5]
                    elif t.dim() == 5 and t.shape[0] == 1 and t.shape[3] == 1:
                        timestep = 0 if t.shape[-1] > 0 else 0
                        # (H, W)
                        img2d = t[0, :, :, 0, timestep]
                        img2d = _to_2d_image(img2d)
                        plt.imshow(img2d, cmap='viridis')
                        plt.title(f"Fusion Input {i} (B,H,W,1,T)\nT={timestep}  {tuple(t.shape)}")
                        plt.colorbar()

                    else:
                        # Generic fallback: collapse to 2-D
                        img2d = _to_2d_image(t)
                        plt.imshow(img2d, cmap='viridis')
                        plt.title(f"Fusion Input {i}\nCollapsed -> {img2d.shape} from {tuple(t.shape)}")
                        plt.colorbar()

                except Exception as e:
                    print(f"Error visualizing fusion input {i}: {e}")
                    plt.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
            else:
                plt.text(0.5, 0.5, f"Non-tensor: {type(input_item)}", ha='center', va='center')
                plt.title(f"Fusion Input {i}")

    elif isinstance(inputs, dict):
        keys = list(inputs.keys())[:9]
        rows = int(np.ceil(len(keys) / 3))
        cols = min(3, len(keys))
        for i, key in enumerate(keys):
            plt.subplot(rows, cols, i + 1)
            val = inputs[key]
            if isinstance(val, torch.Tensor):
                img2d = _to_2d_image(val)
                plt.imshow(img2d, cmap='viridis')
                plt.title(f"Fusion Input '{key}'\n{tuple(val.shape)}")
                plt.colorbar()
            else:
                plt.text(0.5, 0.5, f"Non-tensor: {type(val)}", ha='center', va='center')
                plt.title(f"Fusion Input '{key}'")

    elif isinstance(inputs, torch.Tensor):
        img2d = _to_2d_image(inputs)
        plt.imshow(img2d, cmap='viridis')
        plt.title(f"Fusion Input\n{tuple(inputs.shape)}")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig(f"/workspace/CIAK/fusion_inputs_{sample_idx}.png")
    plt.show()
    
def _register_hooks(self) -> None:
    """Register forward hooks to capture model activations."""
    logger.debug("Registering model hooks for exploration mode")
    
    def get_activation(name):
        def hook(module, input_, output):
            if isinstance(output, list):
                self.activation[name] = [o.detach() if isinstance(o, torch.Tensor) else o for o in output]
            elif isinstance(output, dict):
                self.activation[name] = {k: v.detach() if isinstance(v, torch.Tensor) else v 
                                        for k, v in output.items()}
            elif isinstance(output, torch.Tensor):
                self.activation[name] = output.detach()
            else:
                self.activation[name] = output
                logger.warning(f"{name} output type {type(output)} not tensor/list/dict")
        return hook
    
    try:
        if hasattr(self.model, 'encoder'):
            self.hooks.append(self.model.encoder.register_forward_hook(get_activation('encoder')))
            if hasattr(self.model.encoder, 'encoder'):
                for layer in [1, 2, 3, 4]:
                    ln = f'layer{layer}'
                    if hasattr(self.model.encoder.encoder, ln):
                        self.hooks.append(getattr(self.model.encoder.encoder, ln).register_forward_hook(
                            get_activation(f'encoder_{ln}')))
    except Exception as e:
        logger.warning(f"Failed to hook encoder: {e}")
        
    try:
        if hasattr(self.model, 'fusion_net'):
            def fusion_input_hook(module, input_):
                self.activation['pre_fusion_input'] = [i.detach() if isinstance(i, torch.Tensor) else i for i in input_]
                return None
            self.hooks.append(self.model.fusion_net.register_forward_pre_hook(fusion_input_hook))
            self.hooks.append(self.model.fusion_net.register_forward_hook(get_activation('fusion_net_output')))
    except Exception as e:
        logger.warning(f"Failed to hook fusion_net: {e}")
        
    try:
        if hasattr(self.model, 'decoder'):
            self.hooks.append(self.model.decoder.register_forward_hook(get_activation('decoder')))
    except Exception as e:
        logger.warning(f"Failed to hook decoder: {e}")
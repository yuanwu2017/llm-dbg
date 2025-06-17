import torch
import argparse
import os
import sys
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
import numpy as np
import psutil
from collections import defaultdict

def get_size_str(num_bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"

def count_parameters(model):
    """Count trainable and non-trainable parameters"""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    return {
        "trainable": trainable_params,
        "non_trainable": non_trainable_params,
        "total": total_params,
        "total_size": get_size_str(total_params * 4)  # Assuming float32
    }

def extract_layer_number(name):
    """Extract layer number from parameter name"""
    if 'layers.' in name:
        parts = name.split('layers.')
        if len(parts) > 1:
            layer_part = parts[1].split('.')[0]
            if layer_part.isdigit():
                return int(layer_part)
    return None

def get_model_state_dict(model_path):
    """Get state_dict from a model file or HF model"""
    try:
        # Try to load directly from file
        if os.path.isfile(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                return state_dict["state_dict"]
            return state_dict
        # HF model checkpoint might have sharded files
        elif os.path.isdir(model_path):
            checkpoint_files = [f for f in os.listdir(model_path) 
                              if f.startswith("pytorch_model") and f.endswith(".bin")]
            if checkpoint_files:
                # Load first shard for inspection
                first_shard = os.path.join(model_path, checkpoint_files[0])
                state_dict = torch.load(first_shard, map_location="cpu")
                print(f"Loaded first shard from {first_shard} for weight name inspection")
                return state_dict
    except Exception as e:
        print(f"Unable to load state_dict directly from file: {e}")
    
    # Return empty dict if no direct loading possible
    return {}

def analyze_weight_loading_pattern(state_dict):
    """Analyze whether weights are stored with or without .weight suffix"""
    if not state_dict:
        return "Unable to determine - no state dict available"
    
    # Find sample module names that might be in layers
    layer_weights = []
    direct_weights = []
    
    for key in state_dict.keys():
        # Look for patterns like "layers.0.self_attn.q_proj.weight"
        if 'layers' in key and 'weight' in key:
            layer_weights.append(key)
        # Look for patterns like "layers.0.self_attn.q_proj" (without weight)
        elif 'layers' in key and not any(suffix in key for suffix in ['.weight', '.bias']):
            direct_weights.append(key)
    
    # Print examples of each pattern
    print("\n===== WEIGHT LOADING PATTERN ANALYSIS =====")
    if layer_weights:
        print("Found weights with '.weight' suffix (e.g., model.layers.0.self_attn.q_proj.weight):")
        for example in layer_weights[:3]:
            print(f"  {example}")
        if len(layer_weights) > 3:
            print(f"  ... and {len(layer_weights) - 3} more")
    
    if direct_weights:
        print("\nFound weights without '.weight' suffix (e.g., model.layers.0.self_attn.q_proj):")
        for example in direct_weights[:3]:
            print(f"  {example}")
        if len(direct_weights) > 3:
            print(f"  ... and {len(direct_weights) - 3} more")
    
    # Analyze common prefixes
    if state_dict:
        prefixes = defaultdict(int)
        for key in state_dict.keys():
            # Extract prefix before the first dot
            parts = key.split('.')
            if parts:
                prefix = parts[0]
                prefixes[prefix] += 1
        
        print("\nCommon prefixes in weight names:")
        for prefix, count in sorted(prefixes.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {prefix}: {count} weights")
    
    # Determine the primary pattern
    if len(layer_weights) > len(direct_weights):
        return "Most weights use '.weight' suffix pattern (access as module.weight)"
    elif len(direct_weights) > 0:
        return "Many weights use direct tensor pattern (access module directly)"
    else:
        return "Standard pattern with '.weight' suffix detected"

def print_model_structure(model, config, state_dict=None):
    """Print model structure and parameters details with layer deduplication"""
    print("\n===== MODEL STRUCTURE =====")
    print(model)
    
    print("\n===== MODEL CONFIGURATION =====")
    for key, value in config.to_dict().items():
        if not isinstance(value, (dict, list)):
            print(f"{key}: {value}")
    
    print("\n===== MODULE PARAMETERS INFORMATION =====")
    
    # Print state_dict structure if available
    if state_dict:
        weight_keys = list(state_dict.keys())
        print(f"\n===== WEIGHT FILE STRUCTURE =====")
        print(f"Total weights in file: {len(weight_keys)}")
        
        # Show first 10 weight keys as examples
        if weight_keys:
            print("Example weight names:")
            for i, key in enumerate(weight_keys[:10]):
                print(f"  {key}")
            if len(weight_keys) > 10:
                print(f"  ... and {len(weight_keys) - 10} more")
    
    # Organize modules by layer
    layer_modules = defaultdict(list)
    non_layer_modules = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            layer_num = extract_layer_number(name)
            if layer_num is not None:
                layer_modules[layer_num].append((name, module))
            else:
                non_layer_modules.append((name, module))
    
    # Process non-layer modules
    for name, module in non_layer_modules:
        has_weight = hasattr(module, 'weight') and module.weight is not None
        has_bias = hasattr(module, 'bias') and module.bias is not None
        
        weight_shape = module.weight.shape if has_weight else None
        bias_shape = module.bias.shape if has_bias else None
        
        # Try to find matching weights in state_dict
        weight_file_name = None
        bias_file_name = None
        if state_dict:
            # Check for direct match
            if name + '.weight' in state_dict:
                weight_file_name = name + '.weight'
            elif name.replace('.', '_') + '.weight' in state_dict:
                weight_file_name = name.replace('.', '_') + '.weight'
            
            if name + '.bias' in state_dict:
                bias_file_name = name + '.bias'
            elif name.replace('.', '_') + '.bias' in state_dict:
                bias_file_name = name.replace('.', '_') + '.bias'
            
            # If not found, search for similar names
            if not weight_file_name and has_weight:
                similar_keys = [k for k in state_dict.keys() if k.endswith('weight') and name.split('.')[-1] in k]
                if similar_keys:
                    weight_file_name = f"Possible matches: {', '.join(similar_keys[:3])}"
                    if len(similar_keys) > 3:
                        weight_file_name += f" and {len(similar_keys) - 3} more"
        
        print(f"\nModule: {name}")
        print(f"  Type: {module.__class__.__name__}")
        print(f"  Has weights: {has_weight}, Shape: {weight_shape}")
        if weight_file_name:
            print(f"  Weight file name: {weight_file_name}")
        print(f"  Has bias: {has_bias}, Shape: {bias_shape}")
        if bias_file_name:
            print(f"  Bias file name: {bias_file_name}")
    
    # Check if we have layers
    if layer_modules:
        # Get the shapes of parameters in each layer
        layer_signatures = {}
        for layer_num, modules in layer_modules.items():
            signature = []
            for name, module in modules:
                if hasattr(module, 'weight') and module.weight is not None:
                    signature.append((name.split('.')[-1], 'weight', tuple(module.weight.shape)))
                if hasattr(module, 'bias') and module.bias is not None:
                    signature.append((name.split('.')[-1], 'bias', tuple(module.bias.shape)))
            layer_signatures[layer_num] = tuple(sorted(signature))
        
        # Group layers by signature
        signature_to_layers = defaultdict(list)
        for layer_num, signature in layer_signatures.items():
            signature_to_layers[signature].append(layer_num)
        
        # Print information for each signature group
        for signature, layer_nums in signature_to_layers.items():
            first_layer = min(layer_nums)
            print(f"\n--- Layer Group (showing layer {first_layer} as representative) ---")
            if len(layer_nums) > 1:
                print(f"  Identical layers: {sorted(layer_nums)}")
            
            # Print details of the first layer in each group
            for name, module in layer_modules[first_layer]:
                has_weight = hasattr(module, 'weight') and module.weight is not None
                has_bias = hasattr(module, 'bias') and module.bias is not None
                
                weight_shape = module.weight.shape if has_weight else None
                bias_shape = module.bias.shape if has_bias else None
                
                # Try to find matching weights in state_dict
                weight_file_name = None
                bias_file_name = None
                if state_dict:
                    # For each layer, try different naming patterns
                    layer_patterns = [
                        name + '.weight',
                        name.replace('.', '_') + '.weight',
                        f"layers.{first_layer}.{name.split('.')[-1]}.weight"
                    ]
                    
                    for pattern in layer_patterns:
                        if pattern in state_dict:
                            weight_file_name = pattern
                            break
                    
                    if not weight_file_name and has_weight:
                        # Check for layer-specific patterns
                        layer_part = f"layers.{first_layer}"
                        similar_keys = [k for k in state_dict.keys() if layer_part in k and k.endswith('weight')]
                        if similar_keys:
                            weight_file_name = f"Possible matches: {', '.join(similar_keys[:3])}"
                            if len(similar_keys) > 3:
                                weight_file_name += f" and {len(similar_keys) - 3} more"
                
                print(f"\nModule: {name}")
                print(f"  Type: {module.__class__.__name__}")
                print(f"  Has weights: {has_weight}, Shape: {weight_shape}")
                if weight_file_name:
                    print(f"  Weight file name: {weight_file_name}")
                print(f"  Has bias: {has_bias}, Shape: {bias_shape}")
                if bias_file_name:
                    print(f"  Bias file name: {bias_file_name}")

    # ===== DIRECT PARAMETERS SECTION =====
    print("\n===== DIRECT PARAMETERS (not following weight/bias pattern) =====")
    
    # First, collect all direct parameters
    all_direct_params = {}
    for name, module in model.named_modules():
        direct_params = []
        for param_name, param in module._parameters.items():
            if param is not None and param_name not in ['weight', 'bias']:
                if param.shape:  # Only include non-empty parameters
                    direct_params.append((param_name, param.shape))
        
        if direct_params:
            all_direct_params[name] = direct_params
    
    # Separate layer and non-layer modules with direct parameters
    layer_direct_params = {}
    non_layer_direct_params = {}
    
    for name, direct_params in all_direct_params.items():
        layer_num = extract_layer_number(name)
        if layer_num is not None:
            layer_direct_params[(layer_num, name)] = direct_params
        else:
            non_layer_direct_params[name] = direct_params
    
    # Process non-layer direct parameters
    for name, direct_params in non_layer_direct_params.items():
        print(f"\nModule: {name}")
        for param_name, shape in direct_params:
            print(f"  Direct parameter: {param_name}, Shape: {shape}")
    
    # Group layer direct parameters by signature
    if layer_direct_params:
        # Group layer parameters by common patterns
        layer_signatures = {}
        
        for (layer_num, full_name), params in layer_direct_params.items():
            # Create a normalized signature without layer number
            module_name = full_name
            if 'layers.' in module_name:
                parts = module_name.split('.')
                # Remove the layer number part but keep 'layers' keyword
                module_signature = '.'.join([parts[0]] + parts[2:])
            else:
                module_signature = module_name
            
            # Create a tuple of parameter names and shapes for signature
            param_signature = tuple(sorted([(p_name, tuple(shape)) for p_name, shape in params]))
            signature = (module_signature, param_signature)
            
            if layer_num not in layer_signatures:
                layer_signatures[layer_num] = []
            layer_signatures[layer_num].append((signature, full_name, params))
        
        # Group signatures across layers
        signature_to_layers = defaultdict(list)
        for layer_num, signatures in layer_signatures.items():
            for signature, full_name, params in signatures:
                signature_to_layers[signature].append((layer_num, full_name, params))
        
        # Print one representative for each signature group
        for signature, instances in signature_to_layers.items():
            # Sort by layer number
            instances.sort()
            first_instance = instances[0]
            first_layer = first_instance[0]
            module_name = first_instance[1]
            params = first_instance[2]
            
            # Get all layers that have this same signature
            layers_with_sig = [layer for layer, _, _ in instances]
            
            print(f"\n--- Direct Parameters Module (showing layer {first_layer} as representative) ---")
            if len(layers_with_sig) > 1:
                print(f"  Identical in layers: {sorted(layers_with_sig)}")
            
            print(f"Module: {module_name}")
            for param_name, shape in params:
                print(f"  Direct parameter: {param_name}, Shape: {shape}")

def estimate_memory_requirements(model):
    """Estimate memory requirements for model loading and inference"""
    # Model parameters memory
    param_count = count_parameters(model)
    
    print("\n===== MEMORY REQUIREMENTS ESTIMATION =====")
    print(f"Number of model parameters: {param_count['total']:,}")
    print(f"Model parameters memory: {param_count['total_size']}")
    
    # For FP16/BF16 models
    print(f"If using FP16/BF16 precision: {get_size_str(param_count['total'] * 2)}")
    
    # For INT8 quantized models
    print(f"If using INT8 quantization: {get_size_str(param_count['total'])}")
    
    # Memory during inference
    print("\nAdditional memory required during inference:")
    
    # Rough estimate based on model size
    if hasattr(model.config, 'n_positions') and hasattr(model.config, 'hidden_size'):
        seq_len = model.config.n_positions
        hidden_size = model.config.hidden_size
        batch_size = 1
        
        # Estimate KV cache size (this is a rough estimate)
        if hasattr(model.config, 'n_layer'):
            n_layers = model.config.n_layer
        elif hasattr(model.config, 'num_hidden_layers'):
            n_layers = model.config.num_hidden_layers
        else:
            n_layers = 12  # default assumption
            
        kv_cache_size = 2 * batch_size * seq_len * n_layers * hidden_size * 4  # 4 bytes per float
        
        print(f"KV cache estimate (seq_len={seq_len}): {get_size_str(kv_cache_size)}")
        print(f"Total inference memory estimate: {get_size_str(param_count['total'] * 4 + kv_cache_size)}")
    else:
        # Very rough estimation
        print(f"Estimated additional memory for inference: {get_size_str(param_count['total'] * 4 * 0.5)} - {get_size_str(param_count['total'] * 4 * 1.5)}")
    
    # RAM requirements
    ram = psutil.virtual_memory()
    print(f"\nSystem memory status:")
    print(f"Total memory: {get_size_str(ram.total)}")
    print(f"Available memory: {get_size_str(ram.available)}")
    
    if param_count['total'] * 4 > ram.available:
        print("\n⚠️ WARNING: Model parameters may exceed available memory!")

def main():
    parser = argparse.ArgumentParser(description="Print model structure and memory requirements")
    parser.add_argument("--model", type=str, required=True, 
                        help="Path to the model or model identifier from huggingface.co/models")
    parser.add_argument("--model_type", type=str, default="causal_lm",
                        choices=["base", "causal_lm"], 
                        help="Type of model to load")
    args = parser.parse_args()
    
    # Try to load state dict directly if it's a local file
    state_dict = get_model_state_dict(args.model)
    
    # Load model configuration
    print(f"Loading model configuration: {args.model}")
    config = AutoConfig.from_pretrained(args.model)
    
    # Record memory before model loading
    initial_memory = psutil.Process(os.getpid()).memory_info().rss
    
    # Load model on CPU
    print("Loading model to CPU...")
    try:
        if args.model_type == "base":
            model = AutoModel.from_pretrained(args.model, config=config)
        else:  # causal_lm
            model = AutoModelForCausalLM.from_pretrained(args.model, config=config)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    
    # Record memory after model loading
    final_memory = psutil.Process(os.getpid()).memory_info().rss
    memory_used = final_memory - initial_memory
    
    print(f"\nMemory required for model loading: {get_size_str(memory_used)}")
    
    if state_dict:
        weight_pattern = analyze_weight_loading_pattern(state_dict)
        print(f"\nWeight loading pattern: {weight_pattern}")
    
    # Print model structure and parameters
    print_model_structure(model, config, state_dict)
    
    # Analyze weight loading pattern
    analyze_weight_loading_pattern(state_dict)
    
    # Estimate memory requirements
    estimate_memory_requirements(model)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()

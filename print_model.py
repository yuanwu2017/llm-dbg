import torch
import argparse
import os
import sys
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
import numpy as np
import psutil

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

def print_model_structure(model, config):
    """Print model structure and parameters details"""
    print("\n===== MODEL STRUCTURE =====")
    print(model)
    
    print("\n===== MODEL CONFIGURATION =====")
    for key, value in config.to_dict().items():
        if not isinstance(value, (dict, list)):
            print(f"{key}: {value}")
    
    print("\n===== MODULE PARAMETERS INFORMATION =====")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            has_weight = hasattr(module, 'weight') and module.weight is not None
            has_bias = hasattr(module, 'bias') and module.bias is not None
            
            weight_shape = module.weight.shape if has_weight else None
            bias_shape = module.bias.shape if has_bias else None
            
            print(f"\nModule: {name}")
            print(f"  Type: {module.__class__.__name__}")
            print(f"  Has weights: {has_weight}, Shape: {weight_shape}")
            print(f"  Has bias: {has_bias}, Shape: {bias_shape}")

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
    parser.add_argument("--model_name_or_path", type=str, required=True, 
                        help="Path to the model or model identifier from huggingface.co/models")
    parser.add_argument("--model_type", type=str, default="causal_lm",
                        choices=["base", "causal_lm"], 
                        help="Type of model to load")
    args = parser.parse_args()
    
    # Load model configuration
    print(f"Loading model configuration: {args.model_name_or_path}")
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    
    # Record memory before model loading
    initial_memory = psutil.Process(os.getpid()).memory_info().rss
    
    # Load model on CPU
    print("Loading model to CPU...")
    try:
        if args.model_type == "base":
            model = AutoModel.from_pretrained(args.model_name_or_path, config=config)
        else:  # causal_lm
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config)
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    
    # Record memory after model loading
    final_memory = psutil.Process(os.getpid()).memory_info().rss
    memory_used = final_memory - initial_memory
    
    print(f"\nMemory required for model loading: {get_size_str(memory_used)}")
    
    # Print model structure and parameters
    print_model_structure(model, config)
    
    # Estimate memory requirements
    estimate_memory_requirements(model)
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()
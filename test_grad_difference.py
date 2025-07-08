import torch
from transformers import AutoModelForCausalLM

def test_original_model_grad_difference():
    """Test original model output differences with and without torch.no_grad()"""
    
    # Test configuration
    model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"
    torch_device = "xpu" #infer_device()
    
    print(f"=== Testing Original Model Grad Differences ===")
    print(f"Using device: {torch_device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"XPU available: {hasattr(torch, 'xpu') and torch.xpu.is_available() if hasattr(torch, 'xpu') else False}")
    
    # Test input data
    input_ids = torch.LongTensor([[1, 1, 1], [2, 1, 2]]).to(torch_device)
    attention_mask = torch.LongTensor([[1, 1, 1], [1, 0, 1]]).to(torch_device)
    
    # Create original model
    print("\n--- Creating original model ---")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.to(torch_device)
    
    # Set deterministic behavior
    torch.manual_seed(42)
    if torch_device == "cuda":
        torch.cuda.manual_seed(42)
    elif torch_device == "xpu":
        torch.xpu.manual_seed(42)
    torch.use_deterministic_algorithms(True) 
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    # Test 1: Without torch.no_grad()
    print("\n--- Test 1: WITHOUT torch.no_grad() ---")
    print("Model requires_grad status:")
    grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grad_params.append(name)
    print(f"  Parameters requiring grad: {len(grad_params)} (showing first 5: {grad_params[:5]})")
    
    output_with_grad = model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"Output shape: {output_with_grad.logits.shape}")
    print(f"Output dtype: {output_with_grad.logits.dtype}")
    print(f"Output device: {output_with_grad.logits.device}")
    print(f"Output requires_grad: {output_with_grad.logits.requires_grad}")
    print(f"Output grad_fn: {output_with_grad.logits.grad_fn}")
    
    # Test 2: With torch.no_grad()
    print("\n--- Test 2: WITH torch.no_grad() ---")
    with torch.no_grad():
        output_no_grad = model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"Output shape: {output_no_grad.logits.shape}")
    print(f"Output dtype: {output_no_grad.logits.dtype}")
    print(f"Output device: {output_no_grad.logits.device}")
    print(f"Output requires_grad: {output_no_grad.logits.requires_grad}")
    print(f"Output grad_fn: {output_no_grad.logits.grad_fn}")
    
    # Test 3: Another run without torch.no_grad() to check consistency
    print("\n--- Test 3: SECOND run WITHOUT torch.no_grad() ---")
    output_with_grad_2 = model(input_ids=input_ids, attention_mask=attention_mask)
    
    print(f"Output requires_grad: {output_with_grad_2.logits.requires_grad}")
    print(f"Output grad_fn: {output_with_grad_2.logits.grad_fn}")
    
    # Compare outputs
    print("\n=== Comparison Results ===")
    
    # Compare with_grad vs no_grad
    diff_grad_vs_nograd = torch.abs(output_with_grad.logits - output_no_grad.logits)
    max_diff_grad_vs_nograd = torch.max(diff_grad_vs_nograd).item()
    mean_diff_grad_vs_nograd = torch.mean(diff_grad_vs_nograd).item()
    
    print(f"--- With_grad vs No_grad ---")
    print(f"Max difference: {max_diff_grad_vs_nograd:.15f}")
    print(f"Mean difference: {mean_diff_grad_vs_nograd:.15f}")
    print(f"Are exactly equal: {torch.equal(output_with_grad.logits, output_no_grad.logits)}")
    print(f"Are close (rtol=0, atol=0): {torch.allclose(output_with_grad.logits, output_no_grad.logits, rtol=0, atol=0)}")
    print(f"Are close (rtol=1e-5, atol=1e-8): {torch.allclose(output_with_grad.logits, output_no_grad.logits, rtol=1e-5, atol=1e-8)}")
    
    # Compare first with_grad vs second with_grad (consistency check)
    diff_grad_consistency = torch.abs(output_with_grad.logits - output_with_grad_2.logits)
    max_diff_grad_consistency = torch.max(diff_grad_consistency).item()
    mean_diff_grad_consistency = torch.mean(diff_grad_consistency).item()
    
    print(f"--- First with_grad vs Second with_grad (consistency) ---")
    print(f"Max difference: {max_diff_grad_consistency:.15f}")
    print(f"Mean difference: {mean_diff_grad_consistency:.15f}")
    print(f"Are exactly equal: {torch.equal(output_with_grad.logits, output_with_grad_2.logits)}")
    print(f"Are close (rtol=0, atol=0): {torch.allclose(output_with_grad.logits, output_with_grad_2.logits, rtol=0, atol=0)}")
    
    # Detailed analysis if there are differences
    if max_diff_grad_vs_nograd > 0:
        print(f"\n=== Detailed Analysis: with_grad vs no_grad ===")
        max_idx = torch.argmax(diff_grad_vs_nograd)
        max_idx_unravel = torch.unravel_index(max_idx, diff_grad_vs_nograd.shape)
        print(f"Max difference location: {max_idx_unravel}")
        print(f"With_grad value at max diff: {output_with_grad.logits[max_idx_unravel].item():.15f}")
        print(f"No_grad value at max diff: {output_no_grad.logits[max_idx_unravel].item():.15f}")
        
        num_diffs = torch.sum(diff_grad_vs_nograd > 0).item()
        total_elements = diff_grad_vs_nograd.numel()
        print(f"Number of different elements: {num_diffs} / {total_elements} ({100.0 * num_diffs / total_elements:.2f}%)")
        
        # Check distribution of differences
        if num_diffs > 0:
            non_zero_diffs = diff_grad_vs_nograd[diff_grad_vs_nograd > 0]
            print(f"Min non-zero difference: {torch.min(non_zero_diffs).item():.15f}")
            print(f"Max non-zero difference: {torch.max(non_zero_diffs).item():.15f}")
            print(f"Mean non-zero difference: {torch.mean(non_zero_diffs).item():.15f}")
    
    if max_diff_grad_consistency > 0:
        print(f"\n=== Detailed Analysis: grad consistency ===")
        max_idx = torch.argmax(diff_grad_consistency)
        max_idx_unravel = torch.unravel_index(max_idx, diff_grad_consistency.shape)
        print(f"Max difference location: {max_idx_unravel}")
        print(f"First with_grad value: {output_with_grad.logits[max_idx_unravel].item():.15f}")
        print(f"Second with_grad value: {output_with_grad_2.logits[max_idx_unravel].item():.15f}")
        
        num_diffs = torch.sum(diff_grad_consistency > 0).item()
        total_elements = diff_grad_consistency.numel()
        print(f"Number of different elements: {num_diffs} / {total_elements} ({100.0 * num_diffs / total_elements:.2f}%)")
    
    # Test memory usage (if possible)
    if torch_device == "cuda":
        print(f"\n=== Memory Usage ===")
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()
        
        # Test with grad
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
        mem_with_grad = torch.cuda.memory_allocated()
        
        torch.cuda.empty_cache()
        
        # Test without grad
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        mem_no_grad = torch.cuda.memory_allocated()
        
        print(f"Memory with grad: {mem_with_grad - mem_before} bytes")
        print(f"Memory no grad: {mem_no_grad - mem_before} bytes")
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Device: {torch_device}")
    print(f"Original model with_grad vs no_grad:")
    print(f"  Max difference: {max_diff_grad_vs_nograd:.15f}")
    print(f"  Exactly equal: {torch.equal(output_with_grad.logits, output_no_grad.logits)}")
    print(f"  Close (rtol=0, atol=0): {torch.allclose(output_with_grad.logits, output_no_grad.logits, rtol=0, atol=0)}")
    
    if torch_device == "xpu" and max_diff_grad_vs_nograd > 0:
        print(f"\n=== XPU Specific Notes ===")
        print("On XPU, even the original model shows differences between")
        print("with_grad and no_grad modes. This suggests that XPU's")
        print("implementation of gradient computation affects numerical")
        print("precision even for inference-only operations.")
    
    return {
        'max_diff_grad_vs_nograd': max_diff_grad_vs_nograd,
        'max_diff_grad_consistency': max_diff_grad_consistency,
        'exactly_equal': torch.equal(output_with_grad.logits, output_no_grad.logits),
        'close_strict': torch.allclose(output_with_grad.logits, output_no_grad.logits, rtol=0, atol=0),
        'close_tolerant': torch.allclose(output_with_grad.logits, output_no_grad.logits, rtol=1e-5, atol=1e-8)
    }


if __name__ == "__main__":
    # Run the original model test
    test_original_model_grad_difference()
    print("\n" + "="*50 + "\n")

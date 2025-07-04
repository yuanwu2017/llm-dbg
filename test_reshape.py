import torch
import numpy as np
import unittest
import os


def compare_tensors(t1, t2, rtol=1e-5, atol=1e-8, indent=""):
    """
    Compare two tensors in detail, based on compare.py implementation
    
    Args:
        t1, t2: tensors to compare
        rtol: relative tolerance
        atol: absolute tolerance
        indent: output indentation
    
    Returns:
        bool: whether tensors are equal within tolerance
    """
    print(f"{indent}├── Shape: {tuple(t1.shape)} vs {tuple(t2.shape)}")
    print(f"{indent}├── Data type: {t1.dtype} vs {t2.dtype}")
    print(f"{indent}├── Device: {t1.device} vs {t2.device}")
    
    # Move tensors to CPU if they are on different devices
    if t1.device != t2.device:
        print(f"{indent}├── ⚠️ Different devices, moving tensors to CPU for comparison")
        t1 = t1.cpu()
        t2 = t2.cpu()
    
    if t1.shape != t2.shape:
        print(f"{indent}└── ❌ Different shapes")
        return False
    
    if t1.dtype != t2.dtype:
        print(f"{indent}└── ❌ Different data types")
        return False
    
    # Calculate difference statistics
    diff = torch.abs(t1 - t2)
    max_diff = torch.max(diff).item()
    
    # For integer tensors, convert to float to calculate mean
    if diff.dtype.is_floating_point or diff.dtype.is_complex:
        mean_diff = torch.mean(diff).item()
    else:
        mean_diff = torch.mean(diff.float()).item()
    
    print(f"{indent}├── Max difference: {max_diff:.6f}")
    print(f"{indent}├── Mean difference: {mean_diff:.6f}")
    
    # Find position with maximum difference
    if max_diff > atol:
        idx = torch.argmax(diff)
        idx_tuple = np.unravel_index(idx.cpu().numpy(), t1.shape)
        print(f"{indent}├── Max difference position: {idx_tuple}")
        print(f"{indent}│   ├── Value1: {t1[idx_tuple].item():.6f}")
        print(f"{indent}│   └── Value2: {t2[idx_tuple].item():.6f}")
    
    # Check if all elements are close
    is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    if is_close:
        print(f"{indent}└── ✅ All values are within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"{indent}└── ❌ Differences exceed tolerance")
    
    return is_close


def test_reshape_as():
    """
    Test reshape_as operation from shape(48, 64, 64, 64, 64) to shape(48, 4096, 4096)
    Compare results on CPU and XPU devices
    """
    print("Starting reshape_as test...")
    
    # Define input and target shapes
    input_shape = (48, 64, 64, 64, 64)
    target_shape = (48, 4096, 4096)
    
    # Verify shape compatibility
    input_size = np.prod(input_shape)
    target_size = np.prod(target_shape)
    assert input_size == target_size, f"Shape incompatible: {input_size} != {target_size}"
    
    # Create test data
    print(f"Creating random tensor with shape {input_shape}...")
    torch.manual_seed(42)  # Set random seed for reproducibility
    
    # CPU test
    print("Executing reshape operation on CPU...")
    cpu_tensor = torch.randn(input_shape, dtype=torch.float32, device='cpu')
    target_tensor = torch.randn(target_shape, dtype=torch.float32, device='cpu')
    
    # Use reshape_as for reshaping
    cpu_reshaped = cpu_tensor.reshape_as(target_tensor)
    
    print(f"CPU original tensor shape: {cpu_tensor.shape}")
    print(f"CPU target tensor shape: {target_tensor.shape}")
    print(f"CPU reshaped tensor shape: {cpu_reshaped.shape}")
    
    # Verify CPU results
    assert cpu_reshaped.shape == target_shape, f"CPU reshape shape error: {cpu_reshaped.shape} != {target_shape}"
    
    # Verify data integrity: data should be consistent before and after reshape
    print("Verifying CPU reshape data integrity...")
    data_integrity_result = torch.equal(cpu_reshaped.contiguous().view(-1), cpu_tensor.contiguous().view(-1))
    if data_integrity_result:
        print("✓ CPU reshape data integrity verification passed")
    else:
        print("❌ CPU reshape data inconsistent")
        return False
    
    # XPU test (if XPU is available)
    xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
    
    if xpu_available:
        print("Executing reshape operation on XPU...")
        try:
            # Move data to XPU
            xpu_tensor = cpu_tensor.to('xpu')
            xpu_target = target_tensor.to('xpu')
            
            # Execute reshape_as on XPU
            xpu_reshaped = xpu_tensor.reshape_as(xpu_target)
            
            print(f"XPU original tensor shape: {xpu_tensor.shape}")
            print(f"XPU target tensor shape: {xpu_target.shape}")
            print(f"XPU reshaped tensor shape: {xpu_reshaped.shape}")
            
            # Verify XPU results
            assert xpu_reshaped.shape == target_shape, f"XPU reshape shape error: {xpu_reshaped.shape} != {target_shape}"
            
            # Move XPU result back to CPU for comparison
            xpu_reshaped_cpu = xpu_reshaped.cpu()
            
            # Compare CPU and XPU results
            print("Comparing CPU and XPU results...")
            comparison_result = compare_tensors(cpu_reshaped, xpu_reshaped_cpu, rtol=1e-5, atol=1e-8, indent="  ")
            if not comparison_result:
                print("❌ CPU and XPU results are inconsistent")
                return False
            
            print("✓ CPU and XPU results are consistent!")
            
        except Exception as e:
            print(f"XPU test error: {e}")
            return False
    else:
        print("XPU not available, skipping XPU test")
    
    print("✓ reshape_as test passed!")
    return True


def test_reshape_as_performance():
    """
    Performance comparison test
    """
    print("\nStarting performance test...")
    
    input_shape = (48, 64, 64, 64, 64)
    target_shape = (48, 4096, 4096)
    
    # CPU performance test
    cpu_tensor = torch.randn(input_shape, dtype=torch.float32, device='cpu')
    target_tensor = torch.randn(target_shape, dtype=torch.float32, device='cpu')
    
    import time
    
    # CPU timing
    start_time = time.time()
    for _ in range(100):
        cpu_reshaped = cpu_tensor.reshape_as(target_tensor)
    cpu_time = time.time() - start_time
    print(f"CPU average time: {cpu_time/100*1000:.3f} ms")
    
    # XPU performance test (if available)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        try:
            xpu_tensor = cpu_tensor.to('xpu')
            xpu_target = target_tensor.to('xpu')
            
            # Warmup
            for _ in range(10):
                _ = xpu_tensor.reshape_as(xpu_target)
            
            torch.xpu.synchronize()
            start_time = time.time()
            for _ in range(100):
                xpu_reshaped = xpu_tensor.reshape_as(xpu_target)
            torch.xpu.synchronize()
            xpu_time = time.time() - start_time
            
            print(f"XPU average time: {xpu_time/100*1000:.3f} ms")
            print(f"Speedup: {cpu_time/xpu_time:.2f}x")
            
        except Exception as e:
            print(f"XPU performance test error: {e}")


def test_reshape_as_from_files():
    """
    Test reshape_as operation with tensors loaded from files
    Input tensor from attn.decomposed_rel_pos.2.pt
    Target tensor from attn.attn_weights.2.pt
    Compare results on CPU and XPU devices
    """
    print("\nStarting reshape_as test with tensors loaded from files...")
    
    # Define file paths
    input_file = "attn.decomposed_rel_pos.2.pt"
    target_file = "attn.attn_weights.2.pt"
    
    # Check if files exist
    if not os.path.exists(input_file):
        print(f"Warning: Input file {input_file} does not exist, skipping this test")
        return True
    
    if not os.path.exists(target_file):
        print(f"Warning: Target file {target_file} does not exist, skipping this test")
        return True
    
    try:
        # Load tensors from files
        print(f"Loading input tensor from file {input_file}...")
        input_tensor = torch.load(input_file, map_location='cpu')
        
        print(f"Loading target tensor from file {target_file}...")
        target_tensor = torch.load(target_file, map_location='cpu')
        
        # Ensure tensors are float32 type
        input_tensor = input_tensor.to(dtype=torch.float32)
        target_tensor = target_tensor.to(dtype=torch.float32)
        
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Target tensor shape: {target_tensor.shape}")
        
        # Verify shape compatibility
        input_size = input_tensor.numel()
        target_size = target_tensor.numel()
        
        if input_size != target_size:
            print(f"Warning: Tensor sizes incompatible ({input_size} != {target_size}), cannot perform reshape_as operation")
            return True
        
        # CPU test
        print("Executing reshape_as operation on CPU...")
        cpu_input = input_tensor.clone()
        cpu_target = target_tensor.clone()
        
        # Use reshape_as for reshaping
        cpu_reshaped = cpu_input.reshape_as(cpu_target)
        
        print(f"CPU reshaped tensor shape: {cpu_reshaped.shape}")
        
        # Verify CPU results
        assert cpu_reshaped.shape == target_tensor.shape, \
            f"CPU reshape shape error: {cpu_reshaped.shape} != {target_tensor.shape}"
        
        # Verify data integrity: data should be consistent before and after reshape
        print("Verifying CPU reshape data integrity...")
        data_integrity_result = torch.equal(cpu_reshaped.contiguous().view(-1), cpu_input.contiguous().view(-1))
        if data_integrity_result:
            print("✓ CPU reshape data integrity verification passed")
        else:
            print("❌ CPU reshape data inconsistent")
            return False
        
        # XPU test (if XPU is available)
        xpu_available = hasattr(torch, 'xpu') and torch.xpu.is_available()
        
        if xpu_available:
            print("Executing reshape_as operation on XPU...")
            try:
                # Move data to XPU
                xpu_input = cpu_input.to('xpu')
                xpu_target = cpu_target.to('xpu')
                
                # Execute reshape_as on XPU
                xpu_reshaped = xpu_input.reshape_as(xpu_target)
                
                print(f"XPU reshaped tensor shape: {xpu_reshaped.shape}")
                
                # Verify XPU results
                assert xpu_reshaped.shape == target_tensor.shape, \
                    f"XPU reshape shape error: {xpu_reshaped.shape} != {target_tensor.shape}"
                
                # Move XPU result back to CPU for comparison
                xpu_reshaped_cpu = xpu_reshaped.cpu()
                
                # Compare CPU and XPU results
                print("Comparing CPU and XPU results...")
                comparison_result = compare_tensors(cpu_reshaped, xpu_reshaped_cpu, rtol=1e-5, atol=1e-8, indent="  ")
                if not comparison_result:
                    print("❌ CPU and XPU results are inconsistent")
                    return False
                
                print("✓ CPU and XPU results are consistent!")
                
            except Exception as e:
                print(f"XPU test error: {e}")
                return False
        else:
            print("XPU not available, skipping XPU test")
        
        print("✓ reshape_as test with file-loaded tensors passed!")
        return True
        
    except Exception as e:
        print(f"File-loaded tensor test error: {e}")
        return False


class TestReshapeAs(unittest.TestCase):
    """
    Unit test class
    """
    
    def test_reshape_as_correctness(self):
        """Test reshape_as correctness"""
        result = test_reshape_as()
        self.assertTrue(result)
    
    def test_reshape_as_from_files(self):
        """Test reshape_as operation with tensors loaded from files"""
        result = test_reshape_as_from_files()
        self.assertTrue(result)

if __name__ == "__main__":
    print("=== Reshape_as Test Started ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"XPU available: {hasattr(torch, 'xpu') and torch.xpu.is_available() if hasattr(torch, 'xpu') else False}")
   
    # Run unit tests
    print("\n=== Running Unit Tests ===")
    unittest.main(argv=[''], exit=False, verbosity=2)
    

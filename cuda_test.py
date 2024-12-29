import subprocess
import sys
import torch
import platform
import os

def run_command(command):
    """Execute a command and return its output"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error executing {command}: {e.stderr}"

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    if platform.system() == "Windows":
        command = "nvidia-smi"
    else:
        command = "nvidia-smi"
    
    print("\n=== NVIDIA Driver Check ===")
    output = run_command(command)
    print(output)

def check_cuda_version():
    """Check CUDA version"""
    print("\n=== CUDA Version Check ===")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("CUDA is not available")

def check_pytorch_setup():
    """Check PyTorch installation and CUDA capability"""
    print("\n=== PyTorch Setup ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        print(f"Device count: {torch.cuda.device_count()}")

def check_gpu_memory():
    """Check GPU memory usage"""
    if torch.cuda.is_available():
        print("\n=== GPU Memory Usage ===")
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i} Memory Summary:")
            print(torch.cuda.memory_summary(device=i))

def test_cuda_operation():
    """Test a basic CUDA operation"""
    print("\n=== Testing CUDA Operation ===")
    if torch.cuda.is_available():
        try:
            # Create a tensor on CPU
            x = torch.rand(5, 3)
            print("CPU Tensor created successfully")
            
            # Move tensor to GPU
            device = torch.device("cuda")
            x = x.to(device)
            print("Tensor successfully moved to GPU")
            
            # Perform a simple operation
            y = x * 2
            print("CUDA operation completed successfully")
            
        except Exception as e:
            print(f"Error during CUDA operation: {e}")
    else:
        print("CUDA is not available for testing operations")

def main():
    print("=== CUDA Diagnostic Tool ===")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    
    check_nvidia_driver()
    check_cuda_version()
    check_pytorch_setup()
    check_gpu_memory()
    test_cuda_operation()
    
    print("\n=== Environment Variables ===")
    cuda_related_vars = {k: v for k, v in os.environ.items() if 'CUDA' in k}
    for k, v in cuda_related_vars.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main() 
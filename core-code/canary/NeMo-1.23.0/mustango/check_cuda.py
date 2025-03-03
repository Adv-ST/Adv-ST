import torch

def check_cuda_availability():
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print("CUDA is available!")
        # Print additional information
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU only.")
    
    return cuda_available

if __name__ == "__main__":
    check_cuda_availability()
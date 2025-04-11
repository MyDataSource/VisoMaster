import torch
import platform

print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {platform.python_version()}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    # 测试基本操作
    x = torch.ones(5, device=device)
    y = torch.ones(5, device=device)
    z = x + y
    print(f"Basic operation test: {z}")
    
    # 测试矩阵运算
    a = torch.randn(2, 3, device=device)
    b = torch.randn(3, 2, device=device)
    c = torch.matmul(a, b)
    print(f"Matrix multiplication test: {c}")
    
    print("MPS is working correctly with basic operations!")
else:
    print("MPS is not available") 
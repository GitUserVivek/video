import os
import shutil
import psutil

# CPU
print("=== CPU ===")
print("Physical cores:", psutil.cpu_count(logical=False))
print("Logical cores:", psutil.cpu_count(logical=True))
print("CPU usage:", psutil.cpu_percent(interval=1), "%")

# RAM
ram = psutil.virtual_memory()
print("\n=== RAM ===")
print(f"Total: {ram.total / (1024**3):.2f} GB")
print(f"Used:  {ram.used / (1024**3):.2f} GB")
print(f"Free:  {ram.available / (1024**3):.2f} GB")
print(f"Usage: {ram.percent}%")

# Storage
print("\n=== STORAGE ===")
for disk in psutil.disk_partitions():
    try:
        usage = psutil.disk_usage(disk.mountpoint)
        print(f"{disk.mountpoint}:")
        print(f"  Total: {usage.total / (1024**3):.2f} GB")
        print(f"  Used:  {usage.used / (1024**3):.2f} GB")
        print(f"  Free:  {usage.free / (1024**3):.2f} GB")
        print(f"  Usage: {usage.percent}%")
    except PermissionError:
        pass

# GPU
print("\n=== GPU ===")
if shutil.which("nvidia-smi"):
    os.system("nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv")
else:
    print("NVIDIA GPU not detected (or nvidia-smi is unavailable).")

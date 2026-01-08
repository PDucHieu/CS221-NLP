# run.py - Đặt cùng cấp với app.py
import os
import sys

# Đảm bảo import đúng
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import và chạy app
from app import main

if __name__ == "__main__":
    # Kiểm tra cấu trúc thư mục
    print("Checking directory structure...")
    required_dirs = ['models', 'preprocessing', 'data_loader', 'evaluation']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"Creating directory: {dir_name}")
            os.makedirs(dir_name, exist_ok=True)
    
    # Chạy Streamlit
    import subprocess
    subprocess.run(["streamlit", "run", "app.py"])
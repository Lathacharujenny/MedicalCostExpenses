import os

# Define the path to the new file
file_path = os.path.join('src1', 'hello.py')
file_path1 = os.path.join('src1', 'hello1.py')
# Create the directory if it doesn't exist
os.makedirs(file_path1, exist_ok=True)
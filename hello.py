import os
import urllib.request

# Define the directory name
dir_name = '../data/UCI/'

# Check if the parent directory exists, if not, create it
parent_dir = os.path.dirname(dir_name)
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

# Now create the target directory
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Define the URL and file path
url = 'https://github.com/fairlearn/talks/raw/main/2021_scipy_tutorial/data/diabetic_data.csv'
file_path = os.path.join(dir_name, 'diabetic_data.csv')

# Download the file
print(f"Downloading raw data from '{url}'...")
urllib.request.urlretrieve(url, file_path)
print(f"Data downloaded to '{file_path}'")
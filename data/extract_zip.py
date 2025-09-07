import zipfile
import os

def extract_zip(zip_path, extract_to=None):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted all files to: {extract_to}")
    print(f"Extracted all files to: {extract_to}")

def process_directory(directory):
    extract_to = directory
    with os.scandir(directory) as entries:
        files = [entry.name for entry in entries if entry.is_file()]
        for file in files:
            if file.endswith(".zip"):
                extract_to = os.path.join(directory, file[:-4])
                extract_zip(os.path.join(directory, file), extract_to)
                os.remove(os.path.join(directory, file))


if __name__ == "__main__":
    types = ["sitting-down", "standing-up", "walking", "running", "climbing-stairs"]
    for typee in types:
        directorytory = typee
        process_directory(directory)
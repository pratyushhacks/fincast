import os

PATH = "data/raw"

# list all top level folders in PATH
folders = [f for f in os.listdir(PATH) if os.path.isdir(os.path.join(PATH, f))]
print("Folders in data/raw:")
for f in folders:
    print(f" - {f}")
    
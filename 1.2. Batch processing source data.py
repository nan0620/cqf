import os
import shutil

# Source folder and destination folder
source_folder = '/Users/nanjiang/cqf/spx_eod_2023'
destination_folder = '/Users/nanjiang/cqf/spx_eod_unhandled'

# Create the destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate through files in the source folder
for filename in os.listdir(source_folder):
    # Full path of the file
    source_file = os.path.join(source_folder, filename)

    # If it's a file and not a folder
    if os.path.isfile(source_file):
        # Full path of the destination file
        destination_file = os.path.join(destination_folder, filename)

        # Copy the file
        shutil.copy(source_file, destination_file)
        print(f'File {filename} has been copied to {destination_folder}')

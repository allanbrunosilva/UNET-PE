import os
import nrrd
from PIL import Image

def extract(input_nrrd_path, output_folder_path):
    """
    Reads a NRRD file and saves each slice as a PNG image in the specified output folder.

    Parameters:
    - input_nrrd_path: Path to the input NRRD file.
    - output_folder_path: Path to the output folder where PNG images will be saved.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Load the NRRD file
    data, _ = nrrd.read(input_nrrd_path)

    # Save each slice as a PNG image
    for i in range(data.shape[2]):  # Assuming the last dimension is the slices
        # Creating a PIL image from a data slice
        image = Image.fromarray(data[:, :, i])
        image = image.convert('L')  # Convert to grayscale
        
        # Constructing filename for the slice
        filename = os.path.join(output_folder_path, f'{i:04d}.png')
        
        # Saving the image
        image.save(filename)

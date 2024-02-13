import os
import nrrd
from PIL import Image
import SimpleITK as sitk
import numpy as np
import utils.preprocessors.nrrd_preprocessor as nrrd_preprocessor

def extract_and_preprocess(input_nrrd_path, output_folder_path):
    """
    Reads a NRRD file, preprocess, and saves each sagittal slice as a PNG image.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Load the NRRD file
    data, header = nrrd.read(input_nrrd_path)

    # Resample the volume to 1 mm slice thickness
    data = nrrd_preprocessor.resample_volume_to_one_mm(data, header)
    data = sitk.GetArrayFromImage(data)

    WL = 350  # Window Length
    WC = 150  # Window Center

    # Save each slice as a PNG image
    for i in range(data.shape[2]):  # Assuming the last dimension is the slices
        slice_data = data[:, :, i]
        adjusted_slice = nrrd_preprocessor.window_level_adjustment(slice_data, WC, WL)
        
        # Normaliza e recorta a imagem
        normalized_cropped_slice = nrrd_preprocessor.normalize_and_crop_image(adjusted_slice)
        
        # Convertendo para uma imagem PIL para salvar
        image = Image.fromarray((normalized_cropped_slice * 255).astype(np.uint8))
        image = image.convert('L')  # Convert to grayscale
        
        # Constructing filename for the slice
        filename = os.path.join(output_folder_path, f'{i:04d}.png')
        
        # Saving the image
        image.save(filename)

import SimpleITK as sitk
import numpy as np

def resample_volume_to_one_mm(nrrd_data, nrrd_header):
    """
    Resamples the 3D volume data to have 1 mm slice thickness.

    Parameters:
    - nrrd_data: The volume data loaded from a NRRD file.
    - nrrd_header: The header of the NRRD file, containing metadata.

    Returns:
    - Resampled volume data as a SimpleITK Image.
    """
    # Convert the NRRD data to a SimpleITK Image
    original_image = sitk.GetImageFromArray(nrrd_data)
    original_spacing = original_image.GetSpacing()
    new_spacing = (original_spacing[0], original_spacing[1], 1.0)  # Change the z-axis spacing to 1 mm

    original_size = original_image.GetSize()
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]

    # Resample the image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(original_image)
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(original_image)

def window_level_adjustment(image_array, window_center, window_length):
    """
    Adjusts the image array based on the window center and window length for CT images.
    
    Parameters:
    - image_array: NumPy array of the CT image.
    - window_center: Center of the window for the HU scale (WC).
    - window_length: Length of the window for the HU scale (WL).
    
    Returns:
    - Adjusted NumPy array with HU values scaled to the range [0, 255].
    """
    min_hu = window_center - (window_length / 2)
    max_hu = window_center + (window_length / 2)
    image_array = np.clip(image_array, min_hu, max_hu)
    np.subtract(image_array, min_hu, out=image_array)
    np.multiply(image_array, 255 / (max_hu - min_hu), out=image_array)
    image_array = image_array.astype(np.uint8)
    return image_array

def normalize_and_crop_image(image_array):
    """
    Normaliza os valores de uma matriz de imagem para o intervalo [0, 1] e recorta uma janela central de tamanho 364x364.
    
    Parâmetros:
    - image_array: Matriz NumPy da imagem.
    
    Retorna:
    - Matriz NumPy normalizada e recortada da imagem.
    """
    # Normalização para o intervalo [0, 1]
    image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    
    # Recorte para obter uma janela central de 364x364
    height, width = image_array.shape
    center_height, center_width = height // 2, width // 2
    cropped_image = image_array[center_height-182:center_height+182, center_width-182:center_width+182]
    
    return cropped_image


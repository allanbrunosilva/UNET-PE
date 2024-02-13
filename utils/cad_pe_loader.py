import os
import argparse
import extractors.nrrd_extractor as nrrd_extractor
import spliters.dataset_splitter as dataset_splitter

"""
CAD-PE/patient/image
CAD-PE/patient/mask
"""

def extract_cad_pe(main_path, cad_pe_path, n = None):
    i = 1
    for patient_image, patient_mask in zip(os.listdir(cad_pe_path+"/images"), os.listdir(cad_pe_path+"/rs")):
        image_input = cad_pe_path + '/images/' + patient_image
        mask_input = cad_pe_path + '/rs/' + patient_mask

        image_output = f"{main_path}/data/01_preprocessed/CAD-PE/{i}/image/"
        mask_output = f"{main_path}/data/01_preprocessed/CAD-PE/{i}/mask/"

        print(f"Input: {image_input}, {mask_input}")
        print(f"Output: {image_output}, {mask_output}")

        nrrd_extractor.extract_and_preprocess(image_input, image_output)
        nrrd_extractor.extract_and_preprocess(mask_input, mask_output)

        i+=1
        if n != None and i>n:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepares the CAD-PE dataset.")
    parser.add_argument('--main_path', type=str, required=True, help="Path to repository.")
    parser.add_argument('--patients_n', type=int, required=False, help="Patients limit.")

    args = parser.parse_args()

    # Using the passed arguments
    main_path = args.main_path
    patients_n = args.patients_n

    cad_pe_original_path = f"{main_path}/data/00_original/CAD-PE"
    extract_cad_pe(main_path, cad_pe_original_path, patients_n)

    # Set the path to your dataset directory and output directory
    cad_pe_preprocessed_path = f"{main_path}/data/01_preprocessed/CAD-PE"
    cad_pe_splited_path = f"{main_path}/data/02_splited/CAD-PE"
    dataset_splitter.split(cad_pe_preprocessed_path, cad_pe_splited_path)

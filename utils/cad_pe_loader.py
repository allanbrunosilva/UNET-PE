import os
import argparse
import nrrd_extractor
import image_preprocessor
import dataset_splitter

"""
CAD-PE/patient/image
CAD-PE/patient/mask
"""

def extract_cad_pe(main_path, cad_pe_path):
    i = 1
    for patient_image, patient_mask in zip(os.listdir(cad_pe_path+"/images"), os.listdir(cad_pe_path+"/rs")):
        image_input = cad_pe_path + '/images/' + patient_image
        mask_input = cad_pe_path + '/rs/' + patient_mask

        image_output = f"{main_path}/data/01_extracted/CAD-PE/{i}/image/"
        mask_output = f"{main_path}/data/01_extracted/CAD-PE/{i}/mask/"

        print(f"Input: {image_input}, {mask_input}")
        print(f"Output: {image_output}, {mask_output}")

        nrrd_extractor.extract(image_input, image_output)
        nrrd_extractor.extract(mask_input, mask_output)

        i+=1
        if i>5:
            break

def preprocess_cad_pe(main_path, cad_pe_path):
    i = 1
    for patient in os.listdir(cad_pe_path):
        print(patient)

        for image, mask in zip(os.listdir(cad_pe_path+"/"+patient+"/image"), os.listdir(cad_pe_path+"/"+patient+"/mask")):
            image_input = cad_pe_path + "/" + patient + '/image/' + image
            mask_input = cad_pe_path + "/" + patient + '/mask/' + mask

            image_output = f"{main_path}/data/02_preprocessed/CAD-PE/{i}/image/"
            mask_output = f"{main_path}/data/02_preprocessed/CAD-PE/{i}/mask/"

            print(f"Input: {image_input}, {mask_input}")
            print(f"Output: {image_output}, {mask_output}")

            image_preprocessor.preprocess(image_input, image_output)
            image_preprocessor.preprocess(mask_input, mask_output)

        i+=1
        if i>5:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepares the CAD-PE dataset.")
    parser.add_argument('--main_path', type=str, required=True, help="Path to repository.")

    args = parser.parse_args()

    # Using the passed arguments
    main_path = args.main_path

    cad_pe_original_path = f"{main_path}/data/00_original/CAD-PE"
    extract_cad_pe(main_path, cad_pe_original_path)

    cad_pe_extracted_path = f"{main_path}/data/01_extracted/CAD-PE"
    preprocess_cad_pe(main_path, cad_pe_extracted_path)

    # Set the path to your dataset directory and output directory
    cad_pe_preprocessed_path = f"{main_path}/data/02_preprocessed/CAD-PE"
    cad_pe_splited_path = f"{main_path}/data/03_splited/CAD-PE"
    dataset_splitter.split(cad_pe_preprocessed_path, cad_pe_splited_path)

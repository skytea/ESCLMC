import os
import pickle
import sys
import shutil
import uuid
import zipfile
import argparse
import numpy as np
import SimpleITK as sitk
import cv2
import torch
from sklearn import preprocessing
from radiomics import featureextractor
import pickle

from model import *

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_WEIGHTS_PATH = os.path.join(APP_ROOT, "min_loss0.pth")
RADIOMICS_CONFIG_DIR = os.path.join(APP_ROOT, "radiomics_config")
NNUNET_BASE_DIR = os.path.join(APP_ROOT, 'nnUNet')

# --- Radiomics setting ---
settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = None
settings['interpolator'] = sitk.sitkBSpline
settings['geometryTolerance'] = 10000


# ==========================================
# ==========================================

def cropping(dcm_path, destination_path):
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    dcms = os.listdir(dcm_path)
    dcms = [f for f in dcms if not f.startswith('.')]

    count = 0
    print(f"Running Cropping on {len(dcms)} files...")
    for dcm_name in dcms:
        path = os.path.join(dcm_path, dcm_name)
        try:
            temp_dcm = sitk.ReadImage(path)
            img_array = sitk.GetArrayFromImage(temp_dcm)

            if len(img_array.shape) == 3:
                img_array = img_array[0]

            img_array = img_array.reshape((512, 512))

            high = np.max(img_array)
            low = np.min(img_array)
            lungwin = np.array([low * 1., high * 1.])

            denom = lungwin[1] - lungwin[0]
            if denom == 0: denom = 1

            image = (img_array - lungwin[0]) / denom
            image = (image * 255).astype('uint8')

            size_x = 224
            size_y = 224
            x1, y1 = image.shape[0] // 2, image.shape[1] // 2
            padding_x = int(size_x / 2)
            padding_y = int(size_y / 2)
            cropped_image = image[x1 - padding_x:x1 - padding_x + size_x, y1 - padding_y:y1 - padding_y + size_y]

            image_path = os.path.join(destination_path, f'ES_{count}_0000.png')
            cv2.imwrite(image_path, cropped_image)
            count += 1
        except Exception as e:
            print(f"Skipping file {dcm_name}: {e}")
            continue

    return count


def segmentation(image_path, label_path):
    current_env = os.environ.copy()

    if "nnUNet_raw" not in current_env:
        current_env["nnUNet_raw"] = os.path.join(NNUNET_BASE_DIR, "nnUNet_raw")
    if "nnUNet_preprocessed" not in current_env:
        current_env["nnUNet_preprocessed"] = os.path.join(NNUNET_BASE_DIR, "nnUNet_preprocessed")
    if "nnUNet_results" not in current_env:
        current_env["nnUNet_results"] = os.path.join(NNUNET_BASE_DIR, "nnUNet_results")

    if not os.path.exists(label_path):
        os.makedirs(label_path)

    print("Running nnUNet Segmentation...")
    if shutil.which("nnUNetv2_predict") is None:
        raise RuntimeError("Error: 'nnUNetv2_predict' command not found. Please install nnUNetv2.")

    cmd = f'nnUNetv2_predict -d Dataset001_esophagus -i "{image_path}" -o "{label_path}" -f 0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans'

    import subprocess
    result = subprocess.run(cmd, shell=True, env=current_env)

    if result.returncode != 0:
        raise Exception("nnUNet prediction failed")


def preprocess_scalers():
    print("Initializing Preprocessing Scalers...")

    def load_and_scale(filename):
        filepath = os.path.join(RADIOMICS_CONFIG_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        scaler = pickle.load(open(filepath, "rb"))
        return scaler

    s1 = load_and_scale('s1.pkl')
    s2 = load_and_scale('s2.pkl')
    s3 = load_and_scale('s3.pkl')

    def get_features_name(filename):
        filepath = os.path.join(RADIOMICS_CONFIG_DIR, filename)
        with open(filepath, 'r') as f:
            header = f.readline().strip().split('\t')
        return header

    f1 = get_features_name('pcr_feature.txt')
    f2 = get_features_name('os_feature.txt')
    f3 = get_features_name('pfs_feature.txt')

    return s1, s2, s3, f1, f2, f3


def get_radiomics(image_path, label_path, count, s1, s2, s3, pcr_features, os_features, pfs_features):
    print("Extracting Radiomics Features...")
    all_images_path = []
    all_labels_path = []
    for paths in range(count):
        current_path = os.path.join(image_path, f'ES_{paths}_0000.png')
        all_images_path.append(current_path)
        current_path = os.path.join(label_path, f'ES_{paths}.png')
        all_labels_path.append(current_path)

    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.enableAllImageTypes()
    extractor.enableImageTypeByName('LoG', customArgs={'sigma': [1.0, 2.0, 3.0, 4.0, 5.0]})
    extractor.enableAllFeatures()

    image = sitk.ReadImage(all_images_path)
    mask = sitk.ReadImage(all_labels_path)

    featureVector = extractor.execute(image, mask, label=1)

    pcr_feature = [featureVector[kk] for kk in pcr_features]
    os_feature = [featureVector[kk] for kk in os_features]
    pfs_feature = [featureVector[kk] for kk in pfs_features]

    pcr_feature = list(s1.transform([pcr_feature])[0])
    os_feature = list(s2.transform([os_feature])[0])
    pfs_feature = list(s3.transform([pfs_feature])[0])

    radio = pcr_feature + os_feature + pfs_feature
    return radio


def read_images(image_path):
    print("Reading Images for DL Model...")
    images = os.listdir(image_path)

    all_image = None
    count = 0
    for img_path in images:
        each_image_path = os.path.join(image_path, img_path)
        if count >= 16:
            continue
        count += 1
        image_r = np.array(cv2.imread(each_image_path, cv2.IMREAD_GRAYSCALE)).reshape((1, 1, 224, 224))
        all_image = image_r if all_image is None else np.concatenate((all_image, image_r), 1)

    if all_image is None:
        raise ValueError("No images found or read.")

    all_image = all_image.astype('float32')
    all_image = torch.from_numpy(all_image)
    return all_image


def find_dcm_folder(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.startswith('.'):
                return root
    return root_dir


# ==========================================
# ==========================================

def main():
    # 1
    parser = argparse.ArgumentParser(description="ESCLMC Prediction Tool")
    parser.add_argument("input_zip", help="Path to the input ZIP file containing DICOMs")
    args = parser.parse_args()

    input_zip_path = args.input_zip

    if not os.path.exists(input_zip_path):
        print(f"Error: Input file '{input_zip_path}' not found.")
        return

    # 2
    print("--- Loading Models and Configs ---")
    s1, s2, s3, pcr_features, os_features, pfs_features = preprocess_scalers()

    image_resolution = 224
    vision_layers = 12
    vision_width = 16 * 16
    vision_patch_size = 16
    embed_dim = 256

    # 加载主模型
    model = ESCLMC(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        dropout=0.5,
    ).to(DEVICE)

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"Error: Model weights not found at {MODEL_WEIGHTS_PATH}")
        return

    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3
    task_id = str(uuid.uuid4())
    base_work_dir = os.path.join(APP_ROOT, 'temp_run', task_id)
    extract_dir = os.path.join(base_work_dir, 'extracted')
    temp_image_path = os.path.join(base_work_dir, 'temp_image')
    temp_mask_path = os.path.join(base_work_dir, 'temp_mask')

    try:
        # 4
        print(f"--- Processing: {input_zip_path} ---")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(input_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        real_dcm_path = find_dcm_folder(extract_dir)
        print(f"Found DICOMs at: {real_dcm_path}")

        # 5
        # A. Cropping
        image_count = cropping(real_dcm_path, temp_image_path)
        if image_count == 0:
            print("Error: No valid images cropped.")
            return

        # B. Segmentation
        segmentation(temp_image_path, temp_mask_path)

        # C. Feature Extraction & Prediction
        radio = get_radiomics(temp_image_path, temp_mask_path, image_count, s1, s2, s3, pcr_features, os_features,
                              pfs_features)

        image = read_images(temp_image_path)
        image = image.float().to(device=DEVICE)

        radio_tensor = torch.tensor([radio]).float().to(device=DEVICE)

        # D. Inference Loop
        print("Running Inference...")
        torch.cuda.empty_cache()

        count = 0
        image_features = None
        text_features = None
        radio_features = None
        ebed_1 = None
        ebed_0 = None

        while count < 4:
            try:
                image_features, text_features, radio_features, ebed_1, ebed_0 = model(image, radio_tensor)
                break
            except Exception as e:
                print(f"Inference attempt {count} failed: {e}")
                count += 1

        if image_features is None:
            print("Error: Inference failed after retries.")
            return

        x = None
        for ff in [image_features, text_features, radio_features]:
            if ff is None: continue
            if x is None: x = ff; continue
            x += ff
        x = x / x.norm(dim=1, keepdim=True)

        difference_1 = (x @ ebed_1.t() + 1.0) / 2.0
        difference_0 = (x @ ebed_0.t() + 1.0) / 2.0
        pred = difference_1 if difference_1.item() >= difference_0.item() else (difference_0 * -1.0 + 1.0)

        # 6
        print("\n" + "=" * 30)
        print(f"✅ Prediction Result")
        print(f"probability: {pred.item()}")
        print("=" * 30 + "\n")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 7
        if os.path.exists(base_work_dir):
            shutil.rmtree(base_work_dir)
            print("Temp files cleaned up.")


if __name__ == '__main__':
    main()
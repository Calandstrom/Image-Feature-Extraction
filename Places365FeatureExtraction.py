import os
import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
IMAGE_DATA_DIR  = 'data' # Replace with the path to your image data directory
PREVALENCE_CSV  = '' # Replace with teh path to your csv (MORE INFO ABOUT CSV STRUCTURE IN README)
PCN_COL         = 'hospital_id' # Column name for the PCN identifier
PREV_COL        = '' # Column name for your associated data
WEIGHTS_PATH    = 'alexnet_places365.pth.tar' # Replace with the path to your weights
OUTPUT_NPY      = 'features_4096.npy'
OUTPUT_CSV      = 'features_for_R.csv'
IMAGE_ABSOLUTE_BASE = '/' # Replace with the output path
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")
print(f"Image root  : {os.path.abspath(IMAGE_DATA_DIR)}")

# ══════════════════════════════════════════════════════════════════════
# STEP 1 — Load & aggregate prevalence
# ══════════════════════════════════════════════════════════════════════
def load_prevalence(csv_path, pcn_col, prev_col):
    df = pd.read_csv(csv_path)

    # Strip excess white space to make sure the columns match
    df.columns = df.columns.str.strip()

    pcn_df = (df
        .groupby(pcn_col)[prev_col]
        .mean()
        .reset_index()
        .rename(columns={pcn_col: 'pcn_name', prev_col: 'prevalence'})
    )
    print(f"Unique PCNs after aggregation: {len(pcn_df)}")
    return pcn_df

# ══════════════════════════════════════════════════════════════════════
# STEP 2 — Build lookup of folders and metadata files
# ══════════════════════════════════════════════════════════════════════
def build_lookup(image_data_dir):
    folder_lookup   = {}
    metadata_lookup = {}

    for entry in os.listdir(image_data_dir):
        full_path = os.path.join(image_data_dir, entry)

        if os.path.isdir(full_path) and entry.startswith('images_'):
            key = entry.replace('images_', '', 1)
            folder_lookup[key] = full_path

        elif entry.startswith('images_metadata_'):
            key = (entry
                .replace('images_metadata_', '', 1)
                .replace('.parquet', '')
                .replace('.csv', '')
            )
            metadata_lookup[key] = full_path

    return folder_lookup, metadata_lookup

def normalise(s):
    return (str(s).lower()
        .replace('&', 'and')
        .replace('-', '')
        .replace('(', '')
        .replace(')', '')
        .replace(',', '')
        .replace("'", '')
        .replace('  ', ' ')
        .replace(' ', '_')
    )

def match_pcns(pcn_df, folder_lookup, metadata_lookup):
    matched   = []
    unmatched = []

    for _, row in pcn_df.iterrows():
        pcn_name   = row['pcn_name']
        prevalence = row['prevalence']
        key        = pcn_name.replace(' ', '_')

        has_folder   = key in folder_lookup
        has_metadata = key in metadata_lookup

        if has_folder and has_metadata:
            matched.append({
                'pcn_name'      : pcn_name,
                'prevalence': prevalence,
                'folder_path'   : folder_lookup[key],
                'metadata_path' : metadata_lookup[key]
            })
        else:
            norm_key = normalise(pcn_name)
            fuzzy_folder   = next(
                (v for k, v in folder_lookup.items()
                 if normalise(k) == norm_key), None
            )
            fuzzy_metadata = next(
                (v for k, v in metadata_lookup.items()
                 if normalise(k) == norm_key), None
            )

            if fuzzy_folder and fuzzy_metadata:
                matched.append({
                    'pcn_name'      : pcn_name,
                    'prevalence': prevalence,
                    'folder_path'   : fuzzy_folder,
                    'metadata_path' : fuzzy_metadata
                })
            else:
                unmatched.append(pcn_name)

    print(f"\nMatching summary:")
    print(f"  ✓ Matched   : {len(matched)} PCNs")
    print(f"  ✗ Unmatched : {len(unmatched)} PCNs")

    return pd.DataFrame(matched)

# ══════════════════════════════════════════════════════════════════════
# STEP 3 — Load AlexNet Places365, hook fc2 (4096-dim)
# ══════════════════════════════════════════════════════════════════════
def load_model(weights_path):
    model = models.alexnet(num_classes=365)
    checkpoint = torch.load(weights_path, map_location='cpu')
    state_dict = {
        k.replace('module.', ''): v
        for k, v in checkpoint['state_dict'].items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)
    print("AlexNet Places365 loaded successfully")
    return model

def attach_hook(model):
    """
    AlexNet classifier:
      [0] Dropout
      [1] Linear(9216→4096)  fc1
      [2] ReLU
      [3] Dropout
      [4] Linear(4096→4096)  fc2  ← hook here
      [5] ReLU
      [6] Linear(4096→365)   output
    """
    captured = {}

    def hook_fn(module, input, output):
        captured['fc2'] = output.detach().cpu().numpy()

    model.classifier[4].register_forward_hook(hook_fn)
    return model, captured

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ══════════════════════════════════════════════════════════════════════
# STEP 4 — Load valid image paths from metadata
# ══════════════════════════════════════════════════════════════════════
def load_valid_image_paths(metadata_path):



    try:
        if metadata_path.endswith('.parquet'):
            meta = gpd.read_parquet(metadata_path)
        else:
            meta = pd.read_csv(metadata_path)

        valid = meta[meta['status'] == 'ok']

        if len(valid) == 0:
            return []

        resolved = []
        for p in valid['file_path'].tolist():
            abs_path = os.path.join(IMAGE_ABSOLUTE_BASE, p)
            if os.path.exists(abs_path):
                resolved.append(abs_path)

        return resolved

    except Exception as e:
        print(f"    Error reading metadata {metadata_path}: {e}")
        return []

# ══════════════════════════════════════════════════════════════════════
# STEP 5 — Extract features per PCN
# ══════════════════════════════════════════════════════════════════════
def extract_features_for_pcn(image_paths, model, captured):
    vectors = []

    for fpath in image_paths:
        if not os.path.exists(fpath):
            continue
        try:
            img    = Image.open(fpath).convert('RGB')
            tensor = preprocess(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                _ = model(tensor)

            vectors.append(captured['fc2'].squeeze())

        except Exception as e:
            print(f"    Skipping {os.path.basename(fpath)}: {e}")
            continue

    return np.mean(vectors, axis=0) if vectors else None

def run_extraction(matched_df, model, captured):
    results        = {}
    prevalence_map = {}
    skipped        = []

    for _, row in tqdm(matched_df.iterrows(),
                       total=len(matched_df),
                       desc="Extracting features"):
        pcn_name      = row['pcn_name']
        metadata_path = row['metadata_path']
        prevalence    = row['prevalence']

        image_paths = load_valid_image_paths(metadata_path)

        if not image_paths:
            print(f"  WARNING: No valid images for {pcn_name}")
            skipped.append(pcn_name)
            continue

        avg_vector = extract_features_for_pcn(image_paths, model, captured)

        if avg_vector is None:
            print(f"  WARNING: Feature extraction failed for {pcn_name}")
            skipped.append(pcn_name)
            continue

        results[pcn_name]        = avg_vector
        prevalence_map[pcn_name] = prevalence

    return results, prevalence_map, skipped

# ══════════════════════════════════════════════════════════════════════
# STEP 6 — Save outputs
# ══════════════════════════════════════════════════════════════════════
def save_outputs(results, prevalence_map):
    if not results:
        print("ERROR: No features extracted — nothing to save.")
        return

    pcn_names      = list(results.keys())
    feature_matrix = np.stack([results[p] for p in pcn_names])
    prevalences    = np.array([prevalence_map[p] for p in pcn_names])

    np.save(OUTPUT_NPY, {
        'pcn_names'     : np.array(pcn_names),
        'features'      : feature_matrix,
        'prevalence': prevalences
    })

    feature_cols = [f'f{i}' for i in range(feature_matrix.shape[1])]
    df_out = pd.DataFrame(feature_matrix, columns=feature_cols)
    df_out.insert(0, 'pcn_name',       pcn_names)
    df_out.insert(1, 'prevalence', prevalences)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\nOutputs saved:")
    print(f"  {OUTPUT_NPY} — feature matrix {feature_matrix.shape}")
    print(f"  {OUTPUT_CSV} — ready for R SPLS")

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    print("\n── Step 1: Load & aggregate prevalence ───────────────────")
    pcn_df = load_prevalence(PREVALENCE_CSV, PCN_COL, PREV_COL)

    print("\n── Step 2: Match PCNs to image folders ───────────────────")
    folder_lookup, metadata_lookup = build_lookup(IMAGE_DATA_DIR)
    matched_df = match_pcns(pcn_df, folder_lookup, metadata_lookup)

    if len(matched_df) == 0:
        print("ERROR: No PCNs matched. Check IMAGE_DATA_DIR and column names.")
        exit(1)

    matched_df.to_csv('pcn_matched.csv', index=False)

    print("\n── Step 3: Load AlexNet Places365 ───────────────────────")
    model, captured = attach_hook(load_model(WEIGHTS_PATH))

    print("\n── Step 4 & 5: Extract features ──────────────────────────")
    results, prevalence_map, skipped = run_extraction(matched_df, model, captured)

    print("\n── Step 6: Save outputs ──────────────────────────────────")
    save_outputs(results, prevalence_map)

    if skipped:
        print(f"\nSkipped ({len(skipped)}): {skipped}")

    print(f"\n✓ Done. {len(results)} PCNs extracted → ready for SPLS in R.")
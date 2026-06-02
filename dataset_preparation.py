# %% [markdown]
# # Dataset Harmonization & Preparation for MobileNetV2
# Run these cells in your Jupyter Notebook to harmonize and split the Kaggle datasets.

# %%
import os
import shutil
import random
from pathlib import Path

# ==========================================
# 1. DIRECTORY PATHS CONFIGURATION
# ==========================================
MAIN_DIR = r"C:\Punya GW\Kuliah\ProjectKel11"

ARCHIVES = {
    "archive1": r"C:\Punya GW\Kuliah\ProjectKel11\Archive (1)",
    "archive2": r"C:\Punya GW\Kuliah\ProjectKel11\Archive (2)",
    "archive3": r"C:\Punya GW\Kuliah\ProjectKel11\Archive (3)",
    "archive4": r"C:\Punya GW\Kuliah\ProjectKel11\Archive (4)",
}

STAGING_SEGAR = r"C:\Punya GW\Kuliah\ProjectKel11\staging\segar"
STAGING_TIDAK_SEGAR = r"C:\Punya GW\Kuliah\ProjectKel11\staging\tidak_segar"

FINAL_TRAIN_SEGAR = r"C:\Punya GW\Kuliah\ProjectKel11\dataset_final\training\segar"
FINAL_TRAIN_TIDAK_SEGAR = r"C:\Punya GW\Kuliah\ProjectKel11\dataset_final\training\tidak_segar"
FINAL_VALID_SEGAR = r"C:\Punya GW\Kuliah\ProjectKel11\dataset_final\validation\segar"
FINAL_VALID_TIDAK_SEGAR = r"C:\Punya GW\Kuliah\ProjectKel11\dataset_final\validation\tidak_segar"

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

# %%
# ==========================================
# 2. DATASET MAPPING RULES (MBG INGREDIENTS)
# ==========================================
# Dictionary pemetaan dari nama folder/file asli ke nama standar komoditas
DESIRED_COMMODITIES = {
    'spinach': 'spinach', 
    'corn': 'corn', 
    'chilli pepper': 'chilli', 
    'chilli': 'chilli',
    'capsicum': 'chilli',
    'bell pepper': 'chilli',
    'garlic': 'garlic', 
    'onion': 'onion',
    'banana': 'banana', 
    'tomato': 'tomato', 
    'orange': 'orange', 
    'apple': 'apple',
    'cabbage': 'cabbage', 
    'brinjal': 'eggplant', 
    'eggplant': 'eggplant', 
    'cucumber': 'cucumber', 
    'bean': 'bean',
    'potato': 'potato', 
    'carrot': 'carrot', 
    'papaya': 'papaya', 
    'mango': 'mango', 
    'watermelon': 'watermelon'
}

def get_category_and_commodity(path_str):
    """
    Returns (category, commodity_name) based on the full path string.
    Category is either 'segar', 'tidak_segar', or None (if it doesn't match).
    """
    path_lower = path_str.lower().replace('\\', '/')
    parts = path_lower.split('/')
    
    found_commodity = None
    
    # 1. Exact match (e.g., folder bernama 'spinach', 'papaya', 'mango')
    for part in reversed(parts):
        if part in DESIRED_COMMODITIES:
            found_commodity = DESIRED_COMMODITIES[part]
            break
            
    # 2. Prefix match (e.g., folder bernama 'freshbanana', 'rottenapples')
    if not found_commodity:
        for part in reversed(parts):
            for key, std_name in DESIRED_COMMODITIES.items():
                valid_names = [
                    f"fresh{key}", f"fresh{key}s", f"rotten{key}", f"rotten{key}s",
                    f"fresh_{key}", f"fresh_{key}s", f"rotten_{key}", f"rotten_{key}s"
                ]
                if part in valid_names:
                    found_commodity = std_name
                    break
            if found_commodity:
                break
                
    if not found_commodity:
        return None, None
        
    # Tentukan kondisi (Segar / Tidak Segar)
    is_rotten = any('rotten' in p or 'stale' in p or 'spoil' in p for p in parts)
    
    if is_rotten:
        return 'tidak_segar', found_commodity
    else:
        return 'segar', found_commodity

# %%
# ==========================================
# 3. DIRECTORY SETUP
# ==========================================
def create_directories():
    print("Creating required directories...")
    directories = [
        STAGING_SEGAR, STAGING_TIDAK_SEGAR,
        FINAL_TRAIN_SEGAR, FINAL_TRAIN_TIDAK_SEGAR,
        FINAL_VALID_SEGAR, FINAL_VALID_TIDAK_SEGAR
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directories created successfully!\n")

# %%
# ==========================================
# 4. EXTRACTION AND STAGING LOGIC
# ==========================================
def extract_and_stage_images():
    print("Starting image extraction to staging areas...")
    
    counts = {'segar': 0, 'tidak_segar': 0}
    
    for arc_id, arc_path in ARCHIVES.items():
        if not os.path.exists(arc_path):
            print(f"Warning: {arc_path} does not exist. Skipping.")
            continue
            
        print(f"Processing {arc_id}...")
        
        # Use os.walk for deep search
        for root, dirs, files in os.walk(arc_path):
            category, commodity = get_category_and_commodity(root)
            
            if category is None:
                continue # Bukan bahan baku MBG yang kita inginkan
                
            # Copy image to staging
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in VALID_EXTENSIONS:
                    source_file = os.path.join(root, file)
                    
                    # Create unique filename: archive1_spinach_0.jpg
                    unique_filename = f"{arc_id}_{commodity}_{counts[category]}{ext}"
                    
                    # Determine destination
                    if category == 'segar':
                        dest_file = os.path.join(STAGING_SEGAR, unique_filename)
                    else:
                        dest_file = os.path.join(STAGING_TIDAK_SEGAR, unique_filename)
                        
                    shutil.copy2(source_file, dest_file)
                    counts[category] += 1

    print(f"\nExtraction complete!")
    print(f"Total 'segar' images staged: {counts['segar']}")
    print(f"Total 'tidak_segar' images staged: {counts['tidak_segar']}\n")

# %%
# ==========================================
# 5. DATA SPLITTING LOGIC (80/20)
# ==========================================
def split_data():
    print("Starting data split (80% Train, 20% Validation)...")
    
    split_results = {
        'segar': {'train': 0, 'val': 0},
        'tidak_segar': {'train': 0, 'val': 0}
    }
    
    categories = [
        ('segar', STAGING_SEGAR, FINAL_TRAIN_SEGAR, FINAL_VALID_SEGAR),
        ('tidak_segar', STAGING_TIDAK_SEGAR, FINAL_TRAIN_TIDAK_SEGAR, FINAL_VALID_TIDAK_SEGAR)
    ]
    
    for cat_name, staging_dir, train_dir, val_dir in categories:
        if not os.path.exists(staging_dir):
            continue
            
        files = [f for f in os.listdir(staging_dir) if os.path.isfile(os.path.join(staging_dir, f))]
        
        random.seed(42)
        random.shuffle(files)
        
        split_idx = int(len(files) * 0.8)
        
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        for f in train_files:
            shutil.move(os.path.join(staging_dir, f), os.path.join(train_dir, f))
            split_results[cat_name]['train'] += 1
            
        for f in val_files:
            shutil.move(os.path.join(staging_dir, f), os.path.join(val_dir, f))
            split_results[cat_name]['val'] += 1
            
    return split_results

# %%
# ==========================================
# 6. SUMMARY LOGGING
# ==========================================
def print_summary(results):
    print("="*65)
    print(" "*20 + "DATASET SUMMARY CHART")
    print("="*65)
    print(f"{'Category':<15} | {'Training (80%)':<15} | {'Validation (20%)':<15} | {'Total':<10}")
    print("-" * 65)
    
    total_train = 0
    total_val = 0
    total_all = 0
    
    for cat in ['segar', 'tidak_segar']:
        train_count = results[cat]['train']
        val_count = results[cat]['val']
        cat_total = train_count + val_count
        
        total_train += train_count
        total_val += val_count
        total_all += cat_total
        
        print(f"{cat.capitalize():<15} | {train_count:<15} | {val_count:<15} | {cat_total:<10}")
        
    print("-" * 65)
    print(f"{'TOTAL':<15} | {total_train:<15} | {total_val:<15} | {total_all:<10}")
    print("="*65)

# %%
# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    create_directories()
    extract_and_stage_images()
    results = split_data()
    print_summary(results)

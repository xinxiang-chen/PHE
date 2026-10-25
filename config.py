import os

# Override with `export PHE_DATA_ROOT=/path/to/datasets` (e.g. on CARC)
data_root = os.environ.get('PHE_DATA_ROOT', '/workspace/datasets')

# --------------------
# DATASETS ROOT
# --------------------
cub_root = os.path.join(data_root, 'cub')
car_root = os.path.join(data_root, 'stanford_cars')
food_101_root = os.path.join(data_root, 'food101')
oxford_pet_root = os.path.join(data_root, 'oxford_pets')
inaturalist_root = os.path.join(data_root, 'iNaturalist')

# --------------------
# DINO PRETRAIN ROOT
# --------------------
pretrain_path = os.path.join(data_root, 'dino/dino_vitbase16_pretrain.pth')

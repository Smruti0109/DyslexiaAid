import os
import shutil
from sklearn.model_selection import train_test_split

# Corrected base directory where dyslexic/non_dyslexic images are located
base_dir = r'C:\Users\Smruti Deshpande\Desktop\Projects\Dyslexia_Detection-main\data'
output_dir = os.path.join(base_dir, 'split_data')  # output: data/split_data/

for class_name in ['dyslexic', 'non_dyslexic']:
    class_dir = os.path.join(base_dir, class_name)
    image_paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png'))]
    
    train_imgs, val_imgs = train_test_split(image_paths, test_size=0.2, random_state=42)

    for split, imgs in [('train', train_imgs), ('val', val_imgs)]:
        split_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for img in imgs:
            shutil.copy(img, split_dir)

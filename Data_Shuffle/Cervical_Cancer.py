import os
import random
import shutil


def sample_files(source_dir1, destination_dir1, sample_percentage):
    files1 = os.listdir(source_dir1)
    sample_size1 = int(len(files1) * sample_percentage)
    sampled_files1 = random.sample(files1, sample_size1)
    for file in sampled_files1:
        source_path = os.path.join(source_dir1, file)
        destination_path = os.path.join(destination_dir1, file)
        shutil.move(source_path, destination_path)



source_parent_folder = "./MultiCancer/Cervical_Cancer/"
destination_parent_folder = './MultiCancer/Cervical_Cancer/Test/'
sample_percentage = 0.1  # 10%


filename = ['cervix_dyk','cervix_koc','cervix_mep','cervix_pab','cervix_sfi']
for i in filename:
    source_folder = os.path.join(source_parent_folder, i)
    destination_folder = os.path.join(destination_parent_folder, i)
    os.makedirs(destination_folder, exist_ok=True)
    sample_files(source_folder, destination_folder, sample_percentage)

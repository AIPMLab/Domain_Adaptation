import os
import random
import shutil

def sample_files(source_dir1, source_dir2, destination_dir1,destination_dir2, sample_percentage):

    files1 = os.listdir(source_dir1)
    sample_size1 = int(len(files1) * sample_percentage)
    sampled_files1 = random.sample(files1, sample_size1)
    for file in sampled_files1:
        source_path = os.path.join(source_dir1, file)
        destination_path = os.path.join(destination_dir1, file)
        shutil.move(source_path, destination_path)


    files2 = os.listdir(source_dir2)
    sample_size2 = int(len(files2) * sample_percentage)
    sampled_files2 = random.sample(files2, sample_size2)
    for file in sampled_files2:
        source_path = os.path.join(source_dir2, file)
        destination_path = os.path.join(destination_dir2, file)
        shutil.move(source_path, destination_path)


source_folder1 = "./MultiCancer/Kidney_Cancer/kidney_normal/"
source_folder2 = "./MultiCancer/Kidney_Cancer/kidney_tumor/"
destination_folder1 = './MultiCancer/Kidney_Cancer/Test/kidney_normal'
destination_folder2 = './MultiCancer/Kidney_Cancer/Test/kidney_tumor'
sample_percentage = 0.1  # 10%
os.makedirs(destination_folder1, exist_ok=True)
os.makedirs(destination_folder2, exist_ok=True)
sample_files(source_folder1, source_folder2, destination_folder1, destination_folder2, sample_percentage)

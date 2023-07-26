import os
import shutil


txt_file = './COVID/train_COVIDx_CT-3A.txt'


source_folder = './COVID/images/'


target_folder = './COVID/train/'


if not os.path.exists(target_folder):
    os.makedirs(target_folder)


with open(txt_file, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip() 
        columns = line.split(' ')  

        filename = columns[0] 
        print(filename)
        label = columns[1]  


        source_path = os.path.join(source_folder, filename)


        target_path = os.path.join(target_folder, label)


        if not os.path.exists(target_path):
            os.makedirs(target_path)


        shutil.move(source_path, target_path)

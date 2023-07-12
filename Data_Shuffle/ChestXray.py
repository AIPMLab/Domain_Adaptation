import os
import shutil


source_folder = "./ChestXray"  
target_folder = "./ChestXray/test" 


with open("./ChestXray/test_list.txt", "r") as file:
    filenames = file.read().splitlines()  


for i in range(1, 13):
    folder_name = f"images_{i}"
    folder_path = os.path.join(source_folder, folder_name)
    if not os.path.isdir(folder_path):
        continue  


    new_folder_path = os.path.join(target_folder, folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    for filename in filenames:
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
           
            shutil.move(file_path, new_folder_path)

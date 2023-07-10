import os
import shutil
import csv


csv_file_path = './Product/test.csv'
with open(csv_file_path, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader) 


    for row in csv_reader:
        folder_name = row[1] 

    
        if not os.path.exists('./Product/test/' + folder_name):
            os.makedirs('./Product/test/' + folder_name)


        file_name = row[0] 
        file_path = './Product/test/test/' + file_name  
        folder_path = './Product/test/' + folder_name + '/' + file_name
        shutil.move(file_path, folder_path)

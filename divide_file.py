import os
import shutil
def get_filecount(path_to_directory):
    if os.path.exists(path_to_directory):
        path,dirs,files = os.walk(path_to_directory).__next__()
        file_count = len(files)
        return file_count
    else :
        print("path does not exist")
        return 0
os.makedirs('images_1')
os.makedirs('images_2')

count = get_filecount('images')
for i in range(count):
    if i <= count/2 :
        shutil.copy('images/'+str(i+1)+'.jpg', os.path.join("images_1", str(i+1) +'.jpg'))
    else:
        shutil.copy('images/' + str(i + 1) + '.jpg', os.path.join("images_2", str(i + 1) + '.jpg'))
print('finished')
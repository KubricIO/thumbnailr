from sklearn import model_selection
import os
import shutil

def data_split(dir,dest_path,folder):
    path, dirs, files = os.walk(dir).__next__()
    train,test = model_selection.train_test_split(files,test_size=0.1,train_size=0.9)
    print(test)
    for i in range(len(test)):
        shutil.copy2(dir+'/'+test[i], dest_path+'/test/'+folder)
    train, val = model_selection.train_test_split(train, test_size=0.1, train_size=0.9)
    print(val)
    for i in range(len(val)):
        shutil.copy2(dir + '/' + val[i], dest_path+'/val/'+folder)
    for i in range(len(train)):
        shutil.copy2(dir + '/' + train[i], dest_path + '/train/'+folder)
for i in range(5):
    data_split('/home/karthik/thumbnailr/mark_1/folder'+str(i+1),'/home/karthik/thumbnailr/mark_1','rate'+str(i+1))

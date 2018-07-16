import glob
import os
import shutil
def round_off(dirc,no_dp):
    list_of_files = glob.glob(dirc+'/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    for i in range(no_dp):
        shutil.copy(latest_file,dirc+'/'+'dub'+str(i+1)+'.jpg')
        i+=1

#round_off('./bad',4)



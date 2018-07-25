import os 
import shutil

def move_files(src, dst, no):
	path,dirs,files = os.walk(src).__next__()
	count=0
	for file in files:
		shutil.move(src+'/'+file, dst + '/999'+file)
		count+=1
		if(count==no):
			break

    

move_files("val/rate1","test/rate1",320)
move_files("val/rate2","test/rate2",320)
move_files("val/rate3","test/rate3",320)
#move_files("f3","f3_extra",325)
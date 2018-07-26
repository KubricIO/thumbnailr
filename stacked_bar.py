import numpy as np
#import tkinter
import matplotlib.pyplot as plt


N = 3
class1 = (387,204,81)
class2 = (198,338,136)
class3 = (92,106,474)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
acc = '96'
p1 = plt.bar(ind, class1, width, yerr=None)
p2 = plt.bar(ind, class2, width,bottom=class1, yerr=None)
p3 = plt.bar(ind, class3, width,bottom=[i+j for i,j in zip(class1, class2)], yerr=None)
plt.ylabel('Image count')
plt.title('Class wise Predicted Images | acc :'+acc)
plt.xticks(ind, ('Pred_class1', 'Pred_class2', 'Pred_class3'))
plt.legend((p1, p2,p3), ('Class1', 'Class2','Class3'))

#plt.show()
plt.savefig('figure1.jpg')
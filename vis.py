# Visualization Code
import os
import numpy as np
import random
import matplotlib.pyplot as plt
one = 0
two = 0
three = 0
four = 0
seven = 0
eight = 0
nine = 0
ten = 0
path1 = os.path.join(os.getcwd(),"aclImdb/train/pos")
path2 = os.path.join(os.getcwd(),"aclImdb/train/neg")
path3 = os.path.join(os.getcwd(),"aclImdb/test/pos")
path4 = os.path.join(os.getcwd(),"aclImdb/test/neg")
directory1 = os.listdir(path1)
directory2 = os.listdir(path2)
directory3 = os.listdir(path3)
directory4 = os.listdir(path4)
total_files_train = np.concatenate((directory1,directory2))
total_files_test = np.concatenate((directory3,directory4))
random.shuffle(total_files_train)
random.shuffle(total_files_test) 
total_files_train = total_files_train[:1000]
total_files_test = total_files_test[:1000]
 
def get_y(file_path):
    y_value = file_path.split('_')
    y_value = y_value[1].split('.')
    return y_value[0]

for file in total_files_train:
    if get_y(file) == '1':
       one = one + 1
    if get_y(file) == '2':
       two = two + 1
    if get_y(file) == '3':
       three = three + 1
    if get_y(file) == '4':
       four= four + 1
    if get_y(file) == '7':
       seven = seven + 1
    if get_y(file) == '8':
       eight = eight + 1
    if get_y(file) == '9':
       nine = nine + 1
    if get_y(file) == '10':
       ten = ten + 1
x1 = [1,2,3,4,7,8,9,10]
y1 = [one,two,three,four,seven,eight,nine,ten]
print "In training set, one = " + "{:.1f}".format(one) + ", Two = " + "{:.1f}".format(two) + ",Three = " + "{:.1f}".format(three) + ",Four = " + "{:.1f}".format(four) + ", Seven = " + "{:.1f}".format(seven) + ", Eight = " + "{:.1f}".format(eight) + ", Nine = " + "{:.1f}".format(nine) + ", Ten = " + "{:.1f}".format(ten)

one = 0
two = 0
three = 0
four = 0
seven = 0
eight = 0
nine = 0
ten = 0

for file in total_files_test:
    if get_y(file) == '1':
       one = one + 1
    if get_y(file) == '2':
       two = two + 1
    if get_y(file) == '3':
       three = three + 1
    if get_y(file) == '4':
       four= four + 1
    if get_y(file) == '7':
       seven = seven + 1
    if get_y(file) == '8':
       eight = eight + 1
    if get_y(file) == '9':
       nine = nine + 1
    if get_y(file) == '10':
       ten = ten + 1

print "In testing set, one = " + "{:.1f}".format(one) + ", Two = " + "{:.1f}".format(two) + ",Three = " + "{:.1f}".format(three) + ",Four = " + "{:.1f}".format(four) + ", Seven = " + "{:.1f}".format(seven) + ", Eight = " + "{:.1f}".format(eight) + ", Nine = " + "{:.1f}".format(nine) + ", Ten = " + "{:.1f}".format(ten)
x2 = [1,2,3,4,7,8,9,10]
y2 = [one,two,three,four,seven,eight,nine,ten]
p1=plt.bar(x1, y1, color='blue', edgecolor='yellow', width=1)
p2=plt.bar(x2, y2, color='green', edgecolor='red', width=1,bottom=y1)
plt.ylabel('Number of training points')
plt.xlabel('Labels')
plt.legend((p1[0], p2[0]), ('Train Data', 'Test Data'))
plt.title('Number of training points for each label in training dataset and test dataset')

plt.show()


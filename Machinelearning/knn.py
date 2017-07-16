'''

Akshay REDDYAK

Description of problem formulation, approach and programming methodologies

.) The program renders the training data and splits each line of information into
   [photoid,label] and [vectors]

.) Same method is used on the testing data, so that it will be easy for finding the 
   Eucliedian distance.


Approchs that I have tried:

Computing the Eucliedian distance is common in all the approches

1.) Comparing all the vectors of the test set with all the vectors of the training set.
    The program took a running time of 1.6 hr with an accuracy of around 67%.

2.) In order to increse the computaional time I tried using Threading and Multiprocessing.
    I tried dividing the test set in multiple threads and processes. But Python has 
    GIL(Global interpretor lock) which needed to activated inoreder to maked efficient use of 
    Threading and multiprogramming. It was a dramatic failure.:-(

3.) Then I focused on testing set. 
    .) Then I observed the image and found that skipping every alternate pixel
       will not cause lose of information. But this still took a lot of time, around 30 min.
       (I did not use this approcah)

       Below method I used:
    .) I tried many sort of things and finally came up with an idea of finding the variance between
       the vectors of the test image and setting a filter wndow.
    .) The filter function adds all the vectors in R+G+B format and stores in a list addative, 
       which has a size of 64.
    .) Then I have set a initial Threshold range of (100,-100) and if the variance out of this range then
       position is appended to the list called start. The Theroshold is again updated by adding a 100 to it.
       In order to avoide repeated calulation of same vectors I used condition where a position which is 
       more than 10 steps away is considered. 
    .) The filter() fuction will return the start indices which multiplying by 3 I can track the excact 
       position of R(red) vector. From this point I consider 3 more pixels (Filter window) by adding a 12 and 
       then calculate the Eculiedian distance.


'''




import sys
import math
import timeit
import random
from decimal import Decimal


def read_data(train_file, test_file):
	print "running knn"
	start1 = Decimal(timeit.default_timer(), 5)

	estimate = open("nearest_output.txt", "w")

	lables = ['0', '90', '180', '270']
	confusion_matrix = {'0': [0, 0, 0, 0], '90': [0, 0, 0, 0], '180': [0, 0, 0, 0], '270': [0, 0, 0, 0]}
	Store_Train_ID_Lable = []
	acurracy = 0

	train = open(train_file, "r").readlines()
	test = open(test_file, "r").readlines()

	for i in train:
		train_Photo = i.split()
		Store_Train_ID_Lable.append(train_Photo[:2])

	for i in test:
		start = Decimal(timeit.default_timer(), 5)
		Eucliden_Value = []
		test_Photo = i.split()
		start_value = filter(test_Photo[:len(test_Photo) - 20])

		for j in train:  # set range for training set here
			train_Photo = j.split()
			value = 0
			for s in start_value[1:]:
				for index in range(2 + (s * 3), 2 + (s * 3) + 12):
					value = value + math.pow(int(train_Photo[index]) - int(test_Photo[index]), 2)
			Eucliden_Value.append(math.sqrt(value))

		temp = Store_Train_ID_Lable[Eucliden_Value.index(min(Eucliden_Value))]
		print str(test_Photo[:2]) + " similar to " + str(temp)
		estimate.write(str(test_Photo[0]) + " " + str(temp[1]) + "\n")
		if test_Photo[1] == temp[1]:
			confusion_matrix[test_Photo[1]][lables.index(temp[1])] += 1
			acurracy += 1
		else:
			confusion_matrix[test_Photo[1]][lables.index(temp[1])] += 1

		end = Decimal(timeit.default_timer(), 5)
	# print "Excution time="+str(end-start)


	end1 = Decimal(timeit.default_timer(), 5)
	print "Excution time=" + str((end1 - start1) / 60) + str(" minutes\n")
	print "\naccuracy=" + str((acurracy / float(len(test))) * 100) + str("%\n")
	print "Confusion matrix\n"
	print str("Predicted    ") + "   ".join(lables)

	for i in lables:
		print str(i) + "           " + str(confusion_matrix[i])


def filter(test_Photo):
	addative=[]
	start=[0]
	i=2
	Threshold=100
	while(i<=len(test_Photo[2:])):
		addative.append(sum(int(x) for x in test_Photo[i:i+3]))
		i+=3
	for i in range(len(addative)-1):
		value=(addative[i]-addative[i+1])
		if (value>Threshold or value<-(Threshold)) and i>start[len(start)-1]+10:
			Threshold=(value)+100
			start.append(i)

	return start
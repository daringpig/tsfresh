#!/usr/bin/env python

train_file = "./train/X_train.txt"
with open(train_file,"ab+") as tf:
	test_file = "./test/X_test.txt"	
	with open(test_file,"rb") as rf:
		for line in rf.readlines():
			tf.write(line)

train_label = "./train/y_train.txt"
with open(train_label,"ab+") as tf:
	test_label = "./test/y_test.txt"
	with open(test_label,"rb") as rf:
		for line in rf.readlines():
			tf.write(line)

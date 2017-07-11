#!/usr/bin/env python

filelists = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
]

for filename in filelists:
	train_dir = "./train/Inertial Signals/"
	train_file = train_dir + filename + "train.txt"
	with open(train_file,"ab+") as tf:
		test_dir = "./test/Inertial Signals/"
		test_file = test_dir + filename + "test.txt"
		with open(test_file,"rb") as rf:
			for line in rf.readlines():
				tf.write(line)
				#tf.write('\n')

#train_label = "./train/y_train.txt"
#test_label = "./test/y_test.txt"
#with open(train_label,"ab+") as tf:
#	with open(test_label,"rb") as rf:
#		for line in rf.readlines():
#			tf.write(line)
			#tf.write('\n')
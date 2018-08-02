import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import *
import matplotlib.pyplot as plt

def read_data(dataset):
	all_features = []
	timestamp_list = []
	close_list = []
	high_list = []
	low_list = []
	open_price_list = []
	volume_list = []

	for line in dataset:
		l = line.split(',')
		x = list(l[len(l)-1])
		x = x[0:len(x)-1]
		x = ''.join(x)
		l[len(l)-1] = x
		all_features.append(l)
		y = list(l[0])
		for i in y:
			if(i=='-'):
				y.remove(i)
		y = ''.join(y)
		l[0]=y
		if(l[1])=='null':
			l[1]='0.0'
		if(l[4]=='null'):
			l[4]='0.0'
		if(l[2]=='null'):
			l[2]='0.0';
		if(l[3]=='null'):
			l[3]='0.0';
		if(l[5]=='null'):
			l[5]='0.0'
		timestamp, open_price , high, low, close,volume = l
		timestamp_list.append(int(timestamp))
		open_price_list.append(float(open_price))		
		close_list.append(float(close))
		high_list.append(float(high))
		low_list.append(float(low))
		volume_list.append(float(volume))
		
	return timestamp_list, open_price_list, high_list, low_list, close_list, volume_list


def create_labels(close_list,open_price_list):
	label_list = close_list - open_price_list
	label_list = label_list[1:-1]
	for i in range(len(label_list)):
		if(label_list[i]>0):
			label_list[i]=1
		else:
			label_list[i]=0
	return label_list
	
	
def feature_creation(timestamp_list, open_price_list, high_list, low_list, close_list, volume_list,no_of_days):
	open_change_percentage_list = []
	close_change_percentage_list = []
	low_change_percentage_list = []
	high_change_percentage_list = []
	volume_change_percentage_list = []
	
	volume_diff_percentage_list = []
	open_diff_percentage_list = []
	
	open_price_moving_average_list = []	
	close_price_moving_average_list = []
	high_price_moving_average_list = []
	low_price_moving_average_list =[]

	highest_open_price = open_price_list[0]
	lowest_open_price = open_price_list[0]
	highest_volume = volume_list[0]
	lowest_volume = volume_list[0]

	if(no_of_days>len(open_price_list)):
		no_of_days = len(open_price_list)
	for i in range(len(close_list)-no_of_days, len(close_list)):
		if(highest_open_price<open_price_list[i]):
			highest_open_price = open_price_list[i]
		if(lowest_open_price>open_price_list[i]):
			lowest_open_price = open_price_list[i]
		if(highest_volume < volume_list[i]):
			highest_volume = volume_list[i]
		if(lowest_volume > volume_list[i]):
			lowest_volume = volume_list[i]
		
	opensum = open_price_list[0]
	closesum = close_list[0]
	highsum = high_list[0]
	lowsum = low_list[0]
	
	for i in range(1,len(close_list)-1):
		if(close_list[i-1]==0):
			close_list[i-1]=close_list[i-2]
		close_change_percentage = (close_list[i] - close_list[i-1])/close_list[i-1]
		close_change_percentage_list.append(close_change_percentage)
	
		if(open_price_list[i]==0):
			open_price_list[i]=open_price_list[i-1]
		open_change_percentage = (open_price_list[i+1]-open_price_list[i])/open_price_list[i]
		open_change_percentage_list.append(open_change_percentage)
		
		if(high_list[i-1]==0):
			high_list[i-1]=high_list[i-2]
		high_change_percentage = (high_list[i]-high_list[i-1])/high_list[i-1]
		high_change_percentage_list.append(high_change_percentage)
		
		if(volume_list[i-1]==0):
			volume_list[i-1] = volume_list[i-2]
		volume_change_percentage = (volume_list[i]-volume_list[i-1])/volume_list[i-1]
		volume_change_percentage_list.append(volume_change_percentage)

		if(low_list[i-1]==0):
			low_list[i-1]=low_list[i-2]
		low_change_percentage = (low_list[i] - low_list[i-1])/low_list[i-1]
		low_change_percentage_list.append(low_change_percentage)

		volume_diff = (volume_list[i] - volume_list[i-1])/(highest_volume-lowest_volume)
		volume_diff_percentage_list.append(volume_diff)

	
		open_diff = (open_price_list[i+1]-open_price_list[i])/(highest_open_price-lowest_open_price)
		open_diff_percentage_list.append(open_diff)

		open_price_moving_average = float(opensum/i+1) 
		open_price_moving_average_list.append(open_price_moving_average)

		high_price_moving_average = float(highsum/i+1) 
		high_price_moving_average_list.append(high_price_moving_average)

		close_price_moving_average = float(closesum/i+1) 
		close_price_moving_average_list.append(close_price_moving_average)

		low_price_moving_average = float(lowsum/i+1) 
		low_price_moving_average_list.append(low_price_moving_average)


	''' Combine the above features '''
	Close_change_percentage_list = np.array(close_change_percentage_list)
	Open_change_percentage_list = np.array(open_change_percentage_list)
	High_change_percentage_list = np.array(high_change_percentage_list)
	Low_change_percentage_list = np.array(low_change_percentage_list)
	Volume_change_percentage_list = np.array(volume_change_percentage_list)
	open_price_list = np.array(open_price_list)
	close_list = np.array(close_list)
	Open_diff_percentage_list = np.array(open_diff_percentage_list)
	Volume_diff_percentage_list = np.array(volume_diff_percentage_list)
		
	''' Making set of features '''
	feature1 = np.column_stack((Open_change_percentage_list,Close_change_percentage_list,High_change_percentage_list,Low_change_percentage_list,Volume_change_percentage_list))
	feature2 = np.column_stack((Open_change_percentage_list,Close_change_percentage_list,High_change_percentage_list,Low_change_percentage_list,Volume_change_percentage_list,Open_diff_percentage_list,Volume_diff_percentage_list))
	feature3 = np.column_stack((Open_change_percentage_list,Close_change_percentage_list,High_change_percentage_list,Low_change_percentage_list,Volume_change_percentage_list,open_price_moving_average_list,close_price_moving_average_list,high_price_moving_average_list,low_price_moving_average_list))
	feature4 = np.column_stack((Open_change_percentage_list,Close_change_percentage_list,High_change_percentage_list,Low_change_percentage_list,Volume_change_percentage_list,open_price_moving_average_list,close_price_moving_average_list,high_price_moving_average_list,low_price_moving_average_list,Open_diff_percentage_list,Volume_diff_percentage_list))

	label_list = create_labels(close_list,open_price_list)
	return feature1, feature2, feature3, feature4, label_list


def checkForAccuracy(actual_open_list,open_price,high_price,low_price,close_price,volume,index):
	x0 = np.ones(len(open_price))
	x1 = np.array(close_price)
	x2 = np.array(high_price)
	x3 = np.array(low_price)
	x4 = np.array(volume)
	X = np.c_[x0,x1,x2,x3,x4]
	y = np.asmatrix(np.array(open_price).reshape(-1,1))
	X_T = np.transpose(X)
	X_pseudo = np.dot(np.linalg.inv(np.dot(X_T,X)),X_T)
	beta = X_pseudo * y
	t1 = close_price[len(close_price)-20:len(close_price)-1]
	t2 = high_price[len(high_price)-20:len(high_price)-1]
	t3 = low_price[len(low_price)-20:len(low_price)-1]
	t4 = volume[len(volume)-20:len(volume)-1]
	sum_t1=0
	sum_t2=0
	sum_t3=0
	sum_t4=0
	for i in t1:
		sum_t1 = sum_t1 + i
	mean_t1 = sum_t1/len(t1)
	for i in t2:
		sum_t2 = sum_t2 + i
	mean_t2 = sum_t2/len(t2)
	for i in t3:
		sum_t3 = sum_t3 + i
	mean_t3 = sum_t3/len(t3)
	for i in t4:
		sum_t4 = sum_t4 + i
	mean_t4 = sum_t4/len(t4)
	y = beta[0] + beta[1]*mean_t1 + beta[2]*mean_t2 + beta[3]*mean_t3 + beta[4]*mean_t4
	print("Newly predicted: {0}".format(y))
	print("Actual price: {0}".format(actual_open_list[index]))

def multivariate(open_price,high_price,low_price,close_price,volume):
	x0 = np.ones(len(open_price))
	x1 = np.array(close_price)
	x2 = np.array(high_price)
	x3 = np.array(low_price)
	x4 = np.array(volume)
	X = np.c_[x0,x1,x2,x3,x4]
	y = np.asmatrix(np.array(open_price).reshape(-1,1))
	X_T = np.transpose(X)
	X_pseudo = np.dot(np.linalg.inv(np.dot(X_T,X)),X_T)
	beta = X_pseudo * y
	print(X.shape)
	print(y.shape)
	print(beta)
	t1 = close_price[len(close_price)-20:len(close_price)-1]
	t2 = high_price[len(high_price)-20:len(high_price)-1]
	t3 = low_price[len(low_price)-20:len(low_price)-1]
	t4 = volume[len(volume)-20:len(volume)-1]
	sum_t1=0
	sum_t2=0
	sum_t3=0
	sum_t4=0
	for i in t1:
		sum_t1 = sum_t1 + i
	mean_t1 = sum_t1/len(t1)
	for i in t2:
		sum_t2 = sum_t2 + i
	mean_t2 = sum_t2/len(t2)
	for i in t3:
		sum_t3 = sum_t3 + i
	mean_t3 = sum_t3/len(t3)
	for i in t4:
		sum_t4 = sum_t4 + i
	mean_t4 = sum_t4/len(t4)
	y = beta[0] + beta[1]*mean_t1 + beta[2]*mean_t2 + beta[3]*mean_t3 + beta[4]*mean_t4
	print("Newly predicted: {0}".format(y))


def svm(feature,label_list):
	length_feature = len(feature)
	len_train = int(0.95*length_feature)
	train_feature = feature[0:len_train]
	test_feature = feature[len_train:]
	train_label = label_list[0:len_train]
	test_label = label_list[len_train:]
	clf = SVC (C=100000,kernel='rbf')
	clf.fit(train_feature,train_label)
	predicted = clf.predict(test_feature)
	print("Accuracy: ",accuracy_score(predicted,test_label)*100, "%")
	print("Precision: ",precision_score(predicted,test_label)*100,"%")
	print("Recall score: ",recall_score(predicted,test_label)*100,"%")
	return predicted,test_label


def plotSVM(predicted,test_labels,name):
	step = np.arange(0,len(test_labels))
	plt.subplot(211)
	plt.xlim(-1,len(test_labels)+1)
	plt.ylim(-1,2)
	plt.ylabel('Actual vlues')
	plt.plot(step,test_labels,linestyle='--', drawstyle='steps',color='blue')
	plt.subplot(212)
	plt.xlim(-1, len(test_labels) + 1)
	plt.ylim(-1, 2)
	plt.xlabel('minutes')
	plt.ylabel('Predicted Values')
	plt.plot(step, predicted, linestyle='--', drawstyle = 'steps',color='green')	
	plt.savefig(name)
	plt.show()

print("ENter 1 to start");
a = int(input());
if(a==1):
	fp = open('dataset/nse.csv')	
	timestamp_list, open_list, high_list, low_list, close_list , volume_list = read_data(fp)
	feature1, feature2, feature3, feature4, label_list = feature_creation(timestamp_list,open_list,high_list,low_list,close_list,volume_list,10)
	#regressionModel(feature1,open_list)
	multivariate(open_list,high_list,low_list,close_list,volume_list)
	print("\n\nSVM rbf with features:\nOpen change%, Close change %, High change %, Low Change%, Volume change%")
	predicted1,test_label1 = svm(feature1,label_list)
	plotSVM(predicted1,test_label1,"static/nse_one.jpg")

	print("\n\nSVM rbf with features:\nOpen change%, Close change %, High change %, Low Change%, Volume change%, open diff%, volume diff%")
	predicted2,test_label2 = svm(feature2,label_list)
	plotSVM(predicted2,test_label2,"static/nse_two.jpg")

	print("\n\nSVM rbf with features:\nOpen change%, Close change %, High change %, Low Change%, Volume change%, open moving avg, close moving avg, high moving avg,low moving avg")
	predicted3,test_label3 = svm(feature3,label_list)
	plotSVM(predicted3,test_label1,"static/nse_three.jpg")
	
	print("\n\nSVM rbf with features:\nOpen change%, Close change %, High change %, Low Change%, Volume change%, open diff%, volume diff%, open moving avg, close moving avg, high moving avg, low moving avg")
	predicted1,test_label1 = svm(feature1,label_list)
	plotSVM(predicted1,test_label1,"static/nse_four.jpg")

	
	date = int(input("To check the accuracy of our prediction \n Enter a date before 3rd march 2018\n Format (yyyymmdd) "))
	index=0
	for i in timestamp_list:
		if i == date:
			index = timestamp_list.index(i)
			break	
	check_open = open_list[:index]
	check_close = close_list[:index]
	check_high = high_list[:index]
	check_low  = low_list[:index]
	check_volume = volume_list[:index]
	checkForAccuracy(open_list,check_open,check_high,check_low,check_close,check_volume,index)
		


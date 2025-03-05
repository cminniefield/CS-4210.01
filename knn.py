#-------------------------------------------------------------------------
# AUTHOR: Caden Minniefield
# FILENAME: decision_tree_2.py
# SPECIFICATION: This program computes the LOO-CV error rate for a 1NN classifier on the spam/ham classification
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

# Map string labels ('ham', 'spam') to integers (0, 1)
label_mapping = {'ham': 0, 'spam': 1}

wrong_predictions = 0
total_predictions = 0

#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    X = []  # Training data
    Y = []  # Training labels
    # Prepare the training set by excluding the current test instance
    for j in range(len(db)):
        if i != j:  # Skip the current test instance
            X.append([float(x) for x in db[j][:-1]])  # Convert features to float
            Y.append(label_mapping[db[j][-1]])  # Convert label (ham/spam) to int

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here

    test_sample = i

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    #--> add your Python code here

    # Prepare the test sample (excluding the label)
    test_features = [float(x) for x in test_sample[:-1]]

    # Make the prediction for the test sample
    predicted_class = clf.predict([test_features])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here

    actual_label = label_mapping[test_sample[-1]]  # Convert the true label to integer
    if predicted_class != actual_label:
        wrong_predictions += 1

    total_predictions += 1

#Print the error rate
#--> add your Python code here
error_rate = wrong_predictions / total_predictions

# Print the error rate
print(f'LOO-CV error rate: {error_rate}')






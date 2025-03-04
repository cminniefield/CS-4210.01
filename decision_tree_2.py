#-------------------------------------------------------------------------
# AUTHOR: Caden Minniefield
# FILENAME: decision_tree_2.py
# SPECIFICATION: This problem creates a decision tree and tests it against contact lenses test data
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  #skipping the header
                dbTraining.append(row)

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here

    # Transform categorical training features into numbers using dictionaries
    age_mapping = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
    prescription_mapping = {"Myope": 1, "Hypermetrope": 2}
    astigmatism_mapping = {"Yes": 1, "No": 2}
    tear_rate_mapping = {"Reduced": 1, "Normal": 2}
    class_mapping = {"Yes": 1, "No": 2}

    # Transform and store feature vectors
    for entry in dbTraining:
        age_num = age_mapping[entry[0]]
        pres_num = prescription_mapping[entry[1]]
        astig_num = astigmatism_mapping[entry[2]]
        tear_num = tear_rate_mapping[entry[3]]

        X.append([age_num, pres_num, astig_num, tear_num])  # Store as a single row

    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    #--> add your Python code here

    Y = [class_mapping[entry[-1]] for entry in dbTraining]

    correct_predictions = 0
    total_predictions = 0
    #Loop your training and test tasks 10 times here
    for i in range(10):

        #Fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        #Read the test data and add this data to dbTest
        #--> add your Python code here

        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # skipping the header
                    dbTest.append(row)

        for data in dbTest:
            # Transform the features of the test instances to numbers following the same strategy done during training,
            # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here

            age_num = age_mapping[data[0]]
            pres_num = prescription_mapping[data[1]]
            astig_num = astigmatism_mapping[data[2]]
            tear_num = tear_rate_mapping[data[3]]
            actual_label = class_mapping[data[4]]

            # Make prediction
            class_predicted = clf.predict([[age_num, pres_num, astig_num, tear_num]])[0]

            #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here

            # Check if prediction matches actual label
            if class_predicted == actual_label:
                correct_predictions += 1
            total_predictions += 1

    #Find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here

    accuracy = correct_predictions / total_predictions

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here

    print(f"{ds}: " + str(accuracy))

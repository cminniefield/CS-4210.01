#-------------------------------------------------------------------------
# AUTHOR: Caden Minniefield
# FILENAME: naive_bayes.py
# SPECIFICATION: This program that will read the file
# weather_training.csv (training set) and output the classification of each of the 10 instances from
# the file weather_test (test set) if the classification confidence is >= 0.75.
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here

X = []  # Training features (4D array)
Y = []  # Training labels (1D array)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here

# Map categorical values to numbers
weather_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
outlook_map = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity_map = {'High': 1, 'Low': 2, 'Normal': 3}
wind_map = {'Weak': 1, 'Strong': 2}
play_map = {'Yes': 1, 'No': 2}

# Reading the CSV file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    for row in reader:
        # Map the feature columns to numbers and append them to X
        feature = [
            weather_map[row[1]],  # Outlook
            outlook_map[row[2]],  # Temperature
            humidity_map[row[3]],  # Humidity
            wind_map[row[4]]  # Wind
        ]
        X.append(feature)

        # Map the label (play) to a number and append it to Y
        Y.append(play_map[row[5]])

#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here

test_data = []  # List to hold the test data instances
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    for row in reader:
        # Map the feature columns to numbers and store the test instances
        feature = [weather_map[row[1]], outlook_map[row[2]], humidity_map[row[3]], wind_map[row[4]]]
        test_data.append(feature)

#Printing the header os the solution
#--> add your Python code here

# 4. Print the header of the solution
print("Test Sample Predictions:")
print("Instance | Prediction (1: Yes, 2: No) | Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for i, test_sample in enumerate(test_data):
    # Get the probability of the class predictions
    probs = clf.predict_proba([test_sample])[0]

    # Check if the confidence for the predicted class is >= 0.75
    predicted_class = clf.predict([test_sample])[0]  # Get the predicted class (1 or 2)

    if max(probs) >= 0.75:
        # Print the result (instance number, prediction, confidence)
        print(f"{i + 1} | {'Yes' if predicted_class == 1 else 'No'} | {max(probs):.2f}")


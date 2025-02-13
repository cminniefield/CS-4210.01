#-------------------------------------------------------------------------
# AUTHOR: Caden Minniefield
# FILENAME: decision_tree.py
# SPECIFICATION: This program reads a table of values and uses machine learning to determine if a person is to be recommended lenses
# FOR: CS 4210- Assignment #1
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('../../Downloads/contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

#transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
#--> add your Python code here

# Transform categorical training features into numbers using dictionaries
age_mapping = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
prescription_mapping = {"Myope": 1, "Hypermetrope": 2}
astigmatism_mapping = {"Yes": 1, "No": 2}
tear_rate_mapping = {"Reduced": 1, "Normal": 2}
class_mapping = {"Yes": 1, "No": 2}

# Transform and store feature vectors
for entry in db:
    age_num = age_mapping[entry[0]]
    pres_num = prescription_mapping[entry[1]]
    astig_num = astigmatism_mapping[entry[2]]
    tear_num = tear_rate_mapping[entry[3]]

    X.append([age_num, pres_num, astig_num, tear_num])  # Store as a single row

#transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
#--> addd your Python code here
Y = [class_mapping[entry[-1]] for entry in db]

#fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

#plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()
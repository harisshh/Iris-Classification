//Importing the necessary packages
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

//Read the contents from the dataset
df = pd.read_csv('Iris.csv')
df.describe()

//Depict the same in graphs
df['PetalWidthCm'].plot.hist()
plt.show()

sns.pairplot(df, hue='Species')

//Separate the input and target in two variables
all_inputs = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
all_classes = df['Species'].values

//Separate testing and training data
(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=1)

//Use Decision Tee Algorithm to classify the flowers
dtc = DecisionTreeClassifier()
dtc.fit(train_inputs, train_classes)
dtc.score(test_inputs, test_classes)

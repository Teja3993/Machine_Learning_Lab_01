#Task 1
#We apply log, square root, and Box-Cox transformations.
#Results are evaluated using histograms and Shapiro-Wilk p-values



#Load the Iris dataset
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#print(iris.feature_names)


#Feature 2 : Sepal Length.

#Applying Transformation to feature "Sepal Length"
import numpy as np
from scipy.stats import boxcox
log_transformed_f2 = np.log1p(df['sepal length (cm)'])
sqrt_transformed_f2 = np.sqrt(df['sepal length (cm)'])
boxcox_transformed_f2,_ = boxcox(df['sepal length (cm)']+1e-6)


#LOG-TRANSFOM

#Check normality test using shapiro-Wilk test
from scipy.stats import shapiro
print(shapiro(df['sepal length (cm)'].sample(100)))
print(shapiro(pd.Series(log_transformed_f2).sample(100)))

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(log_transformed_f2, kde=True)
plt.title('Log-transformed Sepal length')
plt.show()




#SQRT-TRANSFORM

#Check normality test using shapiro-Wilk test
print(shapiro(df['sepal length (cm)'].sample(100)))
print(shapiro(pd.Series(sqrt_transformed_f2).sample(100)))

#Visualization
sns.histplot(sqrt_transformed_f2, kde=True)
plt.title('Sqrt-transformed Sepal length')
plt.show()



#BOXCOX-TRANSFORM

#Check normality test using shapiro-Wilk test
print(shapiro(df['sepal length (cm)'].sample(100)))
print(shapiro(pd.Series(boxcox_transformed_f2).sample(100)))

#Visualization
sns.histplot(boxcox_transformed_f2, kde=True)
plt.title('BoxCox-transformed Sepal length')
plt.show()




#Feature 4 : Petal Length.

#Applying Transformation to feature "Petal Length"
log_transformed_f4 = np.log1p(df['petal length (cm)'])
sqrt_transformed_f4 = np.sqrt(df['petal length (cm)'])
boxcox_transformed_f4,_ = boxcox(df['petal length (cm)']+1e-6)



#LOG-TRANSFOM

#Check normality test using shapiro-Wilk test
print(shapiro(df['petal length (cm)'].sample(100)))
print(shapiro(pd.Series(log_transformed_f4).sample(100)))

#Visualization
sns.histplot(log_transformed_f4, kde=True)
plt.title('Log-transformed Petal length')
plt.show()




#SQRT-TRANSFORM

#Check normality test using shapiro-Wilk test
print(shapiro(df['petal length (cm)'].sample(100)))
print(shapiro(pd.Series(sqrt_transformed_f4).sample(100)))

#Visualization
sns.histplot(sqrt_transformed_f4, kde=True)
plt.title('Sqrt-transformed Petal length')
plt.show()




#BOXCOX-TRANSFORM

#Check normality test using shapiro-Wilk test
print(shapiro(df['petal length (cm)'].sample(100)))
print(shapiro(pd.Series(boxcox_transformed_f4).sample(100)))

#Visualization
sns.histplot(boxcox_transformed_f4, kde=True)
plt.title('Boxcox-transformed Petal length')
plt.show()



#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing necessary packages
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt


# In[2]:


#Loading the data into Pandas Dataframe
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target


# In[3]:


#Creating boxplot for Sepal Length, seperated by Species
sepal_length_boxplot = plt.figure()
sepal_length_boxplot = df.boxplot(column="sepal length (cm)", by = "species")
plt.title("Sepal Length Boxplot")
plt.savefig("sepal_length_boxplot.png", format="png")


# In[4]:


#Creating boxplot for Sepal Width, seperated by Species
sepal_width_boxplot = plt.figure()
output_boxplot2 = df.boxplot(column="sepal width (cm)", by = "species")
plt.title("Sepal Width Boxplot")
plt.savefig("sepal_width_boxplot.png", format="png")


# In[5]:


#Creating boxplot for Petal Length, seperated by Species
petal_length_boxplot = plt.figure()
output_boxplot3 = df.boxplot(column="petal length (cm)", by = "species")
plt.title("Petal Length Boxplot")
plt.savefig("petal_length_boxplot.png", format="png")


# In[6]:


#Creating boxplot for Petal Width, seperated by Species
petal_width_boxplot = plt.figure()
output_boxplot4 = df.boxplot(column="petal width (cm)", by = "species")
plt.title("Petal Width Boxplot")
plt.savefig("petal_width_boxplot.png", format="png")


# In[7]:


#Setting the colormap for scatterplots
colors = {0:"red", 1:"green", 2:"yellow"}


# In[8]:


#Creating the scatterplot for Petal Length vs Petal Width and saving as PNG file.
fig = plt.figure()
petal_plot = fig.add_subplot()
petal_plot.scatter(df["petal length (cm)"], df["petal width (cm)"], c=df["species"].map(colors))
plt.title("Scatterplot of Petal Length vs Petal Width")
fig.savefig("scatter_petal.png", format="png")


# In[9]:


#Creating the scatterplot for Petal Length vs Petal Width and saving as PNG file.
fig = plt.figure()
sepal_plot = fig.add_subplot()
sepal_plot.scatter(df["sepal length (cm)"], df["sepal width (cm)"], c=df["species"].map(colors))
plt.title("Scatterplot of Sepal Length vs Sepal Width")
fig.savefig("scatter_sepal.png", format="png")


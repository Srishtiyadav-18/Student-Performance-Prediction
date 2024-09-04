#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv('StudentsPerformance.csv')       # Dataset from kaggle
print(data.shape)
data.head()


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# # EXPLORING THE DATA FEATURES

# In[6]:


data['gender'].value_counts(normalize = True)   # Counting the relative frequencies of there occurences.


# In[7]:


data['gender'].value_counts(dropna = False).plot.bar(color = 'red')
plt.title('Comparison of Males and Females')
plt.xlabel('gender')
plt.ylabel('count')
plt.show()              # Visualising the number of male and female in the dataset


# In[8]:


print(data['race/ethnicity'].value_counts(normalize = True))
print()
print(data['race/ethnicity'].value_counts())


# In[9]:


data['race/ethnicity'].value_counts(dropna = False).plot.bar(color = 'orange')
plt.title('Comparison of various groups in class')
plt.xlabel('Groups')
plt.ylabel('count')
plt.show()                    # Visualizing the different groups in the dataset


# In[10]:


print(data['parental level of education'].value_counts(normalize = True))
print()
data['parental level of education'].value_counts(dropna = False).plot.bar()
plt.title('Comparison of Parental Education')
plt.xlabel('Degree')
plt.ylabel('Count')
plt.show()               # Visualizing the different parental education levels


# In[11]:


print(data['lunch'].value_counts(normalize = True))
print()
data['lunch'].value_counts(dropna = False).plot.bar(color = 'yellow')
plt.title('Comparison of different types of lunch')
plt.xlabel('types of lunch')
plt.ylabel('count')
plt.show()                  # Visualizing different types of lunches


# In[12]:


print(data['math score'].value_counts(normalize = True).unique())
print()
print(data['math score'].value_counts(normalize = True))
print()
data['math score'].value_counts(dropna = False).plot.bar(figsize = (18, 10))
plt.title('Comparison of math scores')
plt.xlabel('score')
plt.ylabel('count')
plt.show()                 # Visualizing maths score of students


# In[13]:


print(data['reading score'].value_counts(normalize = True))
print()
data['reading score'].value_counts(dropna = False).plot.bar(figsize = (18, 10), color = 'orange')
plt.title('Comparison of math scores')
plt.xlabel('score')
plt.ylabel('count')
plt.show()                   # Visualizing reading score


# In[14]:


print((data['writing score'].value_counts(normalize = True)))
print()
data['writing score'].value_counts(dropna = False).plot.bar(figsize = (18, 10), color = 'pink')
plt.title('Comparison of writing scores')
plt.xlabel('score')
plt.ylabel('count')
plt.show()                      # Visualizing writing score


# In[15]:


# Gender vs Etnicity 

x = pd.crosstab(data['gender'], data['race/ethnicity'])
x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (6, 6))


# In[16]:


# Comparison of parental degree and test course

sns.countplot(x = 'parental level of education', data = data, hue = 'test preparation course', palette = 'dark')
plt.show()


# In[17]:


# Feature Engineering on the data to visualize and solve the dataset more accurately
# Setting a passing mark for the students to pass on the three subjects individually
passmarks = 40

# creating a new column pass_math, this column will tell us whether the students are pass or fail
data['pass_math'] = np.where(data['math score']< passmarks, 'Fail', 'Pass')
data['pass_math'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (5, 3))

plt.title('Comparison of students passed or failed in maths')
plt.xlabel('status')
plt.ylabel('count')
plt.show()
print()
print(data['pass_math'].value_counts())


# In[18]:


# creating a new column pass_math, this column will tell us whether the students are pass or fail
data['pass_reading'] = np.where(data['reading score']< passmarks, 'Fail', 'Pass')
data['pass_reading'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (5, 3))

plt.title('Comparison of students passed or failed in maths')
plt.xlabel('status')
plt.ylabel('count')
plt.show()
print()
print(data['pass_reading'].value_counts(dropna = False))


# In[19]:


# creating a new column pass_math, this column will tell us whether the students are pass or fail
data['pass_writing'] = np.where(data['writing score']< passmarks, 'Fail', 'Pass')
data['pass_writing'].value_counts(dropna = False).plot.bar(color = 'blue', figsize = (5, 3))

plt.title('Comparison of students passed or failed in maths')
plt.xlabel('status')
plt.ylabel('count')
plt.show()
print()
print(data['pass_writing'].value_counts(dropna = False))


# In[20]:


# computing the total score for each student

data['total_score'] = data['math score'] + data['reading score'] + data['writing score']

data['total_score'].value_counts(normalize = True)
data['total_score'].value_counts(dropna = True).plot.bar(color = 'cyan', figsize = (40, 8))

plt.title('comparison of total score of all the students')
plt.xlabel('total score scored by the students')
plt.ylabel('count')
plt.show()


# In[21]:


# computing percentage for each of the students

from math import * 
data['percentage'] = data['total_score']/3
for i in range(0, 1000):
  data['percentage'][i] = ceil(data['percentage'][i])
data['percentage'].value_counts(normalize = True)
data['percentage'].value_counts(dropna = False).plot.bar(figsize = (16, 8), color = 'red')
plt.title('Comparison of percentage scored by all the students')
plt.xlabel('percentage score')
plt.ylabel('count')
plt.show()


# In[22]:


# checking which student is fail overall

data['status'] = data.apply(lambda x : 'Fail' if x['pass_math'] == 'Fail' or 
                           x['pass_reading'] == 'Fail' or x['pass_writing'] == 'Fail'
                           else 'pass', axis = 1)

data['status'].value_counts(dropna = False).plot.bar(color = 'gray', figsize = (3, 3))
plt.title('overall results')
plt.xlabel('status')
plt.ylabel('count')
plt.show()


# In[23]:


# Assigning grades to the grades according to the following criteria :
# 0  - 40 marks : grade E
# 41 - 60 marks : grade D
# 60 - 70 marks : grade C
# 70 - 80 marks : grade B
# 80 - 90 marks : grade A
# 90 - 100 marks : grade O

def getgrade(percentage, status):
  if status == 'Fail':
    return 'E'
  if(percentage >= 90):
    return 'O'
  if(percentage >= 80):
    return 'A'
  if(percentage >= 70):
    return 'B'
  if(percentage >= 60):
    return 'C'
  if(percentage >= 40):
    return 'D'
  else :
    return 'E'

data['grades'] = data.apply(lambda x: getgrade(x['percentage'], x['status']), axis = 1 )

data['grades'].value_counts()


# In[24]:


# plotting a pie chart for the distribution of various grades amongst the students

labels = ['Grade 0', 'Grade A', 'Grade B', 'Grade C', 'Grade D', 'Grade E']
sizes = [58, 156, 260, 252, 223, 51]
colors = ['yellow', 'gold', 'lightskyblue', 'lightcoral', 'pink', 'cyan']
explode = (0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001)

patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels)
plt.axis('equal')
plt.tight_layout()
plt.show()


# # FEATURE ENGINERRING

# In[25]:


from sklearn.preprocessing import LabelEncoder

# creating an encoder
le = LabelEncoder()

# label encoding for test preparation course
data['test preparation course'] = le.fit_transform(data['test preparation course'])
data['test preparation course'].value_counts()


# In[26]:


# label encoding for lunch

data['lunch'] = le.fit_transform(data['lunch'])
data['lunch'].value_counts()


# In[27]:


# label encoding for race/ethnicity
# we have to map values to each of the categories

data['race/ethnicity'] = data['race/ethnicity'].replace('group A', 1)
data['race/ethnicity'] = data['race/ethnicity'].replace('group B', 2)
data['race/ethnicity'] = data['race/ethnicity'].replace('group C', 3)
data['race/ethnicity'] = data['race/ethnicity'].replace('group D', 4)
data['race/ethnicity'] = data['race/ethnicity'].replace('group E', 5)

data['race/ethnicity'].value_counts()


# In[28]:


# label encoding for parental level of education

data['parental level of education'] = le.fit_transform(data['parental level of education'])
data['parental level of education'].value_counts()


# In[29]:


# label encoding for gender

data['gender'] = le.fit_transform(data['gender'])
data['gender'].value_counts()


# In[30]:


# label encoding for pass_math

data['pass_math'] = le.fit_transform(data['pass_math'])
data['pass_math'].value_counts()


# In[31]:


# label encoding for pass_reading

data['pass_reading'] = le.fit_transform(data['pass_reading'])
data['pass_reading'].value_counts()


# In[32]:


# label encoding for pass_writing

data['pass_writing'] = le.fit_transform(data['pass_writing'])
data['pass_writing'].value_counts()


# In[33]:


# label encoding for status

data['status'] = le.fit_transform(data['status'])
data['status'].value_counts()


# In[34]:


# label encoding for grades
# we have to map values to each of the categories

data['grades'] = data['grades'].replace('O', 0)
data['grades'] = data['grades'].replace('A', 1)
data['grades'] = data['grades'].replace('B', 2)
data['grades'] = data['grades'].replace('C', 3)
data['grades'] = data['grades'].replace('D', 4)
data['grades'] = data['grades'].replace('E', 5)

data['race/ethnicity'].value_counts()


# In[35]:


data.shape


# In[36]:


# splitting the dependent and independent variables

x = data.iloc[:,:14]
y = data.iloc[:,14]

print(x.shape)
print(y.shape)


# In[37]:


# splitting the dataset into training and test sets

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 45)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[38]:


# importing the MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# creating a scaler
mm = MinMaxScaler()

# feeding the independent variable into the scaler
x_train = mm.fit_transform(x_train)
x_test = mm.transform(x_test)


# # Logistic Regression

# In[39]:


from sklearn.linear_model import  LogisticRegression

# creating a model
model = LogisticRegression()

# feeding the training data to the model
model.fit(x_train, y_train)

# predicting the test set results
y_pred = model.predict(x_test)

# calculating the classification accuracies
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))


# In[40]:


# printing the confusion matrix

from sklearn.metrics import confusion_matrix

# creating a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# printing the confusion matrix
print(cm)


# In[41]:


from sklearn.ensemble import RandomForestClassifier

# creating a model
model = RandomForestClassifier()

# feeding the training data to the model
model.fit(x_train, y_train)

# predicting the x-test results
y_pred = model.predict(x_test)

# calculating the accuracies
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))


# In[42]:


# printing the confusion matrix

from sklearn.metrics import confusion_matrix

# creating a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# printing the confusion matrix
print(cm)


# In[43]:


from sklearn.tree import DecisionTreeClassifier

# creating a model
model = DecisionTreeClassifier()

# feeding the training data to the model
model.fit(x_train, y_train)

# predicting the x-test results
y_pred = model.predict(x_test)

# calculating the accuracies
print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))


# In[44]:


# printing the confusion matrix

from sklearn.metrics import confusion_matrix

# creating a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# printing the confusion matrix
print(cm)


# # CREATING A DATASET

# In[45]:


import numpy as np
import pandas as pd
import random

num_rows = 120
student_ids = [1, 2, 3, 4, 5,6,7,8,9,10]
names = ['Alice', 'Bob', 'Charlie', 'David', 'Emily']
ages = [20, 18, 19,17,]
genders = ['Female', 'Male']
grades = [12, 11, 12, 10, 11,19,17,16,'',None]
studying_class=['Tenth','Twelfth']
course=['completed',None]
phone_numbers = ['123-456-7890', '234-567-8901', '(345) 678-9012', '4567890123', '567-890-1234', None]

stud_data = {
    'student_ID': [random.choice(student_ids) for _ in range(num_rows)],
    'name': [random.choice(names) for _ in range(num_rows)],
    'age': [random.choice(ages) for _ in range(num_rows)],
    'gender': [random.choice(genders) for _ in range(num_rows)],
    'grade': [random.choice(grades) for _ in range(num_rows)],
    'class': [random.choice(studying_class) for _ in range(num_rows)],
    'course':[random.choice(course) for _ in range(num_rows)],
    'Phone Number': [random.choice(phone_numbers) for _ in range(num_rows)]
}

# Convert the NumPy array to a Pandas DataFrame
df = pd.DataFrame(stud_data)

df.to_csv('C:\just folder\studfile.csv', index=False)


# In[46]:


df=pd.read_csv("studfile.csv")
df


# In[47]:


df.info()


# # FEATURE ENGINERRING

# In[48]:


print(df.shape)
df.isnull().sum()


# In[49]:


for i in df.columns:
    print(i,df[i].unique())
    print ("----------------------")


# In[50]:


df.isnull().sum()


# In[51]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# label encoding for test preparation course
df['course'] = le.fit_transform(df['course'])
df['course'].value_counts()


# In[52]:


df.info()


# In[53]:


df['class'].replace(['Tenth'], 10, inplace=True)
df['class'].replace(['Twelfth'], 11, inplace=True)
df['gender'].replace(['Male'], 0, inplace=True)
df['gender'].replace(['Female'], 1, inplace=True)
df


# In[54]:


df.describe()


# In[55]:


df['grade'].fillna(df['grade'].median(), inplace=True)
df['grade'].isnull().sum()


# In[56]:


df.info()


# In[57]:


new_df=df.drop(columns=['student_ID','name','Phone Number'])
new_df.head()


# In[58]:


new_df1=new_df.drop('grade',axis='columns')
new_df1.head()


# In[59]:


grade=df.grade
grade


# # REGRESSION

# In[60]:


reg=linear_model.LinearRegression()
reg.fit(new_df1,grade)


# In[61]:


#PREDICT THE GRADE OF THE STUDENT WITH age=19,gender=0(MALE),class=10

reg.predict([[19,0,10,1]])


# In[62]:


reg.coef_


# In[63]:


reg.intercept_


# In[64]:


19*(0.1844531) + 0*(-0.00571867) + 10* 0.4145057 + 1*(-0.13456677) +5.391926043238934


# In[65]:


# Comparing the distribution of grades among males and females
# MALE-0 ,FEMALE-1
sns.countplot(x = df['grade'], data = df, hue = df['gender'], palette = 'cubehelix')
plt.show()


# In[66]:


df.head()


# In[67]:


df.describe()


# In[ ]:





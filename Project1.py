#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


insurance = pd.read_csv("data.csv")
insurance


# In[3]:


insurance.info()


# In[3]:


insurance.head()


# In[8]:


insurance.describe()


# We can infer somethings by observing the above data:-
#  1. The min age is 18 and max age in 64 so maybe the company doesn't consider the people for insurance having age less the 18 or more than 64.
#  2. bmi tells us that there are wide range of people from severely malnutrioned to severe obese people.
#  3. Min and Max charges vary alot. Even the price difference between 75 percentile nad the max price is very high.

# In[5]:


import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


insurance.hist(bins = 50,figsize = (20,20))


# In[10]:


graph = px.histogram(insurance,x='charges',marginal = 'box',nbins = 50)
graph.update_layout(bargap = 0.1)
graph.show()


# In[11]:


insurance.charges.describe()


# In[12]:


insurance.age.describe()


# In[6]:


graph = px.histogram(insurance,x='age',marginal = 'box',nbins = 50)
graph.update_layout(bargap = 0.1)
graph.show()


# We can see that the distribution of age is fairly uniform except for the ages 18 and 19. The reason has not been mentioned but may be the company is offering the people between ages 18 and 19 something extra than the older people.

# In[13]:


insurance.bmi.describe()


# In[14]:


graph = px.histogram(insurance,x='bmi',marginal = 'box',nbins = 50)
graph.update_layout(bargap = 0.1)
graph.show()


# This looks like a gausian or normal distribution

# In[16]:


graph = px.histogram(insurance,x='charges',marginal = 'box',color = 'smoker',nbins = 50)
graph.update_layout(bargap = 0.1)
graph.show()


# In[17]:


graph = px.histogram(insurance,x='charges',color = 'region',marginal = 'box',nbins = 50)
graph.update_layout(bargap = 0.1)
graph.show()


# From the above two graphs:
# 1. Non smokers pay lesser bill than the smokers
# 2. Region doesn't have and clear affect on the medical bills.

# In[19]:


px.violin(insurance,x='children',y='charges')


# There is a week increasing trend in the charges as the number of children increases.

# Age,smoker,bmi seems to be the factors most affecting the medical bills

# In[20]:


px.scatter(insurance,x='age',y='charges',color = 'smoker')


# In[21]:


px.scatter(insurance,x='bmi',y='charges',color = 'smoker')


# In the scatter plot of charges with age; we can clearly see 3 distinct linear trends; bottom most being the non smoker region and the above two are smoker region.
# 
# In scatter plot of charges with bmi; there is no specific trend but again we see three distinct regions.
# 
# Thus we may infer that in the scatter plot of charges with age; the bottomost linear trend is of the non smoker; the middle one is of the smoker with lesser bmi(less than 30 approx) and the top one is the smoker with more bmi(more than 30 approx)

# In[24]:


insurance.corr()


# In[26]:


px.imshow(insurance.corr(),text_auto = True)


# By the above heatmap it is clear that the correlation of charges with age is the highest.

# Now to train my model I will divide my datasets into 3 categories:
#  1. Non smokers
#  2. Smokers with bmi over 30
#  3. Smoker with bmi lesser than 30
# I will we using the strategy of Ordinary Least Squares(LinearRegression)
# 
# I will compute the loss using root mean squared value.
# Loss lesser than 5000-6000 will be optimum.

# In[27]:


def estimate_charges(age,w,b):
    return w*age+b


# In[28]:


non_smoker_df = insurance[insurance.smoker == 'no']


# In[29]:


import numpy as np


# In[30]:


def rmse(targets,predictions):
    return np.sqrt(np.mean(np.square(targets-predictions)))


# In[31]:


def try_parameters(w,b):
    ages = non_smoker_df.age
    target = non_smoker_df.charges
    predictions = estimate_charges(ages,w,b)
    plt.plot(ages,predictions,'r');
    plt.scatter(ages,target);
    plt.xlabel('Age');
    plt.ylabel('Charges')
    plt.legend(['Prediction','Actual']);
    loss = rmse(target,predictions)
    print("RMSE Loss: ",loss)


# In[32]:


try_parameters(300,-5000)


# # Linear Regression using Scikit-learn

# Scikit-learn provides us with ready made libraries to implement strategies like Ordinary Least Squares or Stochastic gradient descent. 
# 

# In[33]:


from sklearn.linear_model import LinearRegression


# In[34]:


model = LinearRegression()


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


inputs_nonsmoker = non_smoker_df[['age']]
targets_nonsmoker = non_smoker_df.charges
inputs,inputs_test,targets,targets_test = train_test_split(inputs_nonsmoker,targets_nonsmoker,test_size = 0.1)


# In[42]:


model.fit(inputs,targets)


# In[59]:


predictions = model.predict(inputs_test)


# In[60]:


predictions


# In[61]:


rmse(targets_test,predictions)


# This rmse loss is the lowest till now. Plus there are outliers which effects the loss considerably.

# In[46]:


model.coef_ #weight


# In[47]:


model.intercept_ #bias


# In[48]:


try_parameters(model.coef_,model.intercept_)


# Till now we worked with the data having just the non smokers. Let's work with the data containing the smokers.

# In[50]:


smokers_df = insurance[insurance.smoker == 'yes']


# In[51]:


smokers_df


# In[32]:


ages = smokers_df.age
target = smokers_df.charges


# In[52]:


px.scatter(smokers_df,x='age',y='charges')


# Unlike non smokers data here we can clearly see distribution of smokers into two distinct lines. Lets include the third factor which is actually affecting the dependecies of smokers and their medical bills.

# In[53]:


px.scatter(insurance,x = 'bmi',y = 'charges',color = 'smoker')


# In[54]:


high_smoker = smokers_df[smokers_df.bmi >= 30]
low_smoker =  smokers_df[smokers_df.bmi < 30]


# In[55]:


px.scatter(high_smoker,x='age',y='charges')


# In[56]:


px.scatter(low_smoker,x='age',y='charges')


# Clearly we seprated the curves of two linear relations and now we can work on them separately.

# In[63]:


inputs_smokerhigh = high_smoker[['age']]
targets_smokerhigh = high_smoker['charges']
inputs1,inputs_test_high,targets1,targets_test_high = train_test_split(inputs_smokerhigh,targets_smokerhigh,test_size = 0.1)


# In[64]:


model = LinearRegression()


# In[66]:


model.fit(inputs1,targets1)


# In[67]:


targets.describe()


# In[68]:


predictions1 = model.predict(inputs_test_high)


# In[69]:


predictions1


# In[70]:


rmse(predictions1,targets_test_high)


# In[71]:


model.coef_


# In[72]:


model.intercept_


# In[73]:


inputs_lowsmoker = low_smoker[['age']]
targets_lowsmoker = low_smoker['charges']
inputs2,inputs_test_low,targets2,targets_test_low = train_test_split(inputs_lowsmoker,targets_lowsmoker,test_size = 0.1)


# In[74]:


model.fit(inputs2,targets2)


# In[75]:


predictions2 = model.predict(inputs_test_low)


# In[76]:


predictions2


# In[77]:


rmse(predictions2,targets_test_low)


# In[78]:


model.coef_


# In[79]:


model.intercept_


# We can combine the above models to find how our model is performing for the smokers

# In[83]:


smokers_predictions = list(predictions1)+list(predictions2) #this is complete list of predictions for smokers but in increasing trend of bmi


# In[84]:


smokers_predictions = np.array(smokers_predictions)


# In[85]:


smokers_predictions


# In[88]:


targets = pd.Series(list(targets_test_high)+list(targets_test_low))


# In[89]:


rmse(targets,smokers_predictions)


# We created decent models for smokers and non smokers to calculate their annual medical bills.

# Now considering the entire dataset instead of making the distinction between the smokers and non smokers.

# In[90]:


inputs,targets = insurance[['age','bmi','children']],insurance['charges']
model = LinearRegression().fit(inputs,targets)
predictions = model.predict(inputs)
rmse(predictions,targets)


# The model is pretty bad; we analysed the reason pretty well earlier.

# # Using Categorical column in our machine learning model

# For using categorical columns in our machine learning model we would want to make it to a numeric data out of it first.
# Three ways of doing this is:-
# 1. if a categorical column has just two categories then we can replace their vallues with 0 and 1.
# 2. if a categorical colunn has more than two categories, we can preform one-hot encoding i.e. create a new column for each category with 1s and 0s.
# 3. if the categoreis have a natural order(eg. cold,neutral,warm,hot) then we can be converted to numbers directly based on their weights. These are called ordinals.

# In[91]:


sns.barplot(data = insurance,x = 'smoker',y = 'charges');


# The above graph shows avg charge for smoker and non smoker respectively.

# In[92]:


type(insurance)


# In[93]:


smokers_codes = {'no':0,'yes':1}


# In[94]:


insurance['smokers_codes'] = insurance.smoker.map(smokers_codes)


# In[95]:


insurance


# In[96]:


insurance.corr()


# smokers_codes correlation with charges is quite high.

# In[97]:


inputs,targets = insurance[['age','bmi','children','smokers_codes']],insurance['charges']
model = LinearRegression().fit(inputs,targets)
predictions = model.predict(inputs)
rmse(predictions,targets)


# 50% reduction of loss!!!

# In[98]:


sns.barplot(data = insurance,x='sex',y='charges');


# In[99]:


sex_codes = {'female':0,'male':1}


# In[100]:


insurance['sex_codes'] = insurance.sex.map(sex_codes)


# In[101]:


insurance.corr()


# In[102]:


inputs,targets = insurance[['age','bmi','children','smokers_codes','sex_codes']],insurance['charges']
model = LinearRegression().fit(inputs,targets)
predictions = model.predict(inputs)
rmse(predictions,targets)


# We will be using one hot encoding for the region columns as of course i can't think of any specific order of weightage to give to certain regions.

# In[103]:


sns.barplot(data = insurance,x='region',y='charges');


# In[104]:


from sklearn import preprocessing 
enc = preprocessing.OneHotEncoder()
enc.fit(insurance[['region']])
enc.categories_


# In[105]:


one_hot = enc.transform(insurance[['region']]).toarray()
one_hot


# In[106]:


insurance[['northeast','northwest','southeast','southwest']] = one_hot


# In[107]:


insurance


# In[108]:


insurance.corr()


# In[109]:


inputs,targets = insurance[['age','bmi','children','sex_codes','smokers_codes','northeast','northwest','southeast','southwest']],insurance['charges']
model = LinearRegression().fit(inputs,targets)
predictions = model.predict(inputs)
rmse(predictions,targets)


# Now let's consider the models for smoker and non smoker separately and compute the rmse losses.

# In[114]:


non_smoker_df = insurance[insurance.smoker == 'no']


# In[112]:


non_smoker_df.corr()


# In[115]:


smoker_df = insurance[insurance.smoker == 'yes']
high_smoker = smoker_df[smoker_df.bmi >= 30]
low_smoker = smoker_df[smoker_df.bmi < 30]


# In[116]:


high_smoker.corr()


# In[117]:


low_smoker.corr()


# In[120]:


inputs_nonsmoker = non_smoker_df[['age','sex_codes','bmi','children','northeast',
                      'northwest','southeast','southwest']]
inputs1 = high_smoker[['age','sex_codes','bmi','children','northeast',
                      'northwest','southeast','southwest']]
inputs2 = low_smoker[['age','sex_codes','bmi','children','northeast',
                      'northwest','southeast','southwest']]


# As I used Linear Regression to train my model thus its not necessary to perform feature scaling.

# In[121]:


from sklearn.model_selection import train_test_split


# In[124]:


inputs_train,inputs_test,targets_train,targets_test = train_test_split(inputs_nonsmoker,targets_nonsmoker,test_size = 0.1)
inputs_train1,inputs_test1,targets_train1,targets_test1 = train_test_split(inputs1,targets_smokerhigh,test_size = 0.1)
inputs_train2,inputs_test2,targets_train2,targets_test2 = train_test_split(inputs2,targets_lowsmoker,test_size = 0.1)


# In[132]:


model_nonsmokers = LinearRegression().fit(inputs_train,targets_train)
predictions = model.predict(inputs_test)

model_smokers_high = LinearRegression().fit(inputs_train1,targets_train1)
predictions1 = model.predict(inputs_test1)

model_smokers_low = LinearRegression().fit(inputs_train2,targets_train2)
predictions2 = model.predict(inputs_test2)


# In[130]:


from joblib import dump,load


# In[134]:


dump(model_nonsmokers,'medical_insurance_nonsmokers.joblib')
dump(model_smokers_high,'medical_insurance_smokers_high_bmi')
dump(model_smokers_low,'medical_insurance_smokers_low_bmi')


# In[ ]:





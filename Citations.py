#!/usr/bin/env python
# coding: utf-8

# # Predicting Speeding Tickets in Chattanooga - Tensorflow Models
# ___
# 

# <!-- ![CiteDensity.png](attachment:CiteDensity.png) -->
# 
# <div>
# <img src="attachment:CiteDensity.png" style="width:900px">
# </div>

# ___
# This tutorial explores the creation of a multi-layer perceptron neural network out of Tensorflow and Keras libraries, with additional result analysis provided by scikit-learn. Within this tutorial, the reader will begin to understand: 
# 
# * Import of datasets from CSV, and feather files. 
# * Addition of temporal specification variables
# * Addition of spatial markers to specific GPS coordinates
#     * GPS coordinates to Spatial Points
#     * Adjusting projection of Points. 
# * Creation of negative samples based off of a collection of variable combinations.
# * Creation of a variety of neural networks with combinations of input metrics. 
# * Analysis of the performance of the aforementioned networks based on overall Accuracy and Recall. 
# ____

# ### Introduction to Tensorflow
# 
# For those who are not already familiar with the Tensorflow Python library, here is the definition of Tensorflow, from their website itself. 
# 
# >"TensorFlow is an end-to-end open source platform for machine learning. 
# It has a comprehensive, flexible ecosystem of tools, libraries and community resources 
# that lets researchers push the state-of-the-art in ML and developers easily build and deploy 
# ML powered applications."
# >-[Tensorflow.org](https://www.tensorflow.org/)
# 
# Tensorflow is an amazing backend for many common machine learning problems, and provides easy model building regardless of the coder's experience level. The Tensorflow webpage features many example dataset and code pairings to get one acclimated to the coding involved.
# 
# Furthermore, in this tutorial we will be exploring the Sequential model from Keras:  
# 
# >A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor. 
# >-[Keras.io](https://keras.io/guides/sequential_model/)
# 
# Sequential models are wonderful for linear questions, such as 'Will it rain today?', where each input is fed through the model without doubling-back. They are not meant for situations with multiple input/multiple output, or if the layers are meant to be shared. 

# ### Use Case Question
# 
# Where in Chattanooga are speeding tickets being issued? 
# Is it possible to predict which roadways are likely to see speeding violations based off of historical reports? 
# 
# ### Introduction to Data 
# 
# Both sets data utilized in this walkthrough can be accessed and are free for public usage. 
# 
# The citation data set referenced is available from the ChattaData.org page here: [City Court Citations](https://internal.chattadata.org/Public-Safety/City-Court-Citations/th6b-88wc). The citation data includes spatial and temporal data about the location of citations issued from city courts, as well as roughly anonymous data regarding the individual receiving the citation. 
# 
# Roadway data for the area is also available via ChattaData, here: [Chattanooga Roadways](https://internal.chattadata.org/dataset/Chattanooga-Roadways/mw3f-d2mz). Roadway data includes spatial data regarding to individual segments of roadways within the Chattanooga area. Rough address data is provided, and allows for the creation of a singular 'Segment' column.  
# 
# <!-- ![Roadways.png](attachment:Roadways.png) -->
# 
# <div>
# <img src="attachment:Roadways.png" width="550"/>
# </div>
# 

# ____
# ## PreProcessing
# 
# ### Importing and Exploring Data
# 
# The first step to understanding any selection of data is taking a look into what data you actually have. Here, we're importing a dataset that includes rough anonymized data regarding Court Citations. First, let's import the dataset CSV using pandas, and take a look at how many entries we have. 

# In[41]:


import pandas as pd
citations = pd.read_csv("../Data/City_Court_Citations.csv")
print(len(citations.values))


# Next, let's take a look at the types of columns we have. Our data includes temporal and spatial data regarding the citations recorded. 
# 
# Now, let's take a look at the first ten records. 
# 
# This combination of commands lets us take a closer look into what data we have. Notice that there are multiple 'object' columns, which includes the type and date of the violation, as well as information regarding the individual receiving the citation.

# In[42]:


print(citations.dtypes)
citations[0:10]


# Let's cut this data down to just the records where the violation was for speeding, to give us a better answer to our main questions we asked above. We'll also reindex the dataset for ease of understanding. 

# In[43]:


citations = citations[citations['Offense Description'] == 'SPEEDING']
citations = citations.reset_index()


# Now, we'll be taking a look at our reduced data, now that we've cut it down to just the records that pertain to our question. We can see that there were 20,815 speeding tickets within that larger dataset. 
# 
# Again, we'll use the head command to take a look at the first ten records of speeding.  

# In[44]:


print(len(citations))
citations.head()


# We'll need to adjust how our time and date are displayed before splitting the data up, just for simplicity's sake for later usage. Here, we're using a lambda statement to avoid utilizing for loops. For loops are great for assigning variables but they can get slowed down if the dataset becomes too large. While that's not a problem with this smaller dataset, it's good to familarize yourself with time and computation saving code whenever possible.  

# In[45]:


import datetime

citations['Violation Date'] = pd.to_datetime(citations['Violation Date'])
citations.dtypes

citations['Time'], citations['Date']= citations['Violation Date'].apply(lambda x:x.time()), citations['Violation Date'].apply(lambda x:x.date())


# With the time and date split, let's now get some extra variables to help our model understand those values. Neural network models can't parse string variables, so we'll need to pull out the month, day of the week, and hour in order for the model to understand.
# 
# The weekday function finds the day of the week of a given date, and assigns it a value between 0 and 6, where Monday is zero, and Sunday is six. 
# 
# Since the dataset already included a year column, we don't have to find that manually. We're using lambdas here as well. 
# 
# Finally, we'll take a look at the head of the dataset again to make sure all of our commands worked correctly. 

# In[46]:


citations['Month'] = citations['Date'].apply(lambda x: x.month)
citations['WeekDay'] = citations['Date'].apply(lambda x: x.weekday())
citations['Hour'] = citations['Time'].apply(lambda x: x.hour)
citations = citations.rename(columns={"Citation Year": "Year"})
citations.head()


# <!-- This next section is messy, but there's really no way around it. We need to reduce all of the addresses in the dataset to only the roadways. So, our first step is to remove any leading numbers, which indicate the street address. We're also going to remove indicators for Southbound, Northbound, Eastbound, and Westbound traffic. If these were present with every record, we could use them, but they are far too inconsistent in the data, so we'll be taking them out. Further complicating things is that the word 'Highway' is entered in the data in multiple ways. So, we'll be correcting those as well.  -->

# ### Combining Roadway Data with Citation reports. 
# The next section combines our roadway information with the citation information to give each entry a set roadway name.
# 
# First, we'll be creating the geometry column for each of the GPS coordinates in the dataset. 
# 
# Then, we'll be setting a coordinate reference system, so that the roadway data and the citation records can be properly matched. 
# 
# Finally, we'll take a look at the head of the data again. Notice that the Location WKT column is almost exactly the same as the newly created geometry column. While we could have simply assigned that column as the 'geometry', it is simply a measure of caution to create the geometry column ourselves. 

# In[47]:


import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt

citations["geometry"] = citations["Location WKT"].astype(str).apply(wkt.loads)
citations = gpd.GeoDataFrame(citations, geometry="geometry")
citations.crs = "epsg:4326"
citations.head()


# Now, we'll be importing our roadway data, double-checking the type of our roadways, and limiting the entries to only that type. Note that sometimes datasets will have corrupted or incomplete data. This was the case with this dataset, where one entry of the roadways was incomplete. We can see that in the totals printed before and after the selection line. 
# 
# Next, we join the two datasets together with a geopandas command called sjoin. Sjoin determines spatial matching between datasets of any type (Point, Polygon, Multi-line, etc) and one can select how they would like the new merged set to be set up by selecting 'left', 'right', or 'inner' for the how variable. We are looking to select citations within the roadway data, so we will select 'left', since our citations set is the left variable. 
# 
# Something interesting happens when we merge our data, though. notice that the number of records actually increases. This is related to the records that do not fall within a given roadway polygon. The true number of citations retained is shown in our last print line, where we can see the number falling to 18,576. 
# 
# Once again, we'll print the head of the data to verify our changes. 

# In[48]:


roads = gpd.read_file("../Data/Roads/Roads.shp")
print(roads.geometry.type)

print("Roads:", len(roads.values))
roads = roads[roads.geometry.type == 'Polygon']
print("Roads:",len(roads.values))

print("Citation records before merge:",len(citations.values))
citations = gpd.sjoin(citations, roads, how="left", op="within")
print("Records after merge", len(citations.values))

print("Citations before dropping duplicates:",len(citations.values))
citations = citations.dropna(subset=['Road'])
citations = citations.drop_duplicates(subset ='index')
print("Citations after dropping duplicates:",len(citations.values))
citations.head()


# Now, let's take the set of roadway strings we just created and convert them into category variables. This takes all the roadways in question and creates a unique numeric tag for each one. This way our roadways can be input into the model, while retaining their unique identities. 

# In[49]:


citations['Road_Num'] = citations['Road'].astype("category").cat.codes
citations['Road_Num']


# ### Parring Down Variables and Preparing for Negative Sampling
# Now we can identify which columns need to be removed for our model. Anything that is a string or factor cannot be parsed, so it has to go. Also, there are columns that aren't really necessary for the situation, like information about the person who received the ticket. This list of variables to drop includes: 
# - Citation Number
# - Offense Description
# - Offense Code
# - Race of Offender
# - Sex of Offender
# - Violation Date
# - Address 
# - Location WKT
# - Citation_Charge_Link
# - Agency 
# - Date
# - Index from roadways
# 
# Now, before we remove any of these variables, we'll be making a copy to retain all of our information. Also, we'll be making a smaller dataframe with just our roadways and numeric tags, so we can understand the output of our model. 

# In[50]:


processedData = citations.copy()
ids = (citations[["Road", "Road_Num"]].drop_duplicates()).sort_values(by=['Road_Num'])
ids[0:10]
ids.to_csv("../Data/RoadsIDs.csv")


# In[51]:


citations.columns
citations = citations.reset_index()
citations = citations[['Year','Month','WeekDay','Hour','Segment','Road_Num']]
citations.head()
citations.to_csv("../Data/ProcessedData.csv", index=False)


# ### Negative Sampling
# Negative sampling is the creation of negative records from the provided positive records, through the alteration of some if not all of the available variables. Then, the negative samples are compared to the original positive records to remove any possible false negatives. 
# We can roughly determine how many combinations of the remaining variables are possible, minus the records we
# have within the dataset. As shown below, there is an immense amount of negatives possible when considering all combinations of the variables available. However, not all of these options are available, as each roadway has a differing number of segments.

# In[52]:


unique_possible = (len(pd.unique(citations.Road_Num.values)) * len(pd.unique(citations.Segment.values)) * len(pd.unique(citations.WeekDay.values)) * 
len(pd.unique(citations.Year.values)) * len(pd.unique(citations.Hour.values)) * 
len(pd.unique(citations.Month.values))) - len(citations.values)

print(unique_possible)


# Since the creation of negative records is not the focus of this post, we'll just highlight a quick example using a very small slice of our overall dataset. We're slicing a copy of the first ten entries in the citations dataset, and expanding on a Itertools method called product in order to mimic the behavior of an R method called expand. Basically, expand takes a dataset and 'expands' the current entries into every possible combination of said entries. Notice that the unique entries in the created negative set test all match to those found in that sliced piece from citations, but the combinations of those variables created almost a million entries, once the existing citations entries were removed. 

# In[53]:


import itertools

def expandgrid(*items):
   product = list(itertools.product(*items))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(items))}

sliced = pd.DataFrame(citations.values[0:10], columns = citations.columns)
sliced.columns
print(sliced)


foo = pd.DataFrame(expandgrid(sliced.WeekDay, sliced.Month, sliced.Year, 
                               sliced.Road_Num, sliced.Segment, sliced.Hour))
print(len(foo))

foo.head()
foo.columns = ['WeekDay','Month','Year','Road_Num','Segment','Hour']
foo = foo[sliced.columns.values]
foo = foo[~foo.isin(citations)].dropna()
print(len(foo))

for i in foo.columns: 
    print(i, " : ", foo[i].unique())
foo.head()


# Now we can import our actual negative data (created outside of this example, for simplicity's sake). 
# 
# In order to produce a rough ratio for negatives to positives, we'll find how many times the number of citations fits within the negatives, and then take a given iterative of that number. In this case, The length of the negatives is roughly 142 times the length of the citations, so we'll be saving every 142nd entry in the negatives in order to create a roughly 1:1 ratio. 
# 
# The imported negatives have the 'Citation' variable set to zero, since no speeding citation was issued for that temporal/spatial instance. So, we have to add a 'Citation' variable to the actual citation data, and then we are able to merge the sets. All in total, the dataset, once merged, has a length of 37,272 entries with a roughly 1:1 ratio for citations to negatives. 

# In[80]:


import feather
negatives = feather.read_dataframe("../Data/Negatives/Negatives.feather")
print(len(negatives))
cut = int(len(negatives) / len(citations))
print(cut)

negatives = negatives.iloc[::cut, :]
print(len(negatives))
citations['Citation'] = 1
print(len(citations))
data = pd.concat([negatives, citations])
data = data.sort_values(by = ['Year', 'Month'])

print(len(data))


# ___
# ## Neural Network Creation 
# 
# ### Splitting Data and preparing for the neural network
# 
# With our data now in a format that our model can understand, let's create the training and testing sets. First, after all of our imports are done, we shuffle the data. This allows the model to properly understand the data, and allows for the best chance of each epoch receiving a different 'chunk' of the data. Next, we'll be dividing the sets by year, with this year's (2020) citations being our testing set. Finally, let's also take a look at how many records are in each. We can see that there are 36,029 entries in our training, and 1,243 in our testing. 

# In[81]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from keras import callbacks
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing

# Import matplotlib pyplot safely
try:
    import matplotlib.pyplot as plt
except ImportError:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

data = shuffle(data)
training = data[data['Year'] < 2020]
testing = data[data['Year'] == 2020]

print(len(training.values))
print(len(testing.values))


# ### Standardizing Data with MinMaxScaler
# The function below takes in a dataset, removing the 'Y' column and transforming the data using a minimum/maximum scaler. What this means is that for each column in the data, the maximum and minimum values are found, and are reassigned as 1 and 0 respectively. Then, each value in the column is scaled to fall between those two limits. This simplifies the process for the neural network, allowing for faster learning.

# In[82]:


def standardize(data, Y):
    columns = data.columns.values[~data.columns.isin([Y])]
    Y = data[[Y]]
    print(columns)
    scaler = preprocessing.MinMaxScaler()
    # Fit your data on the scaler object
    dataScaled = scaler.fit_transform(data[columns])
    dataScaled = pd.DataFrame(dataScaled, columns=columns)
    return dataScaled, Y


# The following code snippet takes the training and testing sets we just created, and standardizes them, splitting into X and Y for both training and testing. Then, we take a look at the resulting datasets. 
# 
# Note that in this particular dataset, we don't apply the scaler to the Y column, since it is already on a minmax scale. 

# In[83]:


X_Train, Y_Train = standardize(training, 'Citation')
X_Test, Y_Test = standardize(testing, 'Citation')

print(Y_Train.head())
print(X_Train.head())

print(Y_Test.head())
print(X_Test.head())


# ### Creating a Neural Network Model
# 
# Now, we can actually create our neural network. For this example, we'll be utilizing a Sequential model, with an input layer, and output layer, and three layers in between, gradually diminishing the node counts. We compile the model with a variety of optimizers and loss functions, to test which of the list is the best combination. 

# In[97]:


def evaluating_model(X_Train, Y_Train, model, Y_Test, hist):
    
    # This is evaluating the model, and printing the results of the epochs.
    scores = model.evaluate(X_Train, Y_Train, batch_size=10)
    print(scores)
    print("\nModel Training Accuracy:", scores[2] * 100)

    print("Model Training Loss:", sum(hist.history['loss']) / len(hist.history['loss']))

    # Okay, now let's calculate predictions probability.
    predictions = model.predict(X_Test)  

    # Then, let's round to either 0 or 1, since we have only two options.
    predictions_round = [abs(round(x[0])) for x in predictions]

    # Finding accuracy score of the predictions versus the actual Y.
    accscore1 = accuracy_score(Y_Test, predictions_round)
    
    # Printing it as a whole number instead of a percent of 1. (Just easier to read)
    print("Rounded Test Accuracy:", accscore1 * 100)
    
    # Find the Testing loss as well:
    print("Test Loss", sum(hist.history['val_loss']) / len(hist.history['val_loss']))

    ##Finding the AUC for the cycle: 
    fpr, tpr, _ = roc_curve(Y_Test, predictions_round)
    roc_auc = auc(fpr, tpr)
    print('AUC: %f' % roc_auc)

    tn, fp, fn, tp = confusion_matrix(Y_Test, predictions_round).ravel()
    print(tn, fp, fn, tp)
    return tn, fp, fn, tp, roc_auc, fpr, tpr

def main():
    # Setting our metric options
    opts = ['rmsprop','adam','nadam']
    losses = ["binary_crossentropy", "poisson", "mse"]
    mets = ['Recall', 'accuracy', 'AUC']

    results = pd.DataFrame(columns = ['loss', 'recall', 'accuracy', 'auc', 'val_loss', 'val_recall', 
                                      'val_accuracy', 'val_auc', 'Optimizer','Loss'])
    # Cycling through the optimizers, then loss functions to find the best fitting combinations. 
    for o in opts: 
        for l in losses: 
            print("Metrics:", o,l)
            
            # 4. Creation of model structure. 
            model = tf.keras.Sequential()
            # Input layer
            model.add(Dense(len(X_Train.columns), input_dim=len(X_Train.columns), activation='sigmoid'))
            
            # Hidden Layers
            model.add(Dense(int(len(X_Train.columns)/2), activation='sigmoid'))
            model.add(Dense(int(len(X_Train.columns)/3), activation='sigmoid'))
            model.add(Dense(int(len(X_Train.columns)/4), activation='sigmoid'))
            
            # Output layer
            model.add(Dense(1, activation='sigmoid'))
            
            # Compiling each unique model.
            model.compile(loss=l, optimizer=o, metrics=mets)
            
            # Providing a subset of the testing set to validate training. 
            X_valid = X_Test[0:100]
            Y_valid = Y_Test[0:100]
            
            # Fitting the unique models, and creating the model history. 
            hist = model.fit(X_Train, Y_Train, epochs=10, batch_size=25, validation_data=(X_valid, Y_valid), verbose=1)
            
            # Creation of the 'test' dataframe allows for easy saving of each combo of optimizer and loss functions. 
            test = pd.DataFrame.from_dict(hist.history, orient='columns')
            test['Optimizer'] = o
            test['Loss'] = l
            
            tn, fp, fn, tp, roc_auc, fpr, tpr = evaluating_model(X_Train, Y_Train, model, Y_Test, hist)
            model.save_weights("../Models/"+o+"_"+l+".h5")
            print("Saved grid model to disk")
            tf.keras.backend.clear_session()
            
            # Saving results from evaluating_model function, and concating the unique model performance to the whole. 
            test['TN'] = tn
            test['FP'] = fp
            test['TP'] = tp
            test['FN'] = fn
            test['test_recall'] = tp / (tp+fn)
            test['test_accuracy'] = (tp + tn) / (tp+fn+tn+fp)
            test['FPR'] = str(fpr)
            test['TPR'] = str(tpr)
            test['ROC_AUC'] = roc_auc
            results = result = pd.concat([results, test])
    # Saving the completed results of all model combinations.         
    results.to_csv("../Data/TrainingResults.csv", index=False)
    
if __name__ == "__main__":
    main()


# ____
# ## Results Analysis
# Now that the models have completed running, we can analyze their results graphically. Each of the metric combinations is represented by a different color and line type combo, with the colors and linetypes remaining consistent through the following visualization. 

# ### Results - Accuracy Performance
# 
# Here, we can discern that the Nadam optimizer sets begin much lower in accuracy, but catch up to their fellows within one epoch run. The first metric set to begin increase toward the final score is the Adam/MSE pair, with the accuracy reaching 71.89 on the fifth epoch. By the sixth epoch, this metric set had reached its point of convergence. However, this set (72.36) was marginally overtaken by both Nadam/MSE (72.364) and Nadam/Binary Crossentropy (72.397) by the tenth epoch. 

# <!-- ![Acc.png](attachment:Acc.png) -->
# 
# <div>
# <img src="attachment:Acc.png" width="700"/>
# </div>

# Validation Accuracy proved to be a more chaotic affair, with all sets roughly beginning at 76 percent accuracy. The maximum validation accuracy was provided by the combination of RMSProp and Poisson, finishing at 77 percent where the second position fell to Adam/Poisson. 

# <!-- ![Valid_Acc.png](attachment:Valid_Acc.png) -->
# 
# <div>
# <img src="attachment:Valid_Acc.png" width="700"/>
# </div>
# 

# Once testing of each combination was completed, the RMSProp/Poisson set was found to rank highest in Testing Accuracy, with a score of 78.44 percent. The next was Adam/Poisson, with 78.042. 

# <!-- ![Test_Acc.png](attachment:Test_Acc.png) -->
# 
# <div>
# <img src="attachment:Test_Acc.png" width="700"/>
# </div>

# ### Results - Recall Performance
# 
# Recall is the measure of how many records were correctly predicted positive, or the value found by 
# 
# $\frac{True Positive}{(True Positive + False Negative)}$
# 
# Recall of many of the metric sets begins at zero, with all predictions 'No Citation'. However, Nadam/MSE, RMSProp/MSE, and Nadam/Poisson all begin at higher values before falling to zero. While many of the sets began predicting citations on Epoch 4, their performance was most certainly topped by the Adam/MSE set. Note that this set continued to outperform the rest until Epoch 7, where many sets caught up in performance. Epoch 7 is also when the Nadam/Binary Crossentropy set began outperforming, with a Recall of 40.87 percent. This set finished training at 41.67 percent Recall. 

# <!-- ![Recall.png](attachment:Recall.png) -->
# 
# <div>
# <img src="attachment:Recall.png" width="700"/>
# </div>

# Most of the sets performed similarly on validation, with only variety in when higher Recall values were achieved. again, Nadam/MSE was the first to approach max Recall, upon Epoch 4, and RMSProp/MSE was the last with Epochs 6/7. All sets finished validation at 58.33 percent Recall. 

# <!-- ![Valid_Recall.png](attachment:Valid_Recall.png) -->
# 
# <div>
# <img src="attachment:Valid_Recall.png" width="700"/>
# </div>

# Recall upon testing of the various sets presented the Adam/Binary Crossentropy, RMSProp/Binary crossentropy, and RMSProp/MSE sets all performing well, all with Recalls of 64.84. The least performing was RMSProp/Poisson, at 62.94. 

# 
# <!-- ![Test_Recall.png](attachment:Test_Recall.png) -->
# 
# <div>
# <img src="attachment:Test_Recall.png" width="700"/>
# </div>

# ### Creating a prediction set
# 
# Now, we'll be importing a dataset of all citations from the month of April in 2020, and will be suppling them to the model for predictions. Of the 12 citations issued, 8 were correctly identified by the model. The Recall in this case would be roughly 66.7%, somewhat higher than the previously testing recall. Also of note is that even records not correctly identified were above 42.41% probability to see a citation. 

# In[202]:


data = pd.read_csv("../Data/AprilCite.csv")
data = shuffle(data)
cutdata, Y = standardize(data,'Citation')

# Load in model weights from training session. 
model.load_weights("../Models/rmsprop_mse.h5")
probability = model.predict(cutdata)

# Save the predicted values to the Probability column.
data["Probability"] = probability

# Then, let's round to either 0 or 1, since we have only two options (accident or no).
predictions_round = [abs(round(x[0])) for x in probability]
data["Prediction"] = predictions_round

# Printing some of the found values, as well as the total number of predicted accidents for this data.
print("\tMin probability of citation: ", round(float(min(probability) * 100), 2))
print("\tMax probability of citation: ", round(float(max(probability) * 100), 2))
print("\tCorrectly predicted: ", sum(data.Prediction == data.Citation))

data.to_csv("../Data/AprilPredicted.csv")


# However, testing the performance of a model upon only positive data can skew result reporting. Therefore, we must test the model with a given fixed temporal set. In this case, negative samples were specifically selected for Thursdays in April 2020, at 4PM. This means four of the column values within the negatives are set, with only roadway data fluctuating. Then, the citations for that specific timeframe are added and same process is followed as above, passing the data into the predict function and finding the probability of an accident at each of the roadway segments. 
# 
# Given that the issuing of citations depends strongly on many other factors not being considered in this brief example, it is understandable that the model created only accurately identified 28 records. Of course, this is considering a loose defined fifty/fifty probability. In many real-world use cases, a fifty/fifty probability would not necessarily be considered in favor of 'higher certainty' scores like 75 or even 90 % probability. 

# In[213]:


data = pd.read_csv("../Data/AprilTest.csv")
print("Number of records:", len(data))
data = shuffle(data)
cutdata, Y = standardize(data,'Citation')

# Load in model weights from training session. 
model.load_weights("../Models/rmsprop_mse.h5")
probability = model.predict(cutdata)

# Save the predicted values to the Probability column.
data["Probability"] = probability

# Then, let's round to either 0 or 1, since we have only two options (accident or no).
predictions_round = [abs(round(x[0])) for x in probability]
data["Prediction"] = predictions_round

# Printing some of the found values, as well as the total number of predicted accidents for this data.
print("\tMin probability of citation: ", round(float(min(probability) * 100), 2))
print("\tMax probability of citation: ", round(float(max(probability) * 100), 2))
print("\tCorrectly predicted: ", sum(data.Prediction == data.Citation))
print("\tCitations predicted: ", sum(data.Prediction))
print("\tCitations existed: ", sum(data.Citation))

data.to_csv("../Data/AprilPredictedBoth.csv")


# ### Analyzing Results Spatially 
# 
# Finally, we can take a look at the specific areas that the model was most certain a citation would occur. The below graphic presents the fifty roadway segments that the model was most certain that a citation would occur. Note that the window of probability presented here is quite small. 
# 
# Also of interest is the number of shorter length segments that earned the highest probability. Only two segments of a considerable length were included in the top fifty, with a handful of other segments having longer than a few blocks length. 
# 
# Finally, of the entire Chattanooga area the fifty segments with the highest probability of a citation occuring are all located in a small area of the city. Particuarly, the area between 85.325&deg;W and 85.3&deg;W. 

# <!-- ![FiftyHighestCitationProbability.png](attachment:FiftyHighestCitationProbability.png) -->
# 
# 
# <div>
# <img src="attachment:FiftyHighestCitationProbability.png" width="900"/>
# </div>

# ____
# ## Summary
# 
# In this example, we've walked through the preprocessing of data, creation of spatial and temporal variables based off of existing variables, and concluded with the creation of multiple Multilayer Perceptron models, made to predict which sections of local roadways would be likely to see speeding citations in 2020 with a variety of metrics. We've also discussed the results of those models, examining which performed best in both Accuracy and Recall, and put the model to the test with a smaller dataset representing April 2020. 
# 
# We hope you've enjoyed this tutorial. Please join us for more machine learning walkthroughs in the future. 

# If you'd like further help in topics such as:
# 
# - General machine learning education
# - Advanced deep learning modeling
# - Enterprise machine learning infrastructure
# 
# Please [reach out to us and say hello](http://www.pattersonconsultingtn.com/contact.html)

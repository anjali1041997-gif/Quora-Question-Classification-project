#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[37]:


data = pd.read_csv('train.csv')


# In[38]:


data.head()


# In[39]:


data.info()


# In[40]:


# Cutting down data as data is too large to process
rows_to_delete = 1256122
# Randomly select rows to delete
rows_indices_to_delete = data.sample(n=rows_to_delete).index

# Drop the selected rows from the DataFrame
data = data.drop(rows_indices_to_delete)

# Reset the index after removing the rows
data = data.reset_index(drop=True)


# In[41]:


data.describe()


# In[42]:


def tokenize_text(text):
    return word_tokenize(text)


# In[43]:


def preprocess_token(token):
    token = token.lower()
    token = re.sub(r'[^a-zA-Z0-9]', '', token)
    return token


# In[44]:


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]


# In[45]:


def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


# In[46]:


def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


# In[47]:


def preprocess_data(data):
    data['question_text'] = data['question_text'].apply(lambda x: x.lower())  # Convert to lowercase
    data['question_text'] = data['question_text'].apply(tokenize_text)  # Tokenization
    data['question_text'] = data['question_text'].apply(lambda tokens: [preprocess_token(token) for token in tokens])  # Token Lowercasing and Removal of Non-Alphanumeric Characters
    data['question_text'] = data['question_text'].apply(remove_stopwords)  # Removal of Stopwords
    data['question_text'] = data['question_text'].apply(stem_tokens)  # Token Stemming
    data['question_text'] = data['question_text'].apply(lemmatize_tokens)  # Token Lemmatization
    data['question_text'] = data['question_text'].apply(lambda x: ' '.join(x))  # Convert back to text


# In[48]:


preprocess_data(data)


# ### Data visualisation

# In[19]:


import matplotlib.pyplot as plt


# In[41]:


sincere_count=(data["target"]==0).sum()
print(sincere_count)


# In[42]:


insincere_count=(data["target"]==1).sum()
print(insincere_count)


# In[43]:


# Create a list of the counts and labels
counts = [sincere_count, insincere_count]
labels = ['Sincere', 'Insincere']
colors = ['orange', 'blue']  # You can customize colors if needed

# Create a donut pie chart
plt.pie(counts, labels=labels, colors=colors, autopct='%1.8f%%', startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'))

# Draw a white circle in the middle to create the donut chart
centre_circle = plt.Circle((0, 0), 0.6, fc='white')
plt.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that the pie chart is drawn as a circle
plt.axis('equal')

# Set chart title
plt.title('Distribution of Sincere and Insincere Quora Questions')

# Show the chart
plt.show()


# In[44]:


# Calculate the word count for each question text
data['word_count'] = data['question_text'].apply(lambda x: len(x.split()))

# Calculate the frequency of word counts
word_count_frequency = data['word_count'].value_counts().reset_index()
word_count_frequency.columns = ['word_count', 'count']

# Sort by word count for better visualization
word_count_frequency = word_count_frequency.sort_values(by='word_count')

# Plot the bar graph with increasing bars towards the right
plt.figure(figsize=(6,3))
plt.bar(word_count_frequency['word_count'], word_count_frequency['count'], color='blue')
plt.xlabel('Number of Words in Question Text')
plt.ylabel('Number of Question Texts')
plt.title('Word Count Distribution of Clean Text')
plt.xticks(word_count_frequency['word_count'], rotation=90)
plt.show()


# # Bivariate Analysis

# ## Probability density function: Clean Text Distribution

# In[45]:


import seaborn as sns
# Separate sincere and insincere question texts
sincere_texts = data[data['target'] == 0]['question_text']
insincere_texts = data[data['target'] == 1]['question_text']

# Calculate the word count for each class
sincere_word_counts = sincere_texts.apply(lambda x: len(x.split()))
insincere_word_counts = insincere_texts.apply(lambda x: len(x.split()))

# Plot the probability density function (PDF) for both classes
plt.figure(figsize=(6,3))
sns.kdeplot(sincere_word_counts, color='skyblue', label='Sincere', bw_method=0.5)
sns.kdeplot(insincere_word_counts, color='orange', label='Insincere', bw_method=0.5)
plt.xlabel('Number of Words in Question Text')
plt.ylabel('Probability Density')
plt.title('Probability Density Function of Word Count Distribution')
plt.legend()
plt.show()


# ## Cumulative Density Function

# In[45]:


# Plot the cumulative density function (CDF) for both classes
plt.figure(figsize=(6, 2))
sns.kdeplot(sincere_word_counts, color='skyblue', label='Sincere', cumulative=True, fill=True)
sns.kdeplot(insincere_word_counts, color='orange', label='Insincere', cumulative=True, fill=True)
plt.xlabel('Number of Words in Question Text')
plt.ylabel('Cumulative Density')
plt.title('Cumulative Density Function of Word Count Distribution')
plt.legend()
x_ticks = list(range(0, max(max(sincere_word_counts), max(insincere_word_counts)) + 1, 10))
plt.xticks(x_ticks)

# Add grid lines
plt.grid(True)

plt.show()


# ## Boxplot Distribution

# In[49]:


# Plot the boxplot distribution for both classes side by side
plt.figure(figsize=(5,3))
sns.boxplot(x='target', y='word_count', data=data, palette='pastel')
plt.xticks([0, 1], ['Sincere', 'Insincere'])
plt.xlabel('Question Type')
plt.ylabel('Number of Words in Question Text')
plt.title('Boxplot Distribution of Word Count for Sincere and Insincere Questions')
plt.show()


# ## t-Distributed Stochastic Neighbour Embedding

# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# Combine sincere and insincere question texts
all_texts = data['question_text']

# Sample a smaller subset of the data for faster execution
sample_size = 10000
sample_texts = all_texts.sample(sample_size, random_state=42)

# Create TF-IDF representation using TfidfVectorizer
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(sample_texts)

# Perform t-SNE on TF-IDF representation (using Barnes-Hut approximation)
tsne_tfidf = TSNE(n_components=2, random_state=42, method='barnes_hut')
X_tsne_tfidf = tsne_tfidf.fit_transform(X_tfidf.toarray())

# Separate t-SNE results for sincere and insincere questions
sincere_tsne_tfidf = X_tsne_tfidf[data['target'].sample(sample_size, random_state=42) == 0]
insincere_tsne_tfidf = X_tsne_tfidf[data['target'].sample(sample_size, random_state=42) == 1]

# Plot t-SNE results
plt.scatter(sincere_tsne_tfidf[:, 0], sincere_tsne_tfidf[:, 1], label='Sincere', alpha=0.5)
plt.scatter(insincere_tsne_tfidf[:, 0], insincere_tsne_tfidf[:, 1], label='Insincere', alpha=0.5)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization of TF-IDF Data')
plt.legend()
plt.show()


# In[26]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
data = pd.read_csv('train.csv')

# Combine sincere and insincere question texts
all_texts = data['question_text']

# Sample a smaller subset of the data for faster execution
sample_size = 10000
sample_texts = all_texts.sample(sample_size, random_state=42)

# Create BoW representation using CountVectorizer
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(sample_texts)

# Perform t-SNE on BoW representation (using Barnes-Hut approximation)
tsne_bow = TSNE(n_components=2, random_state=42, method='barnes_hut')
X_tsne_bow = tsne_bow.fit_transform(X_bow.toarray())

# Separate t-SNE results for sincere and insincere questions
sincere_tsne_bow = X_tsne_bow[data['target'].sample(sample_size, random_state=42) == 0]
insincere_tsne_bow = X_tsne_bow[data['target'].sample(sample_size, random_state=42) == 1]

# Plot t-SNE results
plt.scatter(sincere_tsne_bow[:, 0], sincere_tsne_bow[:, 1], label='Sincere', alpha=0.5)
plt.scatter(insincere_tsne_bow[:, 0], insincere_tsne_bow[:, 1], label='Insincere', alpha=0.5)
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE Visualization of BoW Data')
plt.legend()
plt.show()


# ## Building models over data

# # Logestic regression over TFIDF- Stemmed Text

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet


# In[6]:


# Load your dataset (replace with your data loading code)
data = pd.read_csv('train.csv')


# In[7]:


# Preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lowercasing and removing non-alphanumeric characters
    tokens = [re.sub(r'[^a-zA-Z0-9]', '', token.lower()) for token in tokens]
    
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, wordnet.VERB) for token in tokens]
    
    return ' '.join(tokens)


# In[8]:


# Cutting down data as data is too large to process
rows_to_delete = 1256122
# Randomly select rows to delete
rows_indices_to_delete = data.sample(n=rows_to_delete).index

# Drop the selected rows from the DataFrame
data = data.drop(rows_indices_to_delete)

# Reset the index after removing the rows
data = data.reset_index(drop=True)


# In[9]:


# Apply preprocessing to the text data
data['question_text'] = data['question_text'].apply(preprocess_text)


# In[11]:


# Downsampling the majority class to balance the dataset
sincere_data = data[data['target'] == 0]
insincere_data = data[data['target'] == 1].sample(n=len(sincere_data), replace=True, random_state=42)
balanced_data = pd.concat([sincere_data, insincere_data])


# In[12]:


# Split the balanced data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(balanced_data['question_text'], balanced_data['target'], test_size=0.2, random_state=42)


# In[13]:


# Create TF-IDF representation using TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# In[44]:


# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1,0.5, 1,5]}  # List of values to search
lr_model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_tfidf, y_train)

# Get the best hyperparameters
best_c = grid_search.best_params_['C']


# In[45]:


# Create the logistic regression model with the best hyperparameter
lr_model_tfidf = LogisticRegression(C=best_c, max_iter=1000)


# In[46]:


# Train the model
lr_model_tfidf.fit(X_train_tfidf, y_train)


# In[47]:


# Make predictions on train and test data
y_train_pred = lr_model_tfidf.predict_proba(X_train_tfidf)[:, 1]
y_test_pred = lr_model_tfidf.predict_proba(X_test_tfidf)[:, 1]


# In[48]:


#Calculate AUC for train and test data
train_auc = roc_auc_score(y_train, y_train_pred)
test_auc = roc_auc_score(y_test, y_test_pred)


# ### Model Evaluation

# In[49]:


# Plot the ROC curve
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)

plt.figure(figsize=(6,2))
plt.plot(fpr_train, tpr_train, label=f'Train AUC = {train_auc:.2f}')
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {test_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Print the best hyperparameter and its corresponding ROC AUC score
print("Best C:", best_c)
print("Best ROC AUC:", grid_search.best_score_)


# ### Confusion Matrix and Classification Report

# In[50]:


from sklearn.metrics import confusion_matrix, classification_report

# Set a threshold for predicting class labels from probabilities
threshold = 0.5
y_train_pred_binary = (y_train_pred >= threshold).astype(int)
y_test_pred_binary = (y_test_pred >= threshold).astype(int)

# Calculate the confusion matrix for train and test data
train_confusion_matrix = confusion_matrix(y_train, y_train_pred_binary)
test_confusion_matrix = confusion_matrix(y_test, y_test_pred_binary)

# Generate the classification report for train and test data
train_classification_report = classification_report(y_train, y_train_pred_binary)
test_classification_report = classification_report(y_test, y_test_pred_binary)

# Print the confusion matrix and classification report
print("Train Confusion Matrix:")
print(train_confusion_matrix)
print("\nTest Confusion Matrix:")
print(test_confusion_matrix)

print("\nTrain Classification Report:")
print(train_classification_report)
print("\nTest Classification Report:")
print(test_classification_report)


# # Logestic regression over Bag of Words- Stemmed Text

# In[51]:


from sklearn.feature_extraction.text import CountVectorizer

# Create Bag of Words (BoW) representation using CountVectorizer
bow_vectorizer = CountVectorizer()
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)


# In[58]:


# Hyperparameter tuning using GridSearchCV
param_grid = {'C': [0.001, 0.01, 0.1,0.5]}  # List of values to search
lr_model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
grid_search = GridSearchCV(lr_model, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train_bow, y_train)


# In[59]:


# Get the best hyperparameters
best_c = grid_search.best_params_['C']


# In[60]:


# Create the logistic regression model with the best hyperparameter
lr_model_bow = LogisticRegression(C=best_c, max_iter=1000)


# In[61]:


# Train the model
lr_model_bow.fit(X_train_bow, y_train)


# In[62]:


# Make predictions on train and test data
y_train_pred = lr_model_bow.predict_proba(X_train_bow)[:, 1]
y_test_pred = lr_model_bow.predict_proba(X_test_bow)[:, 1]


# In[63]:


# Calculate AUC for train and test data
train_auc = roc_auc_score(y_train, y_train_pred)
test_auc = roc_auc_score(y_test, y_test_pred)


# ### Model Evaluation

# In[64]:


# Plot the ROC curve
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)

plt.figure(figsize=(6,2))
plt.plot(fpr_train, tpr_train, label=f'Train AUC = {train_auc:.2f}')
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {test_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Print the best hyperparameter and its corresponding ROC AUC score
print("Best C:", best_c)
print("Best ROC AUC:", grid_search.best_score_)


# ### Confusion Matrix and Classification Report

# In[65]:


# Calculate confusion matrix and classification report
threshold = 0.5
y_train_pred_binary = (y_train_pred >= threshold).astype(int)
y_test_pred_binary = (y_test_pred >= threshold).astype(int)

train_confusion_matrix = confusion_matrix(y_train, y_train_pred_binary)
test_confusion_matrix = confusion_matrix(y_test, y_test_pred_binary)

train_classification_report = classification_report(y_train, y_train_pred_binary)
test_classification_report = classification_report(y_test, y_test_pred_binary)


# In[40]:


# Print the confusion matrix and classification report
print("Train Confusion Matrix:")
print(train_confusion_matrix)
print("\nTest Confusion Matrix:")
print(test_confusion_matrix)

print("\nTrain Classification Report:")
print(train_classification_report)
print("\nTest Classification Report:")
print(test_classification_report)


# ## XGBoost over Bag of Words- Stemmed Text

# In[165]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

# Define the parameter grid
param_dist = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [5, 7, 10,30],
    'n_estimators': [40,100, 200],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 1, 2]
}




# In[166]:


# Create XGBoost classifier
xgb_model = xgb.XGBClassifier(objective='binary:logistic')



# In[167]:


# Perform randomized search with early stopping
random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=20,  # Number of random combinations to try
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,  # Use all available CPU cores
    verbose=2,
    random_state=42
)


# In[168]:


# Fit the model with early stopping
random_search.fit(X_train_bow, y_train, early_stopping_rounds=10, eval_metric='auc', eval_set=[(X_test_bow, y_test)])


# ### AUC score

# In[171]:


# Print the best hyperparameters and corresponding ROC AUC score
print("Best Hyperparameters:", random_search.best_params_)
print("Best ROC AUC:", random_search.best_score_)


# In[174]:


# Get predicted probabilities for both train and test sets
y_train_pred_probs = random_search.predict_proba(X_train_bow)[:, 1]
y_test_pred_probs = random_search.predict_proba(X_test_bow)[:, 1]


# In[175]:


# Calculate ROC curves and AUC values
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_probs)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_probs)
roc_auc_train = roc_auc_score(y_train, y_train_pred_probs)
roc_auc_test = roc_auc_score(y_test, y_test_pred_probs)


# ### Model Evaluation

# In[188]:


# Plot ROC curves
plt.figure(figsize=(5,2))
plt.plot(fpr_train, tpr_train, color='skyblue', lw=2, label='Train ROC curve (AUC = %0.2f)' % roc_auc_train)
plt.plot(fpr_test, tpr_test, color='orange', lw=2, label='Test ROC curve (AUC = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='green', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# ### Heatmaps

# In[95]:


# Extract the results from the randomized search
results = random_search.cv_results_


# In[124]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Create a DataFrame from the RandomizedSearchCV results
results_df = pd.DataFrame(random_search.cv_results_)


# In[162]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your DataFrame is named results_df
results_df['param_max_depth'] = results_df['param_max_depth'].astype(int)

# Pivot the DataFrame for test scores with aggregation
test_pivot_df = results_df.pivot_table(index='param_max_depth', columns='param_n_estimators', values='mean_test_score', aggfunc=np.mean)

# Create a heatmap for test scores
plt.figure(figsize=(5,4))
sns.heatmap(test_pivot_df, annot=True, fmt=".3f", cmap="Reds")
plt.title('Test Scores Heatmap')
plt.xlabel('param_n_estimators')
plt.ylabel('param_max_depth')
plt.show()


# In[161]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your DataFrame is named results_df
results_df['param_max_depth'] = results_df['param_max_depth'].astype(int)

# Pivot the DataFrame for CV scores with aggregation
cv_pivot_df = results_df.pivot_table(index='param_max_depth', columns='param_n_estimators', values='mean_test_score', aggfunc=np.mean)

# Create a heatmap for CV scores with a red color palette
plt.figure(figsize=(5,4))
sns.heatmap(cv_pivot_df, annot=True, fmt=".3f", cmap="Reds")
plt.title('CV Scores Heatmap')
plt.xlabel('param_n_estimators')
plt.ylabel('param_max_depth')
plt.show()


# ### Confusion Matrix

# In[195]:


# Get predictions
y_train_pred = random_search.predict(X_train_bow)
y_test_pred = random_search.predict(X_test_bow)

# Calculate confusion matrices
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("Train Confusion Matrix:")
print(conf_matrix_train)

print("\nTest Confusion Matrix:")
print(conf_matrix_test)


# ### Classification report

# In[196]:


# Generate classification reports
class_report_train = classification_report(y_train, y_train_pred, target_names=['Negative', 'Positive'])
class_report_test = classification_report(y_test, y_test_pred, target_names=['Negative', 'Positive'])

print("Classification Report (Train):\n", class_report_train)
print("Classification Report (Test):\n", class_report_test)


# In[ ]:





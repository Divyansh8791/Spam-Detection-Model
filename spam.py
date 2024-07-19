import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

############################################################### PART-1 : DATA PREPROCESSING

df = pd.read_csv('spam.csv', encoding='ISO-8859-1') 

# drop unnecessary columns .
df.drop(['Unnamed: 2', 'Unnamed: 3' , 'Unnamed: 4'] , axis = 1 , inplace = True)

# removing all html tags from each row of text data.
import re
def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)
    
for i in range(0 , 5572):
    df['v2'][i] = remove_html_tags(df['v2'][i]) 
    df['v2'][i]  = df['v2'][i].lower()


# remove all the stoping words like : is , this , a , that , and , of  etc:-
import spacy
# Load the English NLP model
nlp = spacy.load('en_core_web_sm')
def remove_stopwords(text):
    doc = nlp(text)
    filtered_words = [token.text for token in doc if not token.is_stop]
    return ' '.join(filtered_words)

for i in range(0 , 5572):
    df['v2'][i] = remove_stopwords(df['v2'][i])


########################################################################### PART-3 : SELCTING THE MODEL AND TRAIN THE MODEL.

# change the textual data into needed numerical data.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


x = df['v2']
y = df['v1']

# split the data into training and testing dataset.
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y, test_size = 0.3 , random_state = 0)


# making the model.
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Training the model
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# Evaluation of output . 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Calculate the confusion matrix
from sklearn.metrics import confusion_matrix , precision_score , recall_score
# re = recall_score(y_test , predictions)
# pre  = precision_score(y_test , predictions)
confu  = confusion_matrix(y_test, predictions)
print()
# print()
# print("Precesion Score : " , pre)
# print()
# print("Recall Score : " , re)
# print()
print("Confusion matrix :" , confu)
print()
print()
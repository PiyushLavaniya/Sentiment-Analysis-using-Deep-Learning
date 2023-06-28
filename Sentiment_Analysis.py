#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')


# In[4]:


#reading the data

df = pd.read_csv('Dataset/Reviews.csv')


# In[5]:


df.head()


# In[6]:


ax = df['Score'].value_counts().sort_index().plot(kind = 'bar', title = 'Reviews according to Score', figsize = (20, 10))
ax.set_xlabel('Review Stars')
plt.show()


# In[7]:


df['Text'].values[0]


# In[8]:


print(df.shape)


# # BASIC NLTK OPERATIONS

# In[9]:


#lets take as single example

example = df['Text'][50]
example


# In[10]:


tokens = nltk.word_tokenize(example) #we needto convert our text into a format that computer can interpret and tokenizing isthe way to do it.
tokens[:10]


# In[11]:


#nltk can help in getting the text's part of speech

tagged = nltk.pos_tag(tokens)
tagged[:10]


# In[12]:


#nltk can automatically extract some more interesting information about the text
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# # Using NLTK 's VADER Scoring 

# In[13]:


#To find out the positive, negative and neutral scores pf the specific sentence

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm


# In[14]:


SIA = SentimentIntensityAnalyzer()


# In[15]:


#let's run it for a certain example

SIA.polarity_scores('I am doing really good.') #as you can say it says that it is a positive sentiment and the compound score tell that the value from -1 to 1, that is, it is positive so it is determining it as a positive sentiment


# In[16]:


SIA.polarity_scores('That is too bad') #as you can say this time it is focusing onit ot be a negative sentiment


# In[17]:


SIA.polarity_scores(example) #as you can see, on our example it says that it is a negative response


# In[46]:


#we will need to run this analyzer on our dataset and store the values in a dictionary for further use
Results = {}
for i, row in tqdm(df.iterrows(), total = len(df)):
    text = row['Text']
    my_id = row['Id']
    Results[my_id] = SIA.polarity_scores(text)


# In[47]:


Results


# In[57]:


Results_df = pd.DataFrame(Results).T


# In[58]:


Results_df = Results_df.reset_index().rename(columns={'index': 'Id'})
Results_df = Results_df.merge(df, how='left')


# In[59]:


Results_df.head()


# In[61]:


ax = sns.barplot(data = Results_df, x = 'Score', y = 'compound')
ax.set_title('Amazon Review Scores')
plt.show()


# In[64]:


fig, axs = plt.subplots(1, 3, figsize = (17, 6))
sns.barplot(data = Results_df, x = 'Score', y = 'pos', ax = axs[0])
sns.barplot(data = Results_df, x = 'Score', y = 'neg', ax = axs[1])
sns.barplot(data = Results_df, x = 'Score', y = 'neu', ax = axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Negative')
axs[2].set_title('Neutral')
plt.tight_layout()
plt.show()


# # Now let's use a Pre-trained model from huggingface library

# In[1]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[23]:


Model = f"cardiffnlp/twitter-roberta-base-sentiment"
Tokenizer = AutoTokenizer.from_pretrained(Model)
model = AutoModelForSequenceClassification.from_pretrained(Model)


# In[24]:


example


# In[30]:


encoded_text = Tokenizer(example, return_tensors='pt')


# In[33]:


output = model(**encoded_text)
output


# In[34]:


Scores = output[0][0].detach().numpy()


# In[37]:


Results = softmax(Scores)
Results


# In[38]:


RESULTS = {
    
    'Negative_score' : Results[0],
    'Neutral_score' : Results[1],
    'Positive_score' : Results[2]
}


# In[39]:


RESULTS


# In[ ]:





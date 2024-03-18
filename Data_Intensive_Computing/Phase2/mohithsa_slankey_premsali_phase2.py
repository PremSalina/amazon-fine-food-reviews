#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis using Amazon Fine Food Reviews

# <h3>Problem Statement:</h3>

# This dataset consists of reviews of fine foods from Amazon. For a given review, determining whether the review is positive (rating of 4 or 5) or negative (rating of 1 or 2).

# ### Background of the Problem:

# This data spans more than ten years and includes over 500k customer reviews as of October 2012. Reviews contain information on the product and the user, as well as ratings and simple language. Additionally, all other Amazon categories reviews are included. This dataset is available on the Kaggle website (https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).
# 

# ### Significance of the problem:

# 1) The reviews in this dataset are actual consumer-generated content from the Amazon platform. This means it reflects the language, writing style, and sentiments expressed by real people when reviewing food products.
# 
# 2) The dataset spans over a decade, allowing for the analysis of changing trends and sentiments over time. This longitudinal perspective can be valuable for understanding how consumer opinions about food products evolve.

# ### This Project Potential towards Sentiment Analysis:

# 1) It poses various natural language processing (NLP) challenges like dealing with stopwords,stemming, and Lemmatization . This makes it a good benchmark for evaluating the robustness of sentiment analysis models.
# 
# 2) The dataset is not limited to a specific category of food products. It covers a wide range of food items, making it applicable to a broad spectrum of food-related sentiment analysis tasks.
# 
# 3) The dataset contains a wide range of opinions, from highly positive to highly negative, as well as neutral reviews. This diversity is important for training models that can handle various sentiment expressions.

# ### Overview of Dataset:

# Number of reviews: 568,454 </br>
# Number of users: 256,059 </br>
# Number of products: 74,258 </br>
# Timespan: Oct 1999 — Oct 2012 </br>
# Number of Attributes/Columns in data: 10 </br>

# ### Attributes Information:

# Id - Amazon Customer ID</br>
# ProductId — unique identifier for the product </br>
# UserId — unqiue identifier for the user </br>
# ProfileName - Profile Name of the Customer </br>
# HelpfulnessNumerator — number of users who found the review helpful </br>
# HelpfulnessDenominator — number of users who found the review helpful or not </br>
# Score — rating between 1 to 5 </br>
# Time — timestamp for the review </br>
# Summary — brief summary of the review </br>
# Text — text of the review </br>

# ### Source for this Dataset:

# This dataset is available on the Kaggle website </br>
# <h5>Link :</h5> <a>https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews </a>

# ### Data Cleaning/PreProcessing : 

# In[1]:


#libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import tqdm as tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


# ### Loading the dataset

# In[2]:


raw_data= pd.read_csv("/Users/mohithsainattam/Desktop/Reviews.csv")
raw_data.head()


# ### Shape or size of our dataset

# In[3]:


raw_data.shape


# We can see that 500K users reviewed for the product and there are 10 different type of feature colums given in our dataset

# ### Count of respective scores for all the reviews

# In[4]:


raw_data['Score'].value_counts().plot(kind = 'barh')


# 1) we can see that data is imbalanced and dataset consist of more 5 rating reviews and more positive reviews(score>3) compared to negative reviews.

# In[5]:


filtered_data = raw_data[raw_data['Score']!=3]
filtered_data.shape


# 1) we consider score = 3 as neutral and this cannot be used to judge if user is positive or negative towards the product review. So, we eliminate or clean those data points in our dataset

# In[6]:


filtered_data['Score'].value_counts().plot(kind = 'barh')


# ### Converting Score(numeric data) to binary classification label

# In[7]:


warnings.filterwarnings('ignore')
def partition(x):
    if x<3:
        return 'negative'
    else:
        return 'positive'
dummy_data=filtered_data['Score']
review_column_data=dummy_data.map(partition)
filtered_data['Score']=review_column_data
print(filtered_data.shape)
filtered_data.head()


# 1) We have converted scores>3 as positive and scores<3 as negative , so our problem is now a binary class classification problem

# In[8]:


filtered_data['Score'].value_counts()


# ### 2. Drop Duplicates

# In[9]:


duplicates = filtered_data[filtered_data.duplicated(['ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Score','Time','Summary','Text'])]
print(duplicates.shape)
duplicates


# 1) We can see that there are 256 rows of duplicates found for one product and same user in our dataset that needs to be eliminated.

# In[10]:


sorted_data=filtered_data.sort_values('ProductId',axis=0,ascending=True,inplace=False,kind='quicksort',na_position='last')
print(sorted_data.shape)
sorted_data.head()


# 1) sorting the data before removing duplicates, so that we get the first review of every duplicated record

# In[11]:


#drop_duplicates
final_data=sorted_data.drop_duplicates(subset={'ProfileName','UserId','Time','Text'},keep='first',inplace=False)
print(final_data.shape)


# 1) So, after eliminating the duplicate records in our dataset, we now have 364K reviews from 500k original dataset

# #### 3. Finding Invalid data and cleaning

# In[12]:


#HelpfulnessNumerator should be less than HelpfulnessDenominator. checking if any records are invalid with this scenerio.
invalid_rows = final_data[final_data['HelpfulnessNumerator'] > final_data['HelpfulnessDenominator']]
invalid_rows


# 1) we see that there are two records with HelpfulnessNumerator greater than HelpfulnessDenominator, which is practically not posible as HelpfulnessNumerator represents how many customers find the review to be only helpful whereas HelpfulnessDenominator represents how many customers find the review to be both helpful and not helpful(few may find helpful and few may find not helpful. summation of both is HelpfulnessDenominator).

# In[13]:


final_data=final_data[final_data.HelpfulnessNumerator<=final_data.HelpfulnessDenominator]
final_data.shape


# ### 4.Checking on any missing values in our dataset

# In[14]:


missing_data_percentage = (final_data.isna().mean() * 100).round(2)
missing_data_percentage


# There is no missing values in our dataset.

# In[15]:


(final_data.size/raw_data.size)*100


# We can see that after cleaning the data , we have obtained records for 64% of our original dataset. we have cleaned rest of 36% of the data.

# In[16]:


final_data['Score'].value_counts()


# we can see that our data consists of more positive reviews and less number of negative reviews. So, our dataset is imbalanced. Models may not perform well for imbalanced datasets. So, we should take balanced amount of positive and negative reviews.

# ### Balancing the dataset

# In[17]:


positive_reviews=final_data[final_data['Score'] == 'positive'].sample(n=2000, random_state=42)
negative_reviews=final_data[final_data['Score'] == 'negative'].sample(n=2000, random_state=42)
final_balanced_data=pd.concat([positive_reviews, negative_reviews])
print(final_balanced_data.iloc[0])
print('*************************************************************')
print(final_balanced_data.iloc[2001])


# 1) we have randomly sampled 20k positive reviews and 20k negative reviews and combined those to balance our dataset.

# In[18]:


final_balanced_data['Score'].value_counts().plot(kind='barh')


# 1) Above plot shows that our dataset is balanced with equal count of both positive and negative reviews

# ### Text Preprocessing

# #### 5. Removing HTML tags in our review text

# In[19]:


#printing html tags in one of our review text

count=0
for sent in final_balanced_data['Text'].values:
    if(len(re.findall('<.*?>',sent))):
        print(sent)
        break
    count+=1
print(count)


# 1) From the above text, we can clearly see that there are some HTML tags between our review text, which is of no meaning and should be cleaned in our dataset

# ####  Functions to remove html tags.

# In[20]:


#defining functions for removing html tags and punctuations
def removehtml(review_text):
    clean_html = re.compile('<.*?>')
    cleantext = re.sub(clean_html, ' ', review_text)
    return cleantext


# #### 6. Removing punctuations in our review text

# In[21]:


#### Functions to remove punctuations.


# In[22]:


def removepunc(review_text):
    cleaned_punc = re.sub(r'[?|!|\'|"|#]',r'',review_text)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned_punc)
    return  cleaned


# #### 7. Removing stopwords in the text data

# In[23]:


warnings.filterwarnings('ignore')
nltk.download('stopwords')
stopwords_list = stopwords.words('english')
stopwords = set(stopwords_list)

print(stopwords)


# #### 8. performing stemming to our text data

# In[24]:


sn = nltk.stem.SnowballStemmer('english')
print(sn.stem('tasty'))


# 1) Above words are considered to be stopwords(which are language specific and has no influence in defining the meaning of sentence) and can be eliminated from the text. Also, we perform stemming for each word to eliminate data like (tasty, tastful,taste to tasti, which gives same meaning and with only one word) to achieve lemmatization.

# In[25]:


a=0
string1=' '
final_string=[]
all_positive_words=[]
all_negative_words=[] 
s=''
for sentence in final_balanced_data['Text'].values:
    filtered_sentence=[]
    sent=removehtml(sentence) # remove HTMl tags
    for word in sent.split():
        for cleaned_words in removepunc(word).split(): #remove puncuations
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stopwords): #Removing stopwords
                    sen=(sn.stem(cleaned_words.lower())).encode('utf8') #perform stemming and encoding
                    filtered_sentence.append(sen)
                    if (final_balanced_data['Score'].values)[a] == 'positive': 
                        all_positive_words.append(sen) 
                    if(final_balanced_data['Score'].values)[a] == 'negative':
                        all_negative_words.append(sen) 
                else:
                    continue
            else:
                continue 
    string1 = b" ".join(filtered_sentence) 
    
    final_string.append(string1)
    a+=1


# #### 9. Adding CleanedText column to our dataset

# In[26]:


#adding a column of CleanedText which displays the data after pre-processing of the review
final_balanced_data['CleanedText']=final_string  
final_balanced_data['CleanedText']=final_balanced_data['CleanedText'].str.decode("utf-8")
print(final_balanced_data.shape)
final_balanced_data.head()


# 1) We have added new column named "CleanedText" with this cleaned punctuations, HTML tags,stemming, lemmatization,removing special characters and alphanumeric data. We have also converted the text to utf-8 encoding for further text to vector supporting.

# In[27]:


# printing some random reviews
#positive reviews
review1 = final_balanced_data['CleanedText'].values[0]
print(review1)
print("-"*50)

review2 = final_balanced_data['CleanedText'].values[1000]
print(review2)
print("-"*50)

#negative reviews
review3 = final_balanced_data['CleanedText'].values[2001]
print(review3)
print("-"*50)

review4 = final_balanced_data['CleanedText'].values[3900]
print(review4)
print("-"*50)


# ### 10.Sorting the data based on timestamp

# In[28]:


# converting timestamp to datetime and sorting the data
final_balanced_data['Time']=pd.to_datetime(final_balanced_data['Time'],unit='s')
final_balanced_data=final_balanced_data.sort_values(by='Time')
final_balanced_data.head()


# 1) In the real world unseen data, we get the reviews that are latest. So, its better to train our model with old timestamp data and test the model with new timestamp data to perform well on real world unseen data(which is of latest time). As the time varies customer intrest may vary or product quality may vary. So, take this into consideration we perform sorting based on timestamp

# ### Featurization - EDA

# #### 1. Word Frequency Analysis

# In[29]:


nltk.download('punkt')
final_balanced_data['Tokenized_Text'] = final_balanced_data['CleanedText'].apply(lambda x: word_tokenize(x.lower()))

# Flatten the list of tokens
all_words = [word for tokens in final_balanced_data['Tokenized_Text'] for word in tokens]

# Calculate word frequencies
freq_dist = FreqDist(all_words)

# Get the most common words
top_words = freq_dist.most_common(20)

# Convert to a DataFrame for visualization
top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])

# Plot the word frequencies
plt.figure(figsize=(10, 6))
plt.bar(top_words_df['Word'], top_words_df['Frequency'])
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 20 Most Common Words')
plt.xticks(rotation=45)
plt.show()


# 1) Analyze the frequency distribution of words in the dataset to understand which words are most used. This Analysis will help us to check the weightage of words when the text is converted to vectors before modelling. Also, we have added new column with "Tokenized_Text" that has list of words in our sentences. This will help us to analyze word frequencies and to convert word to vector embedding.

# In[30]:


final_balanced_data.head()


# #### 2. Vocabulary Size

# In[31]:


all_words = [word for tokens in final_balanced_data['Tokenized_Text'] for word in tokens]
vocabulary_size = len(set(all_words))
print(all_words[3])
print(f"The vocabulary size is: {vocabulary_size}")


# 1) Vocabulary Size provides a numerical representation of how many unique words are present in the dataset.
# 2) As per the above observation, a very large vocabulary might lead to overfitting if not properly handled. Techniques like dimensionality reduction or sentence to vectors(like TF-IDF or word embeddings) may be applied before modeling as a feature engineering.

# #### 3.Word Clouds

# In[32]:


all_reviews = ' '.join(final_balanced_data['CleanedText'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# 1) Larger and bolder words in the word cloud are indicative of frequently occurring words in the text. These are likely to be significant keywords or terms in the dataset.
# 2) In our given dataset, Use, one, tast, good, flavor, love, product, food, amazon, order, great are few frequently occurring words in our text and are considered to be more important in giving weightage while modelling.

# #### 4. Sentiment by Document Length

# In[33]:


final_balanced_data['Review Length'] = final_balanced_data['CleanedText'].apply(lambda x: len(str(x).split()))
def partition(x):
    if x =='negative':
        return 0
    else:
        return 1
dummy_data=filtered_data['Score']
review_column_data=dummy_data.map(partition)
final_balanced_data['Numeric_Score']=review_column_data
final_balanced_data.head()


# 1) We have converted Numeric scores to 0 and 1 for further analysis and Also, we have added new column with "Review Length" that has count of words in our review text and "Numeric_Score" with 0 and 1 for negative and positive reviews respectively, which can be used for further analysis.

# In[34]:


# Filter positive and negative reviews
positive_reviews = final_balanced_data[final_balanced_data['Numeric_Score'] == 1]
negative_reviews = final_balanced_data[final_balanced_data['Numeric_Score'] == 0]

plt.figure(figsize=(10, 6))
plt.scatter(positive_reviews['Review Length'], positive_reviews['Numeric_Score'], color='green', label='Positive Reviews', alpha=0.5)
plt.scatter(negative_reviews['Review Length'], negative_reviews['Numeric_Score'], color='red', label='Negative Reviews', alpha=0.5)
plt.xlabel('Length of Review Text')
plt.ylabel('Sentiment (1 for Positive, 0 for Negative)')
plt.title('Scatter Plot of Review Text Lengths for Positive vs Negative Reviews')
plt.legend()
plt.show()


# 1) From the above plot we can see that most of the review text length for positive reviews lie between 0 to 400 words, whereas for negative reviews, that length lie between 0 to 500 except for few points that crossed 500 lengths.
# 2) This can help us to see the average count of words that customers given for both positive and negative reviews. As they are overlapping (similar for both positive and negative), we cannot judge sentiments based on length of review text alone.

# #### 5. Analysis of Helpful Votes

# In[35]:


filtered_data = final_balanced_data[final_balanced_data['HelpfulnessDenominator'] > 0]
filtered_data['HelpfulnessPercentage'] = filtered_data['HelpfulnessNumerator'] / filtered_data['HelpfulnessDenominator']
helpful_percentage_by_sentiment = filtered_data.groupby('Numeric_Score')['HelpfulnessPercentage'].mean()
plt.figure(figsize=(10, 6))
plt.bar(helpful_percentage_by_sentiment.index, helpful_percentage_by_sentiment.values, color=['red', 'green'])
plt.xlabel('Sentiment (0 for Negative, 1 for Positive)')
plt.ylabel('Average Helpful Percentage')
plt.title('Relationship between Sentiment and Helpful Votes')
plt.xticks([0, 1], ['Negative', 'Positive'])
plt.show()


# 1) From the above plot, we can see that positive reviews are found more helpful for customers compared to negative reviews. So, this shows that positive reviews can be similar in real world data and model can perform better for positive reviews compared to negative reviews as most of the customers agree with positive words that are present in our dataset.

# #### 6. Length of Title vs. Length of Review Text 

# In[36]:


final_balanced_data['Title Length'] = final_balanced_data['Summary'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10, 6))
plt.scatter(final_balanced_data['Title Length'], final_balanced_data['Review Length'], alpha=0.5)
plt.xlabel('Title Length')
plt.ylabel('Review Text Length')
plt.title('Length of Title vs Lenth of Review Text')
plt.show()


# 1) From the above plot, we can observe that most of the review title lengths fall under 0 to 20 range. So, we can tell that customers are not willing to explain their thought clearly in title itself.
# 2) So, considering the review text. it would be better and provide more information than title for modelling.

# ### Converting Review Text to Vector for modelling - Feature Engineering

# #### 7. BAG OF WORDS

# In[37]:


#splitting the data into train and test
x_train,x_test,y_train,y_test=train_test_split(final_balanced_data['CleanedText'],final_balanced_data['Score'],test_size=0.3,random_state=42)
x_train,x_cv,y_train,y_cv=train_test_split(x_train,y_train,test_size=0.3,random_state=42)
print(x_train.shape,y_train.shape,y_test.shape,x_test.shape,x_cv.shape,y_cv.shape)


# we are splitting the data to train and text, so that test data(text) is not seen by training data while applying bag of words, tf-idf, word2vec, avg w2v techniques to convert text to vectors.

# In[38]:


count_vect = CountVectorizer(min_df=10, max_features=500)
count_vect.fit(x_train)
print("some feature names ", count_vect.get_feature_names_out()[:20])
print('-'*50)

x_train_bow = count_vect.transform(x_train)
x_test_bow = count_vect.transform(x_test)
x_cv_bow = count_vect.transform(x_cv)
print("Type of count vectorizer ",type(x_train_bow))
print("Shape of BOW vectorizer ",x_train_bow.get_shape())
print("Number of unique words ", x_train_bow.get_shape()[1])
print(x_train_bow.toarray())
print(x_train_bow.shape,y_train.shape)
print(x_test_bow.shape,y_test.shape)
print(x_cv_bow.shape,y_cv.shape)


# The "bags-of-words" form, which ignores structure and instead counts the frequency of each word, is the simplest and most natural way to accomplish this. The bags-of-words is applied using CountVectorizer, which transforms a group of text documents into a matrix of token counts. Our collection of text documents is transformed into a token count matrix by instantiating the CountVectorizer and fitting it to our training data.

# ### 7.1 Bi-Grams

# In[39]:


count_vect = CountVectorizer(ngram_range=(1,2), min_df=10, max_features=5000)
x_train_bow_bigram = count_vect.fit_transform(x_train)
x_test_bow_bigram = count_vect.fit_transform(x_test)
x_cv_bow_bigram = count_vect.transform(x_cv)
print("some feature names ", count_vect.get_feature_names_out()[:20])
print('-'*50)
print("Type of count vectorizer ",type(x_train_bow_bigram))
print("Shape of BOW vectorizer ",x_train_bow_bigram.get_shape())
print("Number of unique words with both unigrams and bigrams ", x_train_bow_bigram.get_shape()[1])
print(x_train_bow_bigram.toarray())
print(x_train_bow_bigram.shape,y_train.shape)
print(x_test_bow_bigram.shape,y_test.shape)
print(x_cv_bow_bigram.shape,y_cv.shape)


# bag-of-bigrams representation is much more powerful than bag-of-words.  bigram refers to a combination of two adjacent words in a text. 

# ### 8. Term Frequency- Inverse Document Frequency (TF-IDF):

# In[40]:


tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)
tf_idf_vect.fit(x_train)
print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names_out()[0:20])
print('-'*50)
x_train_tf_idf = tf_idf_vect.transform(x_train)
x_test_tf_idf = tf_idf_vect.transform(x_test)
print("Type of count vectorizer ",type(x_train_tf_idf))
print("Shape of TFIDF vectorizer ",x_train_tf_idf.get_shape())
print("Number of unique words including both unigrams and bigrams ", x_train_tf_idf.get_shape()[1])
print(x_train_tf_idf.shape,y_train.shape)
print(x_test_tf_idf.shape,y_test.shape)


# Tf-idf allows us to weight terms based on how important they are to a document. These extremely common phrases would cover the frequencies of less common but more interesting terms if we sent the count data straight to a classifier.

# ### 9. Word2Vec

# In[41]:


text_data=final_balanced_data['CleanedText'].values
labels=final_balanced_data['Score'].values


# In[42]:


x_train,x_test,y_train,y_test=train_test_split(text_data,labels,test_size=0.3) #random splitting

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[43]:


i=0
train_list_of_sent=[]
for sent in x_train:
    train_list_of_sent.append(sent.split())
train_list_of_sent[0]


# In[44]:


train_w2v_model=Word2Vec(train_list_of_sent,min_count=5,vector_size=50,workers=4)


# In[45]:


train_w2v_words = list(train_w2v_model.wv.key_to_index)
print("number of words that occured minimum 5 times ",len(train_w2v_words))
print("sample words ", train_w2v_words[0:50])


# In[46]:


i=0
test_list_of_sent=[]
for sentance in x_test:
    test_list_of_sent.append(sentance.split())
test_list_of_sent[0]


# In[47]:


test_w2v_model=Word2Vec(test_list_of_sent,min_count=5,vector_size=50, workers=4)   


# In[48]:


test_w2v_words = list(test_w2v_model.wv.key_to_index)
print("number of words that occured minimum 5 times ",len(test_w2v_words))
print("sample words ", test_w2v_words[0:50])


# In[49]:


# Get the dictionary mapping words to integer indices
w2v_words_dict = train_w2v_model.wv.key_to_index
# Get the list of word strings
train_w2v_words_list = train_w2v_model.wv.index_to_key
print("number of words that occured minimum 5 times ", len(train_w2v_words_list))
print("sample words ", train_w2v_words_list[0:50])


# In[50]:


# Get the dictionary mapping words to integer indices
w2v_words_dict = test_w2v_model.wv.key_to_index
# Get the list of word strings
test_w2v_words_list = test_w2v_model.wv.index_to_key
print("number of words that occured minimum 5 times ", len(test_w2v_words_list))
print("sample words ", test_w2v_words_list[0:50])


# In[51]:


word_to_check = 'awesom'

if word_to_check in train_w2v_model.wv.key_to_index:
    similar_words = train_w2v_model.wv.most_similar(word_to_check)
    print(f"Words similar to '{word_to_check}':")
    for word, similarity in similar_words:
        print(f"{word}: {similarity}")
else:
    print(f"'{word_to_check}' is not in the vocabulary.")


# 1) Word2Vec captures semantic relationships between words. Words with similar meanings tend to be closer in the embedding space.
# 2) This can be very helpful and with less dimensions with meaningful words given equal weightage. So, model can give better results if it sees similar kind of words in unseen data

# ### 9.1 Average -word2vec

# In[52]:


train_vectors = []; 
for sent in (train_list_of_sent): 
    sent_vec = np.zeros(50)  
    cnt_words =0; 
    for word in sent: 
        if word in train_w2v_words:
            vec = train_w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    train_vectors.append(sent_vec)
train_vectors = np.array(train_vectors)
print(train_vectors.shape)
print(train_vectors[0])


# In[53]:


print(train_vectors[0])


# In[54]:


test_vectors = []; 
for sent in test_list_of_sent: 
    sent_vec = np.zeros(50) 
    cnt_words =0; 
    for word in sent:
        if word in test_w2v_words:
            vec = test_w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    test_vectors.append(sent_vec)
test_vectors = np.array(test_vectors)
print(test_vectors.shape)
print(test_vectors[0])


# In[55]:


print(test_vectors[0])


# In[270]:


x_train_avg_w2v=train_vectors
x_test_avg_w2v=test_vectors


# In[271]:


print(x_train_avg_w2v.shape)
print(x_test_avg_w2v.shape)


# 1) word2vec produces individual word vectors for each word in a text. whereas AvgWord2vec aggregates the vectors of all words in a text to create a single vector representation for the entire text.
# 2) AvgWord2vec results in a single vector of the same dimensionality as the individual word vectors. But, it treats all words in a text equally, potentially losing some contextual information. we need to decide which (w2v or AvgW2v) is best based on the computational complexity and metric scores of model that is being used.

# ### 10. TFIDF weighted w2v

# In[372]:


x_train,x_test,y_train,y_test=train_test_split(text_data,labels,test_size=0.3,random_state=5)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[373]:


dictionary = dict(zip(tf_idf_vect.get_feature_names_out(), list(tf_idf_vect.idf_)))


# In[374]:


# tf_idf_w2v words of train data
tfidf_feat=tf_idf_vect.get_feature_names_out()
train_tfidf_w2v=[]
row = 0
for sent in train_list_of_sent:
    sent_vec = np.zeros(50)
    weight_sum = 0
    for word in sent:
        if word in train_w2v_words and word in tfidf_feat:
            vec = train_w2v_model.wv[word]
            tf_idf = dictionary[word]*(sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    train_tfidf_w2v.append(sent_vec)
    row += 1
len(train_tfidf_w2v)


# In[375]:


print(train_tfidf_w2v[0])


# In[376]:


tfidf_feat = tf_idf_vect.get_feature_names_out()
test_tfidf_w2v = []
row = 0
for sent in test_list_of_sent:
    sent_vec = np.zeros(50)
    weight_sum = 0
    for word in sent:
        if word in test_w2v_words and word in tfidf_feat:
            vec = test_w2v_model.wv[word]
            tf_idf = dictionary[word]*(sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    test_tfidf_w2v.append(sent_vec)
    row += 1
len(test_tfidf_w2v)


# In[377]:


print(test_tfidf_w2v[0])


# In[378]:


x_train_tfidf_w2v=np.array(train_tfidf_w2v)
x_test_tfidf_w2v=np.array(test_tfidf_w2v)
print(x_train_tfidf_w2v.shape)
print(x_test_tfidf_w2v.shape)


# 1) TFIDF-Word2Vec Combines the principles of TF-IDF weighting and averaging Word2Vec vectors to create text representations.
# 2) It incorporates TF-IDF scores to give more weight to important words while creating the average vector. Provides a weighted average representation that focuses more on important words in the context of the entire corpus.
# 3) TF-IDF-Word2Vec is particularly useful in scenarios where the importance of words varies within the corpus.
# 4) Similar to word2vec and Avg-w2v, we need to decide which (w2v or AvgW2v or TF-idf w2v) is best based on the computational complexity and metric scores of model that is being used.

# # Project Phase-2

# ### Applying K-NN on Bow Features

# In[65]:


count_vect = CountVectorizer(min_df=10, max_features=500)
count_vect.fit(x_train)
print("some feature names ", count_vect.get_feature_names_out()[:20])
print('-'*50)

x_train_bow = count_vect.transform(x_train)
x_test_bow = count_vect.transform(x_test)
x_cv_bow = count_vect.transform(x_cv)
print("Type of count vectorizer ",type(x_train_bow))
print("Shape of BOW vectorizer ",x_train_bow.get_shape())
print("Number of unique words ", x_train_bow.get_shape()[1])
print(x_train_bow.toarray())
print(x_train_bow.shape,y_train.shape)
print(x_test_bow.shape,y_test.shape)
print(x_cv_bow.shape,y_cv.shape)


# In[139]:


def Grid_search(x_train,y_train,algorithm):
    cv=KFold(n_splits=5)
    myList = list(range(0,50))
    K=list(filter(lambda x: x % 2 != 0, myList))
    neigh=KNeighborsClassifier(algorithm=algorithm)
    parameters = {'n_neighbors':list(filter(lambda x: x % 2 != 0, myList))}
    clf = GridSearchCV(neigh, parameters, cv=cv, scoring='roc_auc',return_train_score=True,verbose=1)
    clf.fit(x_train, y_train)
    
    results = pd.DataFrame.from_dict(clf.cv_results_)
    results = results.sort_values(['param_n_neighbors'])

    train_auc= clf.cv_results_['mean_train_score']
    train_auc_std= clf.cv_results_['std_train_score']
    cv_auc = clf.cv_results_['mean_test_score'] 
    cv_auc_std= clf.cv_results_['std_test_score']
    best_k = clf.best_params_['n_neighbors']
    
    sns.set()
    plt.plot(K, train_auc, label='Train AUC')
    plt.gca().fill_between(K,train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

    plt.plot(K, cv_auc, label='CV AUC')
    plt.gca().fill_between(K,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')
    plt.scatter(K, train_auc, label='Train AUC points')
    plt.scatter(K, cv_auc, label='CV AUC points')
    plt.legend()
    plt.xlabel("K: hyperparameter")
    plt.ylabel("AUC")
    plt.title("ERROR PLOTS")
    plt.show()
    
    print("Best cross-validation score: {:.3f}".format(clf.best_score_))
    print('The best k from gridsearch :',best_k)
    return  best_k


# ##### Hyper parameter tuning

# ### ROC_AUC plot

# In[69]:


best_k=Grid_search(x_train_bow,y_train,'brute')


# In[135]:


from tqdm import tqdm
def cross_validation(x_train,y_train,x_cv,y_cv,algorithm):
    train_auc = []
    cv_auc = []
    myList = list(range(0,50))
    K=list(filter(lambda x: x % 2 != 0, myList))
    for i in tqdm(K):
        neigh = KNeighborsClassifier(n_neighbors=i,algorithm=algorithm)
        neigh.fit(x_train, y_train)

   # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs
        y_train_pred =  neigh.predict_proba(x_train)[:,1]
        y_cv_pred =  neigh.predict_proba(x_cv)[:,1]
    
        train_auc.append(roc_auc_score(y_train,y_train_pred))
        cv_auc.append(roc_auc_score(y_cv, y_cv_pred))

    sns.set()    
    plt.plot(K, train_auc, label='Train AUC')
    plt.plot(K, cv_auc, label='CV AUC')

    plt.scatter(K, train_auc, label='Train AUC points')
    plt.scatter(K, cv_auc, label='CV AUC points')

    plt.legend()
    plt.xlabel("K: hyperparameter")
    plt.ylabel("AUC")
    plt.title("ERROR PLOTS")
    plt.grid()
    plt.show()


# In[71]:


cross_validation(x_train_bow,y_train,x_cv_bow,y_cv,'brute')


# In[72]:


print(best_k)


# In[73]:


# Convert labels to binary format
lb = LabelBinarizer()
y_train_binary = lb.fit_transform(y_train)
y_test_binary = lb.transform(y_test)


# In[136]:


from sklearn.metrics import roc_curve, auc
def test_data(x_train,y_train,x_test,y_test,algorithm):
    neigh = KNeighborsClassifier(n_neighbors=best_k,algorithm=algorithm, n_jobs=-1)
    neigh.fit(x_train, y_train)
    
    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs

    train_fpr, train_tpr, thresholds = roc_curve(y_train, neigh.predict_proba(x_train)[:,1])
    test_fpr, test_tpr, thresholds = roc_curve(y_test, neigh.predict_proba(x_test)[:,1])

    sns.set()
    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
    plt.legend()
    plt.xlabel("False_positive_rate")
    plt.ylabel("True positive_rate")
    plt.title("ROC_Curve")
    plt.grid()
    plt.show()
    print('The AUC_score of test_data is :',auc(test_fpr, test_tpr))


# In[75]:


test_data(x_train_bow,y_train_binary,x_test_bow,y_test_binary,'brute')


# In[137]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def metric(x_train,y_train,x_test,y_test,algorithm):
    knn=KNeighborsClassifier(n_neighbors=best_k,algorithm=algorithm)
    knn.fit(x_train,y_train)
    predict=knn.predict(x_test)

    conf_mat = confusion_matrix(y_test, predict)
    class_label = ["Negative", "Positive"]
    df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
    
    report=classification_report(y_test,predict)
    print(report)
    
    sns.set()
    sns.heatmap(df, annot = True,fmt="d")
    plt.title("Test_Confusion_Matrix")
    plt.xlabel("Predicted_Label")
    plt.ylabel("Actual_Label")
    plt.show()


# In[77]:


metric(x_train_bow,y_train,x_test_bow,y_test,'brute')


# #### Applying KNN brute force on TFIDF Features

# In[78]:


tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=10)
tf_idf_vect.fit(x_train)
print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names_out()[0:20])
print('-'*50)
x_train_tf_idf = tf_idf_vect.transform(x_train)
x_test_tf_idf = tf_idf_vect.transform(x_test)
print("Type of count vectorizer ",type(x_train_tf_idf))
print("Shape of TFIDF vectorizer ",x_train_tf_idf.get_shape())
print("Number of unique words including both unigrams and bigrams ", x_train_tf_idf.get_shape()[1])
print(x_train_tf_idf.shape,y_train.shape)
print(x_test_tf_idf.shape,y_test.shape)


# In[79]:


best_k=Grid_search(x_train_tf_idf,y_train,'brute')


# In[81]:


test_data(x_train_tf_idf,y_train_binary,x_test_tf_idf,y_test_binary,'brute')


# In[82]:


metric(x_train_tf_idf,y_train,x_test_tf_idf,y_test,'brute')


# #### Applying K-NN on Avg-Word2Vec

# In[120]:


print(x_train_avg_w2v.shape)
print(x_test_avg_w2v.shape)
print(y_train.shape)
print(y_test.shape)


# In[84]:


best_k=Grid_search(x_train_avg_w2v,y_train,'brute')


# In[126]:


test_data(x_train_avg_w2v,y_train_binary,x_test_avg_w2v,y_test_binary,'brute')


# In[129]:


metric(x_train_avg_w2v,y_train,x_test_avg_w2v,y_test,'brute')


# #### Applying k-NN on TF-IDF_W2V

# In[132]:


print(x_train_tfidf_w2v.shape)
print(x_test_tfidf_w2v.shape)
print(y_train.shape)
print(y_test.shape)


# In[140]:


best_k=Grid_search(x_train_tfidf_w2v,y_train,'brute')


# In[141]:


test_data(x_train_tfidf_w2v,y_train_binary,x_test_tfidf_w2v,y_test_binary,'brute')


# In[142]:


metric(x_train_tfidf_w2v,y_train,x_test_tfidf_w2v,y_test,'brute')


# In[143]:


get_ipython().system('pip install prettytable')
from prettytable import PrettyTable
    
table = PrettyTable()
table.field_names = ["Vectorizer", "Model", "Hyper_Parameter(K)", "AUC_Score"]
table.add_row(["Bow", 'K_NN_Brute_Force', 49,78.7 ])
table.add_row(["TFIDF", 'K_NN_Brute_Force', 49, 84.1])
table.add_row(["Avg_Word2vec", 'K_NN_Brute_Force', 47, 49.9,])
table.add_row(["TFIDF_Word2vec", 'K_NN_Brute_Force',1 ,49.14 ])
print(table)


# ### Observation:

# 1) From the above table we observed that TF-IDF having the highest AUC score on test data
# 2) TF-IDF model also works reasonabally good on test data .

# # Applying Naive_bayes on BOW

# In[144]:


print(x_train_bow.shape,y_train.shape)
print(x_test_bow.shape,y_test.shape)
print(x_cv_bow.shape,y_cv.shape)


# In[160]:


from sklearn.naive_bayes import MultinomialNB
import math
def Grid_search(X_train,Y_train):
    cv=KFold(n_splits=5)
    alpha_values = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]#alpha from 10^-5 to 10^5
    model=MultinomialNB()
    parameters = {'alpha':alpha_values}
    clf = GridSearchCV(model,parameters, cv=cv, scoring='roc_auc',return_train_score=True,verbose=1)
    clf.fit(X_train, Y_train)
    results = pd.DataFrame.from_dict(clf.cv_results_)
    results = results.sort_values(['param_alpha'])
    train_auc= clf.cv_results_['mean_train_score']
    train_auc_std= clf.cv_results_['std_train_score']
    cv_auc = clf.cv_results_['mean_test_score'] 
    cv_auc_std= clf.cv_results_['std_test_score']
    best_alpha= clf.best_params_['alpha']
    sns.set()
    alpha_values=[math.log(x) for x in alpha_values]
    plt.plot(alpha_values, train_auc, label='Train AUC')
    plt.gca().fill_between(alpha_values,train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')
    plt.plot(alpha_values, cv_auc, label='CV AUC')
    plt.gca().fill_between(alpha_values,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')
    plt.scatter(alpha_values, train_auc, label='Train AUC points')
    plt.scatter(alpha_values, cv_auc, label='CV AUC points')
    plt.legend()
    plt.xlabel("alpha_values: hyperparameter")
    plt.ylabel("AUC")
    plt.title("ERROR PLOTS")
    plt.show()
    
    print("Best cross-validation score: {:.3f}".format(clf.best_score_))
    print('The best alpha from gridsearch :',best_alpha)
    return  best_alpha


# In[146]:


best_alpha=Grid_search(x_train_bow,y_train)


# In[147]:


best_alpha


# In[148]:


def test_data(x_train,y_train,x_test,y_test):
    model=MultinomialNB(alpha=best_alpha)
    model.fit(x_train, y_train)
    train_fpr, train_tpr, thresholds = roc_curve(y_train, model.predict_proba(x_train)[:,1])
    test_fpr, test_tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
    sns.set()
    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
    plt.plot([0, 1], [0, 1], color='green', lw=1, linestyle='--')
    plt.legend()
    plt.xlabel("False_positive_rate")
    plt.ylabel("True positive_rate")    
    plt.title("ROC_Curve")
    plt.grid()
    plt.show()
    print('The AUC_score of test_data is :',auc(test_fpr, test_tpr))


# In[149]:


def metric(x_train,y_train,x_test,y_test):
    model=MultinomialNB(alpha=best_alpha)
    model.fit(x_train, y_train)
    predict=model.predict(x_test)

    conf_mat = confusion_matrix(y_test, predict)
    class_label = ["Negative", "Positive"]
    df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
    
    report=classification_report(y_test,predict)
    print(report)
    
    sns.set()
    sns.heatmap(df, annot = True,fmt="d")
    plt.title("Test_Confusion_Matrix")
    plt.xlabel("Predicted_Label")
    plt.ylabel("Actual_Label")
    plt.show()


# In[150]:


test_data(x_train_bow,y_train_binary,x_test_bow,y_test_binary)


# In[151]:


metric(x_train_bow,y_train,x_test_bow,y_test)


# ### Applying Naive_Bayes on tf-idf feature

# In[152]:


print(x_train_tf_idf.shape,y_train.shape)
print(x_test_tf_idf.shape,y_test.shape)


# In[153]:


best_alpha=Grid_search(x_train_tf_idf,y_train)


# In[154]:


test_data(x_train_tf_idf,y_train_binary,x_test_tf_idf,y_test_binary)


# In[155]:


metric(x_train_tf_idf,y_train,x_test_tf_idf,y_test)


# In[164]:


from prettytable import PrettyTable
    
table = PrettyTable()
table.field_names = ["Vectorizer", "Feature engineering", " Hyper Parameter (alpha)", "AUC_Score"]
table.add_row(["Bow", 'Featurized', 10,88.9 ])
table.add_row(["TFIDF", 'Featurized', 1, 92.92])
print(table)


# ### Observation:

# 1) Compare to Bag of words features represntation , tf-idf features are got the highest 92.92% AUC score on Test data.
# 2) Both are having the 1 as the best alpha by Hyper parameter tuning.
# 3) Both models have resonabally works well for Amazon_food_reviews classification.

# # Apply Logistic regression on BOW

# In[368]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)
x_train_bow = sc.fit_transform(x_train_bow)
x_test_bow = sc.transform(x_test_bow)


# In[369]:


from sklearn.linear_model import LogisticRegression
C = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]#alpha from 10^-5 to 10^5
L1_model=LogisticRegression(penalty='l1',C=C,solver='liblinear',max_iter=500)


# In[169]:


def Grid_search(model,X_train,Y_train):
    
    parameters = {'C':C}
    cv=KFold(n_splits=5)
    clf = GridSearchCV(model,parameters, cv=cv, scoring='roc_auc',return_train_score=True)
    clf.fit(X_train, Y_train)
    
    results = pd.DataFrame.from_dict(clf.cv_results_)
    results = results.sort_values(['param_C'])

    train_auc= clf.cv_results_['mean_train_score']
    train_auc_std= clf.cv_results_['std_train_score']
    cv_auc = clf.cv_results_['mean_test_score'] 
    cv_auc_std= clf.cv_results_['std_test_score']
    best_C= clf.best_params_['C'] #c=1/lamda
    
    sns.set()
    C_values=[math.log(x) for x in C]
    plt.plot(C, train_auc, label='Train AUC')
    # this code is copied from here: https://stackoverflow.com/a/48803361/4084039
    plt.gca().fill_between(C,train_auc - train_auc_std,train_auc + train_auc_std,alpha=0.2,color='darkblue')

    plt.plot(C, cv_auc, label='CV AUC')
    # this code is copied from here: https://stackoverflow.com/a/48803361/4084039
    plt.gca().fill_between(C,cv_auc - cv_auc_std,cv_auc + cv_auc_std,alpha=0.2,color='darkorange')
    plt.scatter(C, train_auc, label='Train AUC points')
    plt.scatter(C, cv_auc, label='CV AUC points')
    plt.legend()
    plt.xlabel("C = 1/λ: hyperparameter")
    plt.ylabel("AUC")
    plt.title("ERROR PLOTS")
    plt.show()
    
    print("Best cross-validation score: {:.3f}".format(clf.best_score_))
    print('The best C from gridsearch :',best_C)
    return  best_C


# In[171]:


import warnings
L1_best_c=Grid_search(L1_model,x_train_bow,y_train)


# In[178]:


def test_data(model,x_train,y_train,x_test,y_test):
   
    model.fit(x_train, y_train)
    train_fpr, train_tpr, thresholds = roc_curve(y_train, model.predict_proba(x_train)[:,1])
    test_fpr, test_tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])
    sns.set()
    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
    plt.plot([0, 1], [0, 1], color='green', lw=1, linestyle='--')
    plt.legend()
    plt.xlabel("False_positive_rate")
    plt.ylabel("True positive_rate")    
    plt.title("ROC_Curve")
    plt.grid()
    plt.show()
    print('The AUC_score of test_data is :',auc(test_fpr, test_tpr))


# In[180]:


L1_model=LogisticRegression(penalty='l1',C=L1_best_c,solver='liblinear',max_iter=500)
test_data(L1_model,x_train_bow,y_train_binary,x_test_bow,y_test_binary)


# In[183]:


def metric(model,x_train,y_train,x_test,y_test):
    
    model.fit(x_train, y_train)
    predict=model.predict(x_test)

    conf_mat = confusion_matrix(y_test, predict)
    class_label = ["Negative", "Positive"]
    df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
    
    report=classification_report(y_test,predict)
    print(report)
    
    sns.set()
    sns.heatmap(df, annot = True,fmt="d")
    plt.title("Test_Confusion_Matrix")
    plt.xlabel("Predicted_Label")
    plt.ylabel("Actual_Label")
    plt.show()


# In[184]:


metric(L1_model,x_train_bow,y_train,x_test_bow,y_test)


# # Apply Logistic regression on TF-IDF

# In[185]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)
x_train_tf_idf = sc.fit_transform(x_train_tf_idf)
x_test_tf_idf = sc.transform(x_test_tf_idf)


# In[186]:


C = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]#alpha from 10^-5 to 10^5
L1_model=LogisticRegression(penalty='l1',C=C,solver='liblinear',max_iter=500)


# In[187]:


L1_best_c=Grid_search(L1_model,x_train_tf_idf,y_train)


# In[189]:


L1_model=LogisticRegression(penalty='l1',C=L1_best_c,solver='liblinear',max_iter=500)
test_data(L1_model,x_train_tf_idf,y_train_binary,x_test_tf_idf,y_test_binary)


# In[190]:


metric(L1_model,x_train_tf_idf,y_train,x_test_tf_idf,y_test)


# # Apply Logistic regression on AVG-W2V

# In[197]:


x_test_avg_w2v= np.array(x_train_avg_w2v)
x_test_avg_w2v= np.array(x_test_avg_w2v)
print(x_train_avg_w2v.shape)
print(x_test_avg_w2v.shape)


# In[203]:


C = [0.01,0.1,1,10]
L1_model=LogisticRegression(penalty='l1',C=C,solver='liblinear',max_iter=500)


# In[204]:


L1_best_c=Grid_search(L1_model,x_train_avg_w2v,y_train)


# In[ ]:


L1_model=LogisticRegression(penalty='l1',C=L1_best_c,solver='liblinear',max_iter=500)
test_data(L1_model,x_train_avg_w2v,y_train_binary,x_test_avg_w2v,y_test_binary)


# In[ ]:


metric(L1_model,x_train_avg_w2v,y_train,x_test_avg_w2v,y_test)


# # Apply Logistic regression on TFIDF_W2V

# In[207]:


x_train_tfidf_w2v= np.array(x_train_tfidf_w2v)
x_test_tfidf_w2v= np.array(x_test_tfidf_w2v)
print(x_train_tfidf_w2v.shape)
print(x_test_tfidf_w2v.shape)


# In[208]:


C = [0.01,0.1,1,10]
L1_model=LogisticRegression(penalty='l1',C=C,solver='liblinear',max_iter=500)


# In[209]:


L1_best_c=Grid_search(L1_model,x_train_tfidf_w2v,y_train)


# In[210]:


L1_model=LogisticRegression(penalty='l1',C=L1_best_c,solver='liblinear',max_iter=500)
test_data(L1_model,x_train_tfidf_w2v,y_train_binary,x_test_tfidf_w2v,y_test_binary)


# In[216]:


metric(L1_model,x_train_tfidf_w2v,y_train,x_test_tfidf_w2v,y_test)


# In[215]:


table = PrettyTable()
table.field_names = ["Vectorizer","Regularization", "Hyperameter(C=1/lamda)", "AUC"]
table.add_row(["BOW","L1",0.1,89.26])
table.add_row(["TFIDF","L1",0.1,91.09])
table.add_row(["AvgW2v","L1",10,88.6])
table.add_row(["TFIDF_AvgW2v","L1",10,49.3])
print(table)


# # Observation

# 1) From the above table we conclude that TFIDF featurization with L1 Regularization have the Highest AUC score of 91.09 %

# # Apply Support Vector Machines on BOW

# In[218]:


from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
sc = StandardScaler(with_mean=False)
x_train_bow = sc.fit_transform(x_train_bow)
x_test_bow = sc.transform(x_test_bow)
x_cv_bow=sc.transform(x_cv_bow)


# In[219]:


def Hyper_parameter(X_train,X_cv,Y_train,Y_cv):
    import warnings
    max_roc_auc=-1
    cv_scores = []
    train_scores = []
    penalties = ['l1', 'l2']
    C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    for i in C:
        for p in penalties:
            model= SGDClassifier(alpha=i, penalty=p, loss='hinge', random_state=42)
            model.fit(X_train,Y_train)
            clf = CalibratedClassifierCV(model, method="sigmoid")
            clf.fit(X_train, Y_train)
            y_score=clf.predict_proba(X_cv)[:,1]
            scores = roc_auc_score(Y_cv, y_score)
            cv_scores.append(scores)
            y_score=clf.predict_proba(X_train)[:,1]
            scores = roc_auc_score(Y_train, y_score)
            train_scores.append(scores)
            s=['0.00001+L1', '0.00001+L2', '0.0001+L1', '0.0001+L2', '0.001+L1', '0.001+L2', '0.01+L1', '0.01+L2',
             '0.1+L1', '0.1+L2', '1+L1', '1+L2', '10+L1', '10+L2', '100+L1', '100+L2','1000+L1','1000+L2','10000+L1','10000+L2']
    optimal_alpha= s[cv_scores.index(max(cv_scores))]
    alpha=[math.log(x) for x in C]#converting values of alpha into logarithm
    fig = plt.figure(figsize=(20,5))
    ax = plt.subplot()
    ax.plot(s, train_scores, label='AUC train')
    ax.plot(s, cv_scores, label='AUC CV')
    plt.title('AUC vs hyperparameter')
    plt.xlabel('alpha')
    plt.ylabel('AUC')
    plt.xticks()
    ax.legend()
    plt.show()
    print('best Cross validation score: {:.3f}'.format(max(cv_scores)))
    print('optimal alpha and penalty for which auc is maximum : ',optimal_alpha)


# In[222]:


import warnings as w
w.filterwarnings("ignore")
Hyper_parameter(x_train_bow,x_cv_bow,y_train,y_cv)


# In[223]:


best_alpha=0.1
best_penalty='l2'


# In[224]:


def test_data(x_train,y_train,x_test,y_test):
    model = SGDClassifier(loss='hinge', penalty=best_penalty, alpha=best_alpha, n_jobs=-1)
    clf = CalibratedClassifierCV(base_estimator=model, cv=None)
   
    clf.fit(x_train, y_train)
    
    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs

    train_fpr, train_tpr, thresholds = roc_curve(y_train, clf.predict_proba(x_train)[:,1])
    test_fpr, test_tpr, thresholds = roc_curve(y_test, clf.predict_proba(x_test)[:,1])

    sns.set()
    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
    plt.plot([0, 1], [0, 1], color='green', lw=1, linestyle='--')
    plt.legend()
    plt.xlabel("False_positive_rate")
    plt.ylabel("True positive_rate")    
    plt.title("ROC_Curve")
    plt.grid()
    plt.show()
    print('The AUC_score of test_data is :',auc(test_fpr, test_tpr))


# In[226]:


test_data(x_train_bow,y_train_binary,x_test_bow,y_test_binary)


# In[229]:


def metric(x_train,y_train,x_test,y_test):
    model = SGDClassifier(loss='hinge', penalty=best_penalty, alpha=best_alpha, n_jobs=-1)
    clf = CalibratedClassifierCV(base_estimator=model, cv=None)
    clf.fit(x_train, y_train)
    predict=clf.predict(x_test)
    conf_mat = confusion_matrix(y_test, predict)
    class_label = ["Negative", "Positive"]
    df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
    report=classification_report(y_test,predict)
    print(report)
    sns.set()
    sns.heatmap(df, annot = True,fmt="d")
    plt.title("Test_Confusion_Matrix")
    plt.xlabel("Predicted_Label")
    plt.ylabel("Actual_Label")
    plt.show()


# In[230]:


metric(x_train_bow,y_train,x_test_bow,y_test)


# # Apply Support Vector Machines on TF-IDF

# In[242]:


sc = StandardScaler(with_mean=False)
x_train_tfidf = sc.fit_transform(x_train_tf_idf)
x_test_tfidf = sc.transform(x_test_tf_idf)
print(y_train.shape)
print(y_test.shape)


# In[265]:


def Hyper_parameter(x_train,x_test,y_train,y_test):
    import warnings
    max_roc_auc=-1
    cv_scores = []
    train_scores = []
    penalties = ['l1', 'l2']
    C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    for i in C:
        for p in penalties:
            model= SGDClassifier(alpha=i, penalty=p, loss='hinge', random_state=42)
            model.fit(x_train,y_train)
            clf = CalibratedClassifierCV(model, method="sigmoid")
            clf.fit(x_train, y_train)
            y_score=clf.predict_proba(x_test)[:,1]
            scores = roc_auc_score(y_test, y_score)
            cv_scores.append(scores)
            y_score=clf.predict_proba(x_train)[:,1]
            scores = roc_auc_score(y_train, y_score)
            train_scores.append(scores)
            s=['0.00001+L1', '0.00001+L2', '0.0001+L1', '0.0001+L2', '0.001+L1', '0.001+L2', '0.01+L1', '0.01+L2',
             '0.1+L1', '0.1+L2', '1+L1', '1+L2', '10+L1', '10+L2', '100+L1', '100+L2','1000+L1','1000+L2','10000+L1','10000+L2']
    optimal_alpha= s[cv_scores.index(max(cv_scores))]
    alpha=[math.log(x) for x in C]#converting values of alpha into logarithm
    fig = plt.figure(figsize=(20,5))
    ax = plt.subplot()
    ax.plot(s, train_scores, label='AUC train')
    ax.plot(s, cv_scores, label='AUC CV')
    plt.title('AUC vs hyperparameter')
    plt.xlabel('alpha')
    plt.ylabel('AUC')
    plt.xticks()
    ax.legend()
    plt.show()
    print('best Cross validation score: {:.3f}'.format(max(cv_scores)))
    print('optimal alpha and penalty for which auc is maximum : ',optimal_alpha)


# In[248]:


Hyper_parameter(x_train_tf_idf,x_test_tf_idf,y_train,y_test)


# In[249]:


best_alpha=1
best_penalty='l2'


# In[250]:


test_data(x_train_tf_idf,y_train_binary,x_test_tf_idf,y_test_binary)


# In[251]:


metric(x_train_tf_idf,y_train,x_test_tf_idf,y_test)


# # Apply Support Vector Machines on Avg_W2V

# In[266]:


x_train_avg_w2v= np.array(x_train_avg_w2v)
x_test_avg_w2v= np.array(x_test_avg_w2v)
print(x_train_avg_w2v.shape)
print(x_test_avg_w2v.shape)
print(y_train.shape)
print(y_test.shape)


# In[262]:


def Hyper_parameter(x_train,x_test,y_train,y_test):
    import warnings
    max_roc_auc=-1
    cv_scores = []
    train_scores = []
    penalties = ['l1', 'l2']
    C = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    for i in C:
        for p in penalties:
            model= SGDClassifier(alpha=i, penalty=p, loss='hinge', random_state=42)
            model.fit(x_train,y_train)
            clf = CalibratedClassifierCV(model, method="sigmoid")
            clf.fit(x_train, y_train)
            y_score=clf.predict_proba(x_test)[:,1]
            scores = roc_auc_score(y_test, y_score)
            cv_scores.append(scores)
            y_score=clf.predict_proba(x_train)[:,1]
            scores = roc_auc_score(y_train, y_score)
            train_scores.append(scores)
            s=['0.00001+L1', '0.00001+L2', '0.0001+L1', '0.0001+L2', '0.001+L1', '0.001+L2', '0.01+L1', '0.01+L2',
             '0.1+L1', '0.1+L2', '1+L1', '1+L2', '10+L1', '10+L2', '100+L1', '100+L2','1000+L1','1000+L2','10000+L1','10000+L2']
    optimal_alpha= s[cv_scores.index(max(cv_scores))]
    alpha=[math.log(x) for x in C]#converting values of alpha into logarithm
    fig = plt.figure(figsize=(20,5))
    ax = plt.subplot()
    ax.plot(s, train_scores, label='AUC train')
    ax.plot(s, cv_scores, label='AUC CV')
    plt.title('AUC vs hyperparameter')
    plt.xlabel('alpha')
    plt.ylabel('AUC')
    plt.xticks()
    ax.legend()
    plt.show()
    print('best Cross validation score: {:.3f}'.format(max(cv_scores)))
    print('optimal alpha and penalty for which auc is maximum : ',optimal_alpha)


# In[272]:


Hyper_parameter(x_train_avg_w2v,x_test_avg_w2v,y_train,y_test)


# In[284]:


best_alpha=0.5
best_penalty='l2'


# In[285]:


test_data(x_train_avg_w2v,y_train_binary,x_test_avg_w2v,y_test_binary)


# In[286]:


metric(x_train_avg_w2v,y_train,x_test_avg_w2v,y_test)


# # Apply Support Vector Machines on Tf-idf_W2V

# In[292]:


Hyper_parameter(x_train_tfidf_w2v,x_test_tfidf_w2v,y_train,y_test)


# In[296]:


best_alpha=0.5
best_penalty='l1'


# In[297]:


test_data(x_train_tfidf_w2v,y_train_binary,x_test_tfidf_w2v,y_test_binary)


# In[298]:


metric(x_train_tfidf_w2v,y_train,x_test_tfidf_w2v,y_test)


# In[300]:


table = PrettyTable()
table.field_names = ["Vectorizer","Regularization", "Hyperameter(C)", "AUC_Score"]
table.add_row(["BOW","L2",0.1,92.9])
table.add_row(["TFIDF","L2",1,92.03])
table.add_row(["AvgW2v","L2",0.5,51.1])
table.add_row(["TFIDF_AvgW2v","L1",0.5,50.0])
print(table)


# # Observation:

# 1) From the above table we conclude that BOW featurization with L2 and best C is 0.1 Regularization have the Highest AUC score i.e 92.9
# 2) Bag of words and TF-IDF are performing better fr this model than Avg_W2V or TF_IDF_W2V

# # Apply Decision Trees on BOW

# In[301]:


from sklearn.tree import DecisionTreeClassifier
def Grid_search(X_train,Y_train):
    Depths=[4,6, 8, 9,10,50,100,500]
    min_split= [10,20,30,40,50,100,500]
    param_grid = {'max_depth': Depths, 'min_samples_split': min_split}
    
    clf = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'roc_auc', cv=3 , n_jobs = -1, pre_dispatch=2,return_train_score=True)
    clf.fit(X_train, Y_train)

   
    print("\n**********AUC Score for CV data **********\n")
    print("\nOptimal depth:", clf.best_estimator_.max_depth)
    print("\nOptimal split:", clf.best_estimator_.min_samples_split)
    print("\nBest Score:", clf.best_score_)

    sns.set()
    df_gridsearch = pd.DataFrame(clf.cv_results_)
    max_scores = df_gridsearch.groupby(['param_max_depth','param_min_samples_split']).max()
    max_scores = max_scores.unstack()[['mean_test_score', 'mean_train_score']]
    sns.heatmap(max_scores.mean_test_score, annot=True, fmt='.4g')
    plt.show()


# In[302]:


warnings.filterwarnings("ignore")
Grid_search(x_train_bow,y_train)


# In[303]:


depth=50
split=500


# In[305]:


def test_data(x_train,y_train,x_test,y_test):
     
    model=DecisionTreeClassifier(max_depth=depth, min_samples_split =split, class_weight='balanced') 
   
    model.fit(x_train, y_train)
    
    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs

    train_fpr, train_tpr, thresholds = roc_curve(y_train, model.predict_proba(x_train)[:,1])
    test_fpr, test_tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])

    sns.set()
    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
    plt.plot([0, 1], [0, 1], color='green', lw=1, linestyle='--')
    plt.legend()
    plt.xlabel("False_positive_rate")
    plt.ylabel("True positive_rate")    
    plt.title("ROC_Curve")
    plt.grid()
    plt.show()
    print('The AUC_score of test_data is :',auc(test_fpr, test_tpr))


# In[307]:


test_data(x_train_bow,y_train_binary,x_test_bow,y_test_binary)


# In[380]:


def metric(x_train,y_train,x_test,y_test):
    
    model=DecisionTreeClassifier(max_depth=depth, min_samples_split =split, class_weight='balanced')
    
    model.fit(x_train, y_train)
    predict=model.predict(x_test)

    conf_mat = confusion_matrix(y_test, predict)
    class_label = ["Negative", "Positive"]
    df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
    
    report=classification_report(y_test,predict)
    print(report)
    
    sns.set()
    sns.heatmap(df, annot = True,fmt="d")
    plt.title("Test_Confusion_Matrix")
    plt.xlabel("Predicted_Label")
    plt.ylabel("Actual_Label")
    plt.show()


# In[311]:


metric(x_train_bow,y_train,x_test_bow,y_test)


# ### Apply Decision Trees on TF-IDF

# In[312]:


warnings.filterwarnings("ignore")
Grid_search(x_train_tf_idf,y_train)


# In[313]:


depth=10
split=500


# In[314]:


test_data(x_train_tf_idf,y_train_binary,x_test_tf_idf,y_test_binary)


# In[315]:


metric(x_train_tf_idf,y_train,x_test_tf_idf,y_test)


# ### Apply Decision Trees on Avg-W2V

# In[317]:


warnings.filterwarnings("ignore")
Grid_search(x_train_avg_w2v,y_train)


# In[318]:


depth=500
split=500


# In[319]:


test_data(x_train_avg_w2v,y_train_binary,x_test_avg_w2v,y_test_binary)


# In[321]:


metric(x_train_avg_w2v,y_train,x_test_avg_w2v,y_test)


# ### Applying Decision Trees on TF-IDF_W2V

# In[325]:


warnings.filterwarnings("ignore")
Grid_search(x_train_tfidf_w2v,y_train)


# In[326]:


depth=500
split=40


# In[327]:


test_data(x_train_tfidf_w2v,y_train_binary,x_test_tfidf_w2v,y_test_binary)


# In[381]:


metric(x_train_tfidf_w2v,y_train,x_test_tfidf_w2v,y_test)


# In[330]:


table = PrettyTable()
table.field_names = ["Vectorizer","Optimal Min_split", "Optimal Depth", "AUC_Score"]
table.add_row(["BOW",50,500,75.75])
table.add_row(["TFIDF",10,500,72.49])
table.add_row(["AvgW2v",500,500,47.81])
table.add_row(["TFIDF_AvgW2v",40,500,50.88])
print(table)


# ### Observation:

# 1) From the above table we conclude that Bag of Words with a optimal Depth of 50 and optimal min_split of 500 have the Highest AUC score i.e 75.75 %

# # Apply Randomforest on BOW

# In[331]:


from sklearn.ensemble import RandomForestClassifier
def Grid_search(model,X_train,Y_train):
    estimators = [50,100,200,300,400,500]
    Depths = [10,20,30,40,50,60]

    param_grid = {'max_depth': Depths, 'n_estimators': estimators}
    
    clf = GridSearchCV(model, param_grid, scoring = 'roc_auc', cv=3 , n_jobs = -1, pre_dispatch=2,return_train_score=True)
    clf.fit(X_train, Y_train)

   
    print("\n**********AUC Score for CV data **********\n")
    print("\nOptimal depth:", clf.best_estimator_.max_depth)
    print("\nOptimal estimators:", clf.best_estimator_.n_estimators)
    print("\nBest Score:", clf.best_score_)

    sns.set()
    df_gridsearch = pd.DataFrame(clf.cv_results_)
    max_scores = df_gridsearch.groupby(['param_max_depth','param_n_estimators']).max()
    max_scores = max_scores.unstack()[['mean_test_score', 'mean_train_score']]
    sns.heatmap(max_scores.mean_test_score, annot=True, fmt='.4g')
    plt.show()


# In[332]:


warnings.filterwarnings("ignore")
model=RandomForestClassifier()
Grid_search(model,x_train_bow,y_train)


# In[333]:


depth=20
estimators=400


# In[334]:


def test_data(model,x_train,y_train,x_test,y_test):
     
   
    model.fit(x_train, y_train)
    
    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs

    train_fpr, train_tpr, thresholds = roc_curve(y_train, model.predict_proba(x_train)[:,1])
    test_fpr, test_tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])

    sns.set()
    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
    plt.plot([0, 1], [0, 1], color='green', lw=1, linestyle='--')
    plt.legend()
    plt.xlabel("False_positive_rate")
    plt.ylabel("True positive_rate")    
    plt.title("ROC_Curve")
    plt.grid()
    plt.show()
    print('The AUC_score of test_data is :',auc(test_fpr, test_tpr))


# In[343]:


model=RandomForestClassifier(max_depth=depth, n_estimators=estimators,class_weight='balanced') 
test_data(model,x_train_bow,y_train_binary,x_test_bow,y_test_binary)


# In[339]:


def metric(model,x_train,y_train,x_test,y_test):
    
    model.fit(x_train, y_train)
    predict=model.predict(x_test)

    conf_mat = confusion_matrix(y_test, predict)
    class_label = ["Negative", "Positive"]
    df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
    
    report=classification_report(y_test,predict)
    print(report)
    
    sns.set()
    sns.heatmap(df, annot = True,fmt="d")
    plt.title("Test_Confusion_Matrix")
    plt.xlabel("Predicted_Label")
    plt.ylabel("Actual_Label")
    plt.show()


# In[340]:


model=RandomForestClassifier(max_depth=depth, n_estimators=estimators,class_weight='balanced') 
metric(model,x_train_bow,y_train,x_test_bow,y_test)


# # Apply Randomforest on TF-IDF

# In[341]:


warnings.filterwarnings("ignore")
model=RandomForestClassifier()
Grid_search(model,x_train_tf_idf,y_train)


# In[342]:


depth=30
estimators=400


# In[344]:


def test_data(model,x_train,y_train,x_test,y_test):
     
   
    model.fit(x_train, y_train)
    
    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs

    train_fpr, train_tpr, thresholds = roc_curve(y_train, model.predict_proba(x_train)[:,1])
    test_fpr, test_tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])

    sns.set()
    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
    plt.plot([0, 1], [0, 1], color='green', lw=1, linestyle='--')
    plt.legend()
    plt.xlabel("False_positive_rate")
    plt.ylabel("True positive_rate")    
    plt.title("ROC_Curve")
    plt.grid()
    plt.show()
    print('The AUC_score of test_data is :',auc(test_fpr, test_tpr))


# In[345]:


model=RandomForestClassifier(max_depth=depth, n_estimators=estimators,class_weight='balanced') 
test_data(model,x_train_tf_idf,y_train_binary,x_test_tf_idf,y_test_binary)


# In[346]:


model=RandomForestClassifier(max_depth=depth, n_estimators=estimators,class_weight='balanced') 
metric(model,x_train_tf_idf,y_train,x_test_tf_idf,y_test)


# # Apply RandomForest on Avg-W2V

# In[348]:


warnings.filterwarnings("ignore")
model=RandomForestClassifier()
Grid_search(model,x_train_avg_w2v,y_train)


# In[354]:


depth=60
estimators=100


# In[355]:


def test_data(model,x_train,y_train,x_test,y_test):
     
   
    model.fit(x_train, y_train)
    
    # roc_auc_score(y_true, y_score) the 2nd parameter should be probability estimates of the positive class
    # not the predicted outputs

    train_fpr, train_tpr, thresholds = roc_curve(y_train, model.predict_proba(x_train)[:,1])
    test_fpr, test_tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])

    sns.set()
    plt.plot(train_fpr, train_tpr, label="train AUC ="+str(auc(train_fpr, train_tpr)))
    plt.plot(test_fpr, test_tpr, label="test AUC ="+str(auc(test_fpr, test_tpr)))
    plt.plot([0, 1], [0, 1], color='green', lw=1, linestyle='--')
    plt.legend()
    plt.xlabel("False_positive_rate")
    plt.ylabel("True positive_rate")    
    plt.title("ROC_Curve")
    plt.grid()
    plt.show()
    print('The AUC_score of test_data is :',auc(test_fpr, test_tpr))


# In[358]:


model=RandomForestClassifier(max_depth=depth, n_estimators=estimators,class_weight='balanced') 
test_data(model,x_train_avg_w2v,y_train_binary,x_test_avg_w2v,y_test_binary)


# In[359]:


model=RandomForestClassifier(max_depth=depth, n_estimators=estimators,class_weight='balanced') 
metric(model,x_train_avg_w2v,y_train,x_test_avg_w2v,y_test)


# # Applying RandomForest on TF_IDF_W2V

# In[360]:


warnings.filterwarnings("ignore")
model=RandomForestClassifier()
Grid_search(model,x_train_tfidf_w2v,y_train)


# In[363]:


depth=60
estimators=50


# In[364]:


model=RandomForestClassifier(max_depth=depth, n_estimators=estimators,class_weight='balanced') 
test_data(model,x_train_tfidf_w2v,y_train_binary,x_test_tfidf_w2v,y_test_binary)


# In[365]:


model=RandomForestClassifier(max_depth=depth, n_estimators=estimators,class_weight='balanced') 
metric(model,x_train_tfidf_w2v,y_train,x_test_tfidf_w2v,y_test)


# In[367]:


table = PrettyTable()
table.field_names = ["Model","Vectorizer","Optimal Depth", "Optimal n_estimator", "AUC_Score"]
table.add_row(["Random Forest","BOW",20,400,88.66])
table.add_row(["Random Forest","TFIDF",30,400,90.40])
table.add_row(["Random Forest","AvgW2v",60,100,51.35])
table.add_row(["Random Forest","TFIDF_AvgW2v",60,50,49.4])
print(table)


# # Observation:

# 1) From the above table we conclude that TFIDF in Random Forest with a optimal Depth of 30 and optimal estimator of 400 have the Highest AUC score i.e 90.40 %

# In[ ]:





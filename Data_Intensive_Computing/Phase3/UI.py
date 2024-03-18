import streamlit as st
import pandas as pd

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import pickle
import joblib
from sklearn.preprocessing import LabelEncoder
st.set_option('deprecation.showPyplotGlobalUse', False)

def removehtml(review_text):
    clean_html = re.compile('<.*?>')
    cleantext = re.sub(clean_html, ' ', review_text)
    return cleantext
def removepunc(review_text):
    cleaned_punc = re.sub(r'[?|!|\'|"|#]',r'',review_text)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned_punc)
    return  cleaned

def load_additional_data(file_path):
    # Implement loading logic for additional data
    # This is a placeholder and should be replaced with your actual loading code
    return pd.read_csv(file_path,index_col=None)
def main():
    st.title("Dataset Input App")

    # Upload dataset
    uploaded_file = st.file_uploader("Choose the data CSV file", type=["csv"],key="file_uploader_main")

    # Upload label
    scores = st.file_uploader("Choose the label CSV file", type=["csv"],key="file_uploader_additional")


    if uploaded_file is not None:
        st.subheader("Preview of the uploaded dataset:")
        raw_data = pd.read_csv(uploaded_file, index_col=None)
        st.write(raw_data)

    # Display additional data if provided
    if scores is not None:
        st.subheader("Preview of the additional dataset:")
        df_scores = load_additional_data(scores)
        st.write(df_scores.head())
        
        # Handle duplicate columns by renaming them
        df_scores.columns = [f"{col}_scores" for col in df_scores.columns]
        
        # Concatenate along columns (axis=1) based on a common column
        combined_df = pd.concat([raw_data, df_scores], axis=1)
        vectorizer_options = ["Bag of Words", "TFIDF_Vectorizer"]
        # Dropdowns for model and vectorizer selection
        model_options = {"K-NearestNeibhors": " KNN", "NaiveBayes": "NaiveBayes", "Logistic Regression": "Logistic","Suport Vector Machines":"SVM",
        "Decision Trees":"DecisionTree","Random Forest":"RandomForest"}
        selected_model = st.selectbox("Select a model:", list(model_options.keys()))

        vectorizer_options = {"Bag of Words": "BOW.pkl", "TFIDF_Vectorizer": "TF_IDF.pkl"}
        selected_vectorizer = st.selectbox("Select a vectorizer:", list(vectorizer_options.keys()))


        #DataPre-processing
        filtered_data = combined_df[combined_df['Score_scores']!=3]
        warnings.filterwarnings('ignore')
        def partition(x):
            if x<3:
                return 'negative'
            else:
                return 'positive'
        dummy_data=combined_df['Score_scores']
        review_column_data=dummy_data.map(partition)
        filtered_data['Score_scores']=review_column_data

        from nltk.corpus import stopwords
        nltk.download('stopwords')
        stopwords_list = stopwords.words('english')
        stopwords = set(stopwords_list)
        sn = nltk.stem.SnowballStemmer('english')
        a=0
        string1=' '
        final_string=[]
        all_positive_words=[]
        all_negative_words=[] 
        s=''
        for sentence in filtered_data['Text'].values:
            filtered_sentence=[]
            sent=removehtml(sentence) # remove HTMl tags
            for word in sent.split():
                for cleaned_words in removepunc(word).split(): #remove puncuations
                    if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                        if(cleaned_words.lower() not in stopwords): #Removing stopwords
                            sen=(sn.stem(cleaned_words.lower())).encode('utf8') #perform stemming and encoding
                            filtered_sentence.append(sen)
                        else:
                            continue
                    else:
                        continue 
            string1 = b" ".join(filtered_sentence) 
            
            final_string.append(string1)
            a+=1
        filtered_data['CleanedText']=final_string  
        filtered_data['CleanedText']=filtered_data['CleanedText'].str.decode("utf-8")

        y_test = filtered_data['Score_scores'].astype(str)

        pickle_text = model_options[selected_model] +"_"+ vectorizer_options[selected_vectorizer]
        st.subheader("Preprocessed text:")
        st.write(filtered_data['CleanedText'].head())
        # Initialize predict variable

        if st.button("Predict"):
            # Perform prediction
            if "KNN_BOW" in pickle_text:
                file_path = '/Users/mohithsainattam/BOW.pkl'
                loaded_model = joblib.load(file_path,"rb")
                x_test_bow = loaded_model.transform(filtered_data['CleanedText'])
                file_path = '/Users/mohithsainattam/KNN_BOW.pkl'
                loaded_model = joblib.load(file_path,"rb")
                predict=loaded_model.predict(x_test_bow)
                conf_mat = confusion_matrix(y_test, predict)
                class_label = ["Negative", "Positive"]
                df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
                report=classification_report(y_test,predict)
                st.text(report)
                st.write(predict)
                sns.set()
                sns.heatmap(df, annot = True,fmt="d")
                plt.title("Test_Confusion_Matrix")
                plt.xlabel("Predicted_Label")
                plt.ylabel("Actual_Label")
                st.pyplot()

                classifier=KNeighborsClassifier(n_neighbors=49,algorithm='brute')
                label_encoder = LabelEncoder()
                y_test_binary = label_encoder.fit_transform(y_test)
                classifier.fit(x_test_bow,y_test)
                
                # Assuming 'classifier' is your trained model
                predicted_probabilities = classifier.predict_proba(x_test_bow)[:, 1]

                # Assuming 'y_test' is your true labels
                fpr, tpr, _ = roc_curve(y_test_binary, predicted_probabilities)
                roc_auc = auc(fpr, tpr)

                # Plot ROC Curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                st.pyplot()
            if "KNN_TF_IDF" in pickle_text:
                file_path = '/Users/mohithsainattam/TF_IDF.pkl'
                loaded_model = joblib.load(file_path,"rb")
                x_test_tfidf = loaded_model.transform(filtered_data['CleanedText'])
                file_path = '/Users/mohithsainattam/KNN_TF_IDF.pkl'
                loaded_model = joblib.load(file_path,"rb")
                predict=loaded_model.predict(x_test_tfidf)
                conf_mat = confusion_matrix(y_test, predict)
                class_label = ["Negative", "Positive"]
                df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
                report=classification_report(y_test,predict)
                st.text(report)
                st.write(predict)
                sns.set()
                sns.heatmap(df, annot = True,fmt="d")
                plt.title("Test_Confusion_Matrix")
                plt.xlabel("Predicted_Label")
                plt.ylabel("Actual_Label")
                st.pyplot()

                classifier=KNeighborsClassifier(n_neighbors=49,algorithm='brute')
                label_encoder = LabelEncoder()
                y_test_binary = label_encoder.fit_transform(y_test)
                classifier.fit(x_test_tfidf,y_test)
                
                # Assuming 'classifier' is your trained model
                predicted_probabilities = classifier.predict_proba(x_test_tfidf)[:, 1]

                # Assuming 'y_test' is your true labels
                fpr, tpr, _ = roc_curve(y_test_binary, predicted_probabilities)
                roc_auc = auc(fpr, tpr)

                # Plot ROC Curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                st.pyplot()
            if "NaiveBayes_BOW" in pickle_text:
                file_path = '/Users/mohithsainattam/BOW.pkl'
                loaded_model = joblib.load(file_path,"rb")
                x_test_bow = loaded_model.transform(filtered_data['CleanedText'])
                file_path = '/Users/mohithsainattam/NaiveBayes_BOW.pkl'
                loaded_model = joblib.load(file_path,"rb")
                predict=loaded_model.predict(x_test_bow)
                conf_mat = confusion_matrix(y_test, predict)
                class_label = ["Negative", "Positive"]
                df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
                report=classification_report(y_test,predict)
                st.text(report)
                st.write(predict)
                sns.set()
                sns.heatmap(df, annot = True,fmt="d")
                plt.title("Test_Confusion_Matrix")
                plt.xlabel("Predicted_Label")
                plt.ylabel("Actual_Label")
                st.pyplot()
                classifier = MultinomialNB()
                #classifier=KNeighborsClassifier(n_neighbors=49,algorithm='brute')
                label_encoder = LabelEncoder()
                y_test_binary = label_encoder.fit_transform(y_test)
                classifier.fit(x_test_bow,y_test)
                
                # Assuming 'classifier' is your trained model
                predicted_probabilities = classifier.predict_proba(x_test_bow)[:, 1]

                # Assuming 'y_test' is your true labels
                fpr, tpr, _ = roc_curve(y_test_binary, predicted_probabilities)
                roc_auc = auc(fpr, tpr)

                # Plot ROC Curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                st.pyplot()
            if "NaiveBayes_TF_IDF" in pickle_text:
                file_path = '/Users/mohithsainattam/TF_IDF.pkl'
                loaded_model = joblib.load(file_path,"rb")
                x_test_tfidf = loaded_model.transform(filtered_data['CleanedText'])
                file_path = '/Users/mohithsainattam/NaiveBayes_tfidf.pkl'
                loaded_model = joblib.load(file_path,"rb")
                predict=loaded_model.predict(x_test_tfidf)
                conf_mat = confusion_matrix(y_test, predict)
                class_label = ["Negative", "Positive"]
                df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
                report=classification_report(y_test,predict)
                st.text(report)
                st.write(predict)
                sns.set()
                sns.heatmap(df, annot = True,fmt="d")
                plt.title("Test_Confusion_Matrix")
                plt.xlabel("Predicted_Label")
                plt.ylabel("Actual_Label")
                st.pyplot()
                classifier = MultinomialNB()
                #classifier=KNeighborsClassifier(n_neighbors=49,algorithm='brute')
                label_encoder = LabelEncoder()
                y_test_binary = label_encoder.fit_transform(y_test)
                classifier.fit(x_test_tfidf,y_test)
                
                # Assuming 'classifier' is your trained model
                predicted_probabilities = classifier.predict_proba(x_test_tfidf)[:, 1]

                # Assuming 'y_test' is your true labels
                fpr, tpr, _ = roc_curve(y_test_binary, predicted_probabilities)
                roc_auc = auc(fpr, tpr)

                # Plot ROC Curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                st.pyplot()
            if "Logistic_BOW" in pickle_text:
                file_path = '/Users/mohithsainattam/BOW.pkl'
                loaded_model = joblib.load(file_path,"rb")
                x_test_bow = loaded_model.transform(filtered_data['CleanedText'])
                file_path = '/Users/mohithsainattam/Logistic_BOW.pkl'
                loaded_model = joblib.load(file_path,"rb")
                predict=loaded_model.predict(x_test_bow)
                conf_mat = confusion_matrix(y_test, predict)
                class_label = ["Negative", "Positive"]
                df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
                report=classification_report(y_test,predict)
                st.text(report)
                st.write(predict)
                sns.set()
                sns.heatmap(df, annot = True,fmt="d")
                plt.title("Test_Confusion_Matrix")
                plt.xlabel("Predicted_Label")
                plt.ylabel("Actual_Label")
                st.pyplot()

                classifier = LogisticRegression(penalty='l1',C=0.1,solver='liblinear',max_iter=500)
                label_encoder = LabelEncoder()
                y_test_binary = label_encoder.fit_transform(y_test)
                classifier.fit(x_test_bow,y_test)
                
                # Assuming 'classifier' is your trained model
                predicted_probabilities = classifier.predict_proba(x_test_bow)[:, 1]

                # Assuming 'y_test' is your true labels
                fpr, tpr, _ = roc_curve(y_test_binary, predicted_probabilities)
                roc_auc = auc(fpr, tpr)

                # Plot ROC Curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                st.pyplot()
            if "Logistic_TF_IDF" in pickle_text:
                file_path = '/Users/mohithsainattam/TF_IDF.pkl'
                loaded_model = joblib.load(file_path,"rb")
                x_test_tfidf = loaded_model.transform(filtered_data['CleanedText'])
                file_path = '/Users/mohithsainattam/Logistic_TFIDF.pkl'
                loaded_model = joblib.load(file_path,"rb")
                predict=loaded_model.predict(x_test_tfidf)
                conf_mat = confusion_matrix(y_test, predict)
                class_label = ["Negative", "Positive"]
                df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
                report=classification_report(y_test,predict)
                st.text(report)
                st.write(predict)
                sns.set()
                sns.heatmap(df, annot = True,fmt="d")
                plt.title("Test_Confusion_Matrix")
                plt.xlabel("Predicted_Label")
                plt.ylabel("Actual_Label")
                st.pyplot()

                classifier = LogisticRegression(penalty='l1',C=0.1,solver='liblinear',max_iter=500)
                label_encoder = LabelEncoder()
                y_test_binary = label_encoder.fit_transform(y_test)
                classifier.fit(x_test_tfidf,y_test)
                
                # Assuming 'classifier' is your trained model
                predicted_probabilities = classifier.predict_proba(x_test_tfidf)[:, 1]

                # Assuming 'y_test' is your true labels
                fpr, tpr, _ = roc_curve(y_test_binary, predicted_probabilities)
                roc_auc = auc(fpr, tpr)

                # Plot ROC Curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                st.pyplot()
            if "SVM_BOW" in pickle_text:
                file_path = '/Users/mohithsainattam/BOW.pkl'
                loaded_model = joblib.load(file_path,"rb")
                x_test_bow = loaded_model.transform(filtered_data['CleanedText'])
                file_path = '/Users/mohithsainattam/SVM_BOW.pkl'
                loaded_model = joblib.load(file_path,"rb")
                predict=loaded_model.predict(x_test_bow)
                conf_mat = confusion_matrix(y_test, predict)
                class_label = ["Negative", "Positive"]
                df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
                report=classification_report(y_test,predict)
                st.text(report)
                st.write(predict)
                sns.set()
                sns.heatmap(df, annot = True,fmt="d")
                plt.title("Test_Confusion_Matrix")
                plt.xlabel("Predicted_Label")
                plt.ylabel("Actual_Label")
                st.pyplot()

                classifier = SGDClassifier(penalty='l2',alpha=0.1,loss='log', random_state=42) 
                label_encoder = LabelEncoder()
                y_test_binary = label_encoder.fit_transform(y_test)
                classifier.fit(x_test_bow,y_test)
                
                # Assuming 'classifier' is your trained model
                predicted_probabilities = classifier.predict_proba(x_test_bow)[:, 1]

                # Assuming 'y_test' is your true labels
                fpr, tpr, _ = roc_curve(y_test_binary, predicted_probabilities)
                roc_auc = auc(fpr, tpr)

                # Plot ROC Curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                st.pyplot()
            if "SVM_TF_IDF" in pickle_text:
                file_path = '/Users/mohithsainattam/TF_IDF.pkl'
                loaded_model = joblib.load(file_path,"rb")
                x_test_tfidf = loaded_model.transform(filtered_data['CleanedText'])
                file_path = '/Users/mohithsainattam/SVM_TFIDF.pkl'
                loaded_model = joblib.load(file_path,"rb")
                predict=loaded_model.predict(x_test_tfidf)
                conf_mat = confusion_matrix(y_test, predict)
                class_label = ["Negative", "Positive"]
                df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
                report=classification_report(y_test,predict)
                st.text(report)
                st.write(predict)
                sns.set()
                sns.heatmap(df, annot = True,fmt="d")
                plt.title("Test_Confusion_Matrix")
                plt.xlabel("Predicted_Label")
                plt.ylabel("Actual_Label")
                st.pyplot()

                classifier = SGDClassifier(penalty='l2',alpha=0.1,loss='log', random_state=42) 
                label_encoder = LabelEncoder()
                y_test_binary = label_encoder.fit_transform(y_test)
                classifier.fit(x_test_tfidf,y_test)
                
                # Assuming 'classifier' is your trained model
                predicted_probabilities = classifier.predict_proba(x_test_tfidf)[:, 1]

                # Assuming 'y_test' is your true labels
                fpr, tpr, _ = roc_curve(y_test_binary, predicted_probabilities)
                roc_auc = auc(fpr, tpr)

                # Plot ROC Curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                st.pyplot()
            if "DecisionTree_BOW" in pickle_text:
                file_path = '/Users/mohithsainattam/BOW.pkl'
                loaded_model = joblib.load(file_path,"rb")
                x_test_bow = loaded_model.transform(filtered_data['CleanedText'])
                file_path = '/Users/mohithsainattam/DecisionTree_BOW.pkl'
                loaded_model = joblib.load(file_path,"rb")
                predict=loaded_model.predict(x_test_bow)
                conf_mat = confusion_matrix(y_test, predict)
                class_label = ["Negative", "Positive"]
                df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
                report=classification_report(y_test,predict)
                st.text(report)
                st.write(predict)
                sns.set()
                sns.heatmap(df, annot = True,fmt="d")
                plt.title("Test_Confusion_Matrix")
                plt.xlabel("Predicted_Label")
                plt.ylabel("Actual_Label")
                st.pyplot()

                classifier = DecisionTreeClassifier(random_state=42)
                label_encoder = LabelEncoder()
                y_test_binary = label_encoder.fit_transform(y_test)
                classifier.fit(x_test_bow,y_test)
                
                # Assuming 'classifier' is your trained model
                predicted_probabilities = classifier.predict_proba(x_test_bow)[:, 1]

                # Assuming 'y_test' is your true labels
                fpr, tpr, _ = roc_curve(y_test_binary, predicted_probabilities)
                roc_auc = auc(fpr, tpr)

                # Plot ROC Curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                st.pyplot()
            if "DecisionTree_TF_IDF" in pickle_text:
                file_path = '/Users/mohithsainattam/TF_IDF.pkl'
                loaded_model = joblib.load(file_path,"rb")
                x_test_tfidf = loaded_model.transform(filtered_data['CleanedText'])
                file_path = '/Users/mohithsainattam/DecisionTree_TFIDF.pkl'
                loaded_model = joblib.load(file_path,"rb")
                predict=loaded_model.predict(x_test_tfidf)
                conf_mat = confusion_matrix(y_test, predict)
                class_label = ["Negative", "Positive"]
                df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
                report=classification_report(y_test,predict)
                st.text(report)
                st.write(predict)
                sns.set()
                sns.heatmap(df, annot = True,fmt="d")
                plt.title("Test_Confusion_Matrix")
                plt.xlabel("Predicted_Label")
                plt.ylabel("Actual_Label")
                st.pyplot()

                classifier = DecisionTreeClassifier(random_state=42)
                label_encoder = LabelEncoder()
                y_test_binary = label_encoder.fit_transform(y_test)
                classifier.fit(x_test_tfidf,y_test)
                
                # Assuming 'classifier' is your trained model
                predicted_probabilities = classifier.predict_proba(x_test_tfidf)[:, 1]

                # Assuming 'y_test' is your true labels
                fpr, tpr, _ = roc_curve(y_test_binary, predicted_probabilities)
                roc_auc = auc(fpr, tpr)

                # Plot ROC Curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                st.pyplot()
            if "RandomForest_BOW" in pickle_text:
                file_path = '/Users/mohithsainattam/BOW.pkl'
                loaded_model = joblib.load(file_path,"rb")
                x_test_bow = loaded_model.transform(filtered_data['CleanedText'])
                file_path = '/Users/mohithsainattam/RandomForest_BOW.pkl'
                loaded_model = joblib.load(file_path,"rb")
                predict=loaded_model.predict(x_test_bow)
                conf_mat = confusion_matrix(y_test, predict)
                class_label = ["Negative", "Positive"]
                df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
                report=classification_report(y_test,predict)
                st.text(report)
                st.write(predict)
                sns.set()
                sns.heatmap(df, annot = True,fmt="d")
                plt.title("Test_Confusion_Matrix")
                plt.xlabel("Predicted_Label")
                plt.ylabel("Actual_Label")
                st.pyplot()

                classifier = RandomForestClassifier(depth=20,estimators=400,class_weight='balanced',random_state=42)
                label_encoder = LabelEncoder()
                y_test_binary = label_encoder.fit_transform(y_test)
                classifier.fit(x_test_bow,y_test)
                
                # Assuming 'classifier' is your trained model
                predicted_probabilities = classifier.predict_proba(x_test_bow)[:, 1]

                # Assuming 'y_test' is your true labels
                fpr, tpr, _ = roc_curve(y_test_binary, predicted_probabilities)
                roc_auc = auc(fpr, tpr)

                # Plot ROC Curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                st.pyplot()
            if "RandomForest_TF_IDF" in pickle_text:
                file_path = '/Users/mohithsainattam/TF_IDF.pkl'
                loaded_model = joblib.load(file_path,"rb")
                x_test_tfidf = loaded_model.transform(filtered_data['CleanedText'])
                file_path = '/Users/mohithsainattam/RandomForest_TFIDF.pkl'
                loaded_model = joblib.load(file_path,"rb")
                predict=loaded_model.predict(x_test_tfidf)
                conf_mat = confusion_matrix(y_test, predict)
                class_label = ["Negative", "Positive"]
                df = pd.DataFrame(conf_mat, index = class_label, columns = class_label)
                report=classification_report(y_test,predict)
                st.text(report)
                st.write(predict)
                sns.set()
                sns.heatmap(df, annot = True,fmt="d")
                plt.title("Test_Confusion_Matrix")
                plt.xlabel("Predicted_Label")
                plt.ylabel("Actual_Label")
                st.pyplot()

                classifier = RandomForestClassifier(depth=30,estimators=400,class_weight='balanced',random_state=42)
                label_encoder = LabelEncoder()
                y_test_binary = label_encoder.fit_transform(y_test)
                classifier.fit(x_test_tfidf,y_test)
                
                # Assuming 'classifier' is your trained model
                predicted_probabilities = classifier.predict_proba(x_test_tfidf)[:, 1]

                # Assuming 'y_test' is your true labels
                fpr, tpr, _ = roc_curve(y_test_binary, predicted_probabilities)
                roc_auc = auc(fpr, tpr)

                # Plot ROC Curve
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc='lower right')
                st.pyplot()
            

            

            

if __name__ == "__main__":
    main()

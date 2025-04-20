import numpy as np # linear algebra
import pandas as pd 
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import joblib
df=pd.read_csv('/Users/soundariyavijayakumar/resume_project/UpdatedResumeDataSet.csv')
df.head()
sns.countplot(data=df, x='Category')
plt.xticks(rotation=90)
plt.show()

def clean_html(text):
    clean=re.compile('<.*>')
    return re.sub(clean,'',text)
def convert_lower(text):
    return text.lower()
df['Resume']=df['Resume'].apply(convert_lower)
df['Resume']=df['Resume'].apply(clean_html)
def remove_special(text):
    x=''
    for i in text:
        if i.isalnum():
            x=x+i
        else:
            x=x+' '
    return x
df['Resume']=df['Resume'].apply(remove_special)
import nltk
from nltk.corpus import stopwords
def remove_stopwords(text):
    x=[]
    for i in text.split():
        if i not in stopwords.words('english'):
            x.append(i)
    
    y=x[:]
    x.clear()
    return y
df['Resume']=df['Resume'].apply(remove_stopwords)
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
y=[]
def stem_words(text):
    for i in text:
        y.append(ps.stem(i))
    
    z=y[:]
    y.clear()
    return z
df['Resume']=df['Resume'].apply(stem_words)
def join_back(list_input):
    return " ".join(list_input)
df['Resume']=df['Resume'].apply(join_back)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1000)
X=cv.fit_transform(df['Resume']).toarray()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Category'] = label_encoder.fit_transform(df['Category'])
y=df.iloc[:,0]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

model = GaussianNB()
model.fit(X_train,y_train)
joblib.dump(cv, 'vectorizer.pkl')
joblib.dump(model, 'job_category_model.pkl')
joblib.dump(label_encoder, "label_encoder.pkl")
y_pred1=model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred1)
print("Accuracy:", accuracy)


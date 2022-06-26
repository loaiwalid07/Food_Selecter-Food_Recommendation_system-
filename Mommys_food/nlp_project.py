# -*- coding: utf-8 -*-
"""NLP_Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1je53g5Vh_EcZSd5F2qjq82rR7gRGCcCT
"""

!pip uninstall scikit-learn -y

!pip install -U scikit-learn

import random 
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords,gutenberg
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import ShuffleSplit,KFold,cross_val_score,train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud
from sklearn.decomposition import LatentDirichletAllocation
#___________________________________________ Setup ___________________________________
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.corpus.gutenberg.fileids()
print("Done")

"""# Clean Data"""

import gdown
url_1 = 'https://drive.google.com/uc?id=1Vo8vNZlkAV5ckxPcU3OHMkJ2mVbZJfUS'
output_1 = 'mob_scrape.csv'
gdown.download(url_1,output=output_1,quiet=False)

df=pd.read_csv("mob_scrape.csv",sep=",")
index = df.index
condition = df["Dish"] == "SERIOUSLY PASTA SIX WAYS"
indices = index[condition]
indices_list = indices.tolist()
print(indices_list)
df = df.drop([169,84])
df.to_excel("output.xlsx")
df["Ingredients"]
df = df.reset_index()

def remove_s(s):
    result =s.replace("covers absolutely everything assume beforehand", "")
  
    return result

def remove_numbers(s):
    result = ''.join([i for i in s if not i.isdigit()])
    return result

def remove_single(s):
    result = ' '.join([w for w in s.split() if len(w) > 1])
    return result

stop = stopwords.words('english')
other_stop = [
    'kg',
    'grams',
    'cans',
    'of',
    'bags',
    'tbsp',
    'tsp',
    'pinch',
    'pint',
    'pack',
    'packs',
    'fresh',
    'large',
    'small',
    'gram',
    'tins',
    'knob',
    'chopped',
    'smoked',
    'extra',
    'very',
    'lazy',
    'crushed',
    'ml',
    'litre',
    'bunch',
    'dried',
    'teaspoon',
    'dried',
    'tablespoon',
    'tablespoons',
    'bag',
    'half',
    'red',
    'green',
    'yellow',
    'pink',
    'orange',
    'total cost  covers absolutely everything',
    'tbsps'
    "mob's",
    'creamy',
    'curshed',
    'flat',
    'jar',
    'tin',
    'pitted',
    'bottle',
    'seasoning',
    'only',
    "n'",
    'salt',
     'olive', 
     'pepper',
    'toasted',
    'giant',
    'sliced',
    'diced',
    'minced',
    'knob',
    'knobs',
    'method',
    'frozen',
    'skinless',
    'boneless',
    'loaf',
    'grated',
    'jar',
    'nice',
    'good',
    'total',
    'deli',
     'kitchen',
    'cost',
]
stop.extend(other_stop)

symbols = ['.', '£', '-', '!', '(', ')', ':', ',',"'",']','[']

def remove_stop(s, fuzzy=False):
    words = s.split(' ')
    if fuzzy:
        out = []
        for i in words:
            include = True
            for st in stop:
                if fuzz.ratio(words, stop) > 50:
                    include = False
            if include:
                out.append(i)
        return ' '.join(out)
    elif not fuzzy:
        result = ' '.join([i for i in words if i not in stop])
        return result


def remove_symbol(s):
    for i in symbols:
        s = s.replace(i, "")
    return s.strip()

def extract_entities(s):
    l_words = str(s).lower().split('\n')

    l_words = [remove_numbers(s) for s in l_words]
    l_words = [remove_single(s) for s in l_words]
    l_words = [remove_stop(s) for s in l_words]
    l_words = [remove_symbol(s) for s in l_words]
    l_words = [remove_s(s) for s in l_words]
    l_words = [i for i in l_words if i != '']
    return l_words

ing_entities = df["Ingredients"].map(extract_entities).to_list()
title_entities = df["Dish"].map(extract_entities).to_list()

for i in range(len(df["Ingredients"])):
  df["Ingredients"]=df["Ingredients"].replace(to_replace =df["Ingredients"][i], value =str(ing_entities[i])[1:-1])

df_new=df[['Dish', 'Ingredients',"URL"]].copy()
df_new.rename(columns={'Dish': 'name', 'Ingredients': 'ingredients'}, inplace=True)
df_new.to_csv("newdata.CSV",index=False)

"""# Clustering


"""

import random 
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords,gutenberg
from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import ShuffleSplit,KFold,cross_val_score,train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wordcloud
from sklearn.cluster import k_means
from sklearn.cluster import KMeans
import matplotlib.pyplot as mtp    
from sklearn.metrics import cohen_kappa_score
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics import homogeneity_score
from tqdm import tqdm
#___________________________________________ Setup ___________________________________
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.corpus.gutenberg.fileids()
print("Done")

X= df_new["ingredients"]

# BOW transformer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X).todense()
#print(X_train_counts)
print("Data dimensions before PCA with BOW:",X_train_counts.shape)
from sklearn.decomposition import PCA
#perform PCA to plot 
pca = PCA(n_components=2)
pca.fit(X_train_counts)
BOW_2Dtransformed_data=pca.transform(X_train_counts)
print("Data dimensions after PCA with BOW:",BOW_2Dtransformed_data.shape)
#print(BOW_2Dtransformed_data)

#TF-IDF transform
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tfidf = tf_transformer.transform(X_train_counts).todense()
print("Data dimensions before PCA with TFIDF and BOW:",X_train_tfidf.shape)

pca = PCA(n_components=2)
pca.fit(X_train_tfidf)
tfidf_2Dtransformed_data=pca.transform(X_train_tfidf)
print("Data dimensions after PCA with TFIDF and BOW:",tfidf_2Dtransformed_data.shape)
#print(tfidf_2Dtransformed_data)

wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_train_counts)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 21), wcss)
plt.title('Elbow Method K-mean Model with BOW')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=50, random_state=0)
pred_y = kmeans.fit_predict(X_train_counts)
#plt.scatter(BOW[:,0], BOW[:,1])
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()

y_predict= kmeans.fit_predict(X_train_counts)  
print(y_predict)

mtp.scatter(BOW_2Dtransformed_data[y_predict == 0, 0], BOW_2Dtransformed_data[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster  
mtp.scatter(BOW_2Dtransformed_data[y_predict == 1, 0], BOW_2Dtransformed_data[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster  
mtp.scatter(BOW_2Dtransformed_data[y_predict == 2, 0], BOW_2Dtransformed_data[y_predict == 2, 1], s = 100, c = 'red', label = 'Cluster 3') #for third cluster  
mtp.scatter(BOW_2Dtransformed_data[y_predict == 3, 0], BOW_2Dtransformed_data[y_predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') #for fourth cluster  
mtp.scatter(BOW_2Dtransformed_data[y_predict == 4, 0], BOW_2Dtransformed_data[y_predict == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') #for fifth cluster  
mtp.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')   
mtp.title('K-mean Model with BOW')  
  
mtp.legend()  
mtp.show()

wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_train_tfidf)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 21), wcss)
plt.title('Elbow Method K-mean Model with TF-iDF')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=50, random_state=0)
pred_y = kmeans.fit_predict(X_train_tfidf)
#plt.scatter(X_train_tfidf[:,0], X_train_tfidf[:,1])
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
plt.show()

y_predict= kmeans.fit_predict(X_train_tfidf)  
print(y_predict)

mtp.scatter(tfidf_2Dtransformed_data[y_predict == 0, 0], tfidf_2Dtransformed_data[y_predict == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster  
mtp.scatter(tfidf_2Dtransformed_data[y_predict == 1, 0], tfidf_2Dtransformed_data[y_predict == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster  
mtp.scatter(tfidf_2Dtransformed_data[y_predict == 2, 0], tfidf_2Dtransformed_data[y_predict == 2, 1], s = 100, c = 'red', label = 'Cluster 3') #for third cluster  
mtp.scatter(tfidf_2Dtransformed_data[y_predict == 3, 0], tfidf_2Dtransformed_data[y_predict == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') #for fourth cluster  
mtp.scatter(tfidf_2Dtransformed_data[y_predict == 4, 0], tfidf_2Dtransformed_data[y_predict == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') #for fifth cluster  
mtp.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')   
mtp.title('K-mean Model with TF-iDF')  
  
mtp.legend()  
mtp.show()

clusters=pd.DataFrame({"ing":X,"class":y_predict})
clusters.to_csv("class.csv")

clas_0=[]
clas_1=[]
clas_2=[]
clas_3=[]
clas_4=[]
num=[]
for i in range(len(clusters["class"])):
  if clusters.iloc[i][1] ==0:
    clas_0.append(clusters.iloc[i][0])
    num.append(0)
  elif clusters.iloc[i][1] ==1:
    clas_1.append(clusters.iloc[i][0])
    num.append(1)
  elif clusters.iloc[i][1] ==2:
    clas_2.append(clusters.iloc[i][0])
    num.append(2)
  elif clusters.iloc[i][1] ==3:
    clas_3.append(clusters.iloc[i][0])
    num.append(3)
  else:
    clas_4.append(clusters.iloc[i][0])
    num.append(4)

clusters.sort_values("class")

"""**Top 10 of Items for each Class**"""

def most_frq(y):
  corpus = clusters[clusters["class"]==y]["ing"]
  corpus= corpus.reset_index()
  l=""
  for i in range(len(corpus)):
    l= l + corpus["ing"][i]
  for i in symbols:
    l=l.replace(i,'')
  from collections import Counter

  data_set = l  
  # split() returns list of all the words in the string
  split_it = data_set.split()
    
  # Pass the split_it list to instance of Counter class.
  Counter = Counter(split_it)
    
  # most_common() produces k frequently encountered
  # input values and their respective counts.
  most_occur = Counter.most_common(10)
  print(most_occur)
  item = []
  values = []
  for i,j in most_occur:
    item.append(i)
    values.append(j)
  fig = plt.figure(figsize = (10, 5))
  
  # creating the bar plot
  plt.bar(item, values, color ='maroon',
          width = 0.4)
  
  plt.xlabel("Items")
  plt.ylabel("Frequnecy")
  plt.title(" Top 10 of Items")
  plt.show()

most_frq(0)
most_frq(1)
most_frq(2)
most_frq(3)
most_frq(4)

cuisine=[]
for i in clusters["class"]:
  if i == 0:
    cuisine.append("Chinese")
  elif i == 1:
    cuisine.append("French")
  elif i == 2:
    cuisine.append("Indian")
  elif i == 3:
    cuisine.append("Mexican")
  else:
    cuisine.append("Italian")
final_clusters=pd.DataFrame(([xx,yy,zz] for xx, yy,zz in zip(clusters["ing"],clusters["class"],cuisine)),columns = ["Ingredients","Class","Cuisine"])
final_clusters.to_csv("final_clusters.csv")

"""**Cuisine Comparison**"""

fig = plt.figure(figsize = (10, 5))
  
# creating the bar plot
plt.bar(["Chinese","French","Indian","Mexican","Italian"], [len(clas_0),len(clas_1),len(clas_2),len(clas_3),len(clas_4)], color ='maroon',
        width = 0.4)

plt.xlabel("Cuisine")
plt.ylabel("Count")
plt.title(" Cuisine Comparison")
plt.show()

"""# Classification"""



"""**Data Split** to ------------> train 70 % , test 30 % 

"""

X, y = final_clusters["Ingredients"],final_clusters["Cuisine"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X.shape, y.shape

"""#Classification and Evaluation for KNN"""

#_____________KNN___________________
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import LatentDirichletAllocation
 #   ('lda',LatentDirichletAllocation(n_components=5,random_state=0)),
#plt.title("MultinomialNB Confusion matrix withBOW , LDA and tfidf")

#_____________KNN with BOW ___________________

text_clf_knn = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', KNeighborsClassifier(n_neighbors=4),)
])
text_clf_knn.fit(X_train,y_train)
test_knn =X_test
predicted_knn = text_clf_knn.predict(test_knn)
print("Accuracy for KNN with BOW  : -> ",np.mean(predicted_knn == y_test))
print(metrics.classification_report(y_test, predicted_knn))
plot_confusion_matrix(text_clf_knn, X_test, y_test,xticks_rotation="vertical")
plt.title("KNN Confusion matrix with BOW")
plt.show()

print("\n[The Accuracy by using cross validation]\n")
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
kf = KFold(n_splits=10,shuffle=True,random_state=0)
scores_knn = cross_val_score(text_clf_knn, X, y, cv=kf)
num=0
for i in scores_knn:
  num+=1
  print("Accuracy at Iteration with BOW: %r ->"%(num),i)
print("Total accuracy: %0.2f (+/- %0.2f)" % (scores_knn.mean(), scores_knn.std() * 2))


#_____________KNN with BOW and LDA___________________

text_clf_knn = Pipeline([
    ('vect', CountVectorizer()),
    ('lda',LatentDirichletAllocation(n_components=5,random_state=0)),
    ('clf', KNeighborsClassifier(n_neighbors=4),)
])
text_clf_knn.fit(X_train,y_train)
test_knn =X_test
predicted_knn = text_clf_knn.predict(test_knn)
print("Accuracy for KNN with BOW  and LDA : -> ",np.mean(predicted_knn == y_test))
print(metrics.classification_report(y_test, predicted_knn))
plot_confusion_matrix(text_clf_knn, X_test, y_test,xticks_rotation="vertical")
plt.title("KNN Confusion matrix with BOW  and LDA")
plt.show()

print("\n[The Accuracy by using cross validation]\n")
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
kf = KFold(n_splits=10,shuffle=True,random_state=0)
scores_knn = cross_val_score(text_clf_knn, X, y, cv=kf)
num=0
for i in scores_knn:
  num+=1
  print("Accuracy at Iteration with BOW  and LDA: %r ->"%(num),i)
print("Total accuracy: %0.2f (+/- %0.2f)" % (scores_knn.mean(), scores_knn.std() * 2))
#_____________KNN with BOW ,TfIdf and LDA___________________

text_clf_knn = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('lda',LatentDirichletAllocation(n_components=5,random_state=0)),
    ('clf', KNeighborsClassifier(n_neighbors=4),)
])
text_clf_knn.fit(X_train,y_train)
test_knn =X_test
predicted_knn = text_clf_knn.predict(test_knn)
print("Accuracy for KNN with BOW, TfIdf   and LDA : -> ",np.mean(predicted_knn == y_test))
print(metrics.classification_report(y_test, predicted_knn))
plot_confusion_matrix(text_clf_knn, X_test, y_test,xticks_rotation="vertical")
plt.title("KNN Confusion matrix with BOW, TfIdf and LDA")
plt.show()

print("\n[The Accuracy by using cross validation]\n")
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
kf = KFold(n_splits=10,shuffle=True,random_state=0)
scores_knn = cross_val_score(text_clf_knn, X, y, cv=kf)
num=0
for i in scores_knn:
  num+=1
  print("Accuracy at Iteration with BOW ,TfIdf  and LDA: %r ->"%(num),i)
print("Total accuracy: %0.2f (+/- %0.2f)" % (scores_knn.mean(), scores_knn.std() * 2))



#_____________KNN with BOW and Tfidf___________________

text_clf_knn = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', KNeighborsClassifier(n_neighbors=4),)
])
text_clf_knn.fit(X_train,y_train)
test_knn =X_test
predicted_knn = text_clf_knn.predict(test_knn)
print("Accuracy for KNN with BOW  and tfidf : -> ",np.mean(predicted_knn == y_test))
print(metrics.classification_report(y_test, predicted_knn))
plot_confusion_matrix(text_clf_knn, X_test, y_test,xticks_rotation="vertical")
plt.title("KNN Confusion matrix with BOW  and tfidf")
plt.show()

print("\n[The Accuracy by using cross validation]\n")
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
kf = KFold(n_splits=10,shuffle=True,random_state=0)
scores_knn = cross_val_score(text_clf_knn, X, y, cv=kf)
num=0
for i in scores_knn:
  num+=1
  print("Accuracy at Iteration with BOW  and tfidf: %r ->"%(num),i)
print("Total accuracy: %0.2f (+/- %0.2f)" % (scores_knn.mean(), scores_knn.std() * 2))

"""#Classification and Evaluation for SVM with BOW,TFidf"""

#___________________________ SVM __________________________
#  SVM with LDA
from sklearn.decomposition import LatentDirichletAllocation
    # ('lda',LatentDirichletAllocation(n_components=5,random_state=0)),
#_________________________ SVM with BOW______________________________
text_clf_svm = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', svm.SVC(kernel='linear', C=1),)
])
text_clf_svm.fit(X_train,y_train)
test_svm =X_test
predicted_svm = text_clf_svm.predict(test_svm)
print("Accuracy for SVM with BOW :-> ",np.mean(predicted_svm == y_test))
print(metrics.classification_report(y_test, predicted_svm))
plot_confusion_matrix(text_clf_svm, X_test, y_test,xticks_rotation="vertical")
plt.title("SVM Confusion matrix with BOW")
plt.show()

print("\n[The Accuracy by using cross validation]\n")
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
kf = KFold(n_splits=10,shuffle=True,random_state=0)
scores_svm = cross_val_score(text_clf_svm, X, y, cv=kf)
for i in scores_svm:
  print("Accuracy at Iteration: %r ->"%(np.where(scores_svm == i)),i)
print("Total accuracy of SVM with BOW: %0.2f (+/- %0.2f)" % (scores_svm.mean(), scores_svm.std() * 2))


#_________________________ SVM with BOW and LDA______________________________

text_clf_svm = Pipeline([
        ('vect', CountVectorizer()),
('lda',LatentDirichletAllocation(n_components=5,random_state=0)),
    ('clf', svm.SVC(kernel='linear', C=1),)
])
text_clf_svm.fit(X_train,y_train)
test_svm =X_test
predicted_svm = text_clf_svm.predict(test_svm)
print("Accuracy for  SVM with BOW and LDA :-> ",np.mean(predicted_svm == y_test))
print(metrics.classification_report(y_test, predicted_svm))
plot_confusion_matrix(text_clf_svm, X_test, y_test,xticks_rotation="vertical")
plt.title("SVM Confusion matrix BOW and LDA ")
plt.show()

print("\n[The Accuracy by using cross validation]\n")
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
kf = KFold(n_splits=10,shuffle=True,random_state=0)
scores_svm = cross_val_score(text_clf_svm, X, y, cv=kf)
for i in scores_svm:
  print("Accuracy at Iteration: %r ->"%(np.where(scores_svm == i)),i)
print("Total accuracyof SVM with BOW and LDA: %0.2f (+/- %0.2f)" % (scores_svm.mean(), scores_svm.std() * 2))

#_________________________ SVM with BOW, tf-idf and LDA______________________________
text_clf_svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('lda',LatentDirichletAllocation(n_components=5,random_state=0)),
    ('clf', svm.SVC(kernel='linear', C=1),)
])
text_clf_svm.fit(X_train,y_train)
test_svm =X_test
predicted_svm = text_clf_svm.predict(test_svm)
print("Accuracy for SVM with BOW, TF-idf and LDA :-> ",np.mean(predicted_svm == y_test))
print(metrics.classification_report(y_test, predicted_svm))
plot_confusion_matrix(text_clf_svm, X_test, y_test,xticks_rotation="vertical")
plt.title("SVM Confusion matrix BOW, TF-idf and LDA ")
plt.show()

print("\n[The Accuracy by using cross validation]\n")
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
kf = KFold(n_splits=10,shuffle=True,random_state=0)
scores_svm = cross_val_score(text_clf_svm, X, y, cv=kf)
for i in scores_svm:
  print("Accuracy at Iteration: %r ->"%(np.where(scores_svm == i)),i)
print("Total accuracy of SVM with BOW, TF-idf and LDA: %0.2f (+/- %0.2f)" % (scores_svm.mean(), scores_svm.std() * 2))


#_________________________ SVM with BOW and tf-idf______________________________

text_clf_svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', svm.SVC(kernel='linear', C=1),)
])
text_clf_svm.fit(X_train,y_train)
test_svm =X_test
predicted_svm = text_clf_svm.predict(test_svm)
print("Accuracy for SVM with BOW and TF-idf :-> ",np.mean(predicted_svm == y_test))
print(metrics.classification_report(y_test, predicted_svm))
plot_confusion_matrix(text_clf_svm, X_test, y_test,xticks_rotation="vertical")
plt.title("SVM Confusion matrix BOW and TF-idf ")
plt.show()

print("\n[The Accuracy by using cross validation]\n")
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
kf = KFold(n_splits=10,shuffle=True,random_state=0)
scores_svm = cross_val_score(text_clf_svm, X, y, cv=kf)
for i in scores_svm:
  print("Accuracy at Iteration : %r ->"%(np.where(scores_svm == i)),i)
print("Total accuracy of SVM with BOW and TF-idf: %0.2f (+/- %0.2f)" % (scores_svm.mean(), scores_svm.std() * 2))

"""#Classification and Evaluation for Decision Tree"""

#==================================================================================
#_____________________________ Decision Tree __________________________
#==================================================================================

#______________________________ Decision Tree with BOW + LDA ____________________________

text_clf_DT = Pipeline([
    ('vect', CountVectorizer()),
    ('lda',LatentDirichletAllocation(n_components=5,random_state=0)),
    ('clf', DecisionTreeClassifier(),)
])
text_clf_DT.fit(X_train,y_train)
test_DT =X_test
predicted_DT = text_clf_DT.predict(test_DT)
print("Accuracy for DecisionTree with BOW and LDA:-> ",np.mean(predicted_DT == y_test))
print(metrics.classification_report(y_test, predicted_DT))
plot_confusion_matrix(text_clf_DT, X_test, y_test,xticks_rotation="vertical")
plt.title("Decision Tree Confusion matrix BOW and LDA")
plt.show()

print("\n[The Accuracy by using cross validation]\n")
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
kf = KFold(n_splits=10,shuffle=True,random_state=0)
scores_DT = cross_val_score(text_clf_DT, X, y, cv=kf)
for i in scores_DT:
  print("Accuracy at Iteration: %r ->"%(np.where(scores_DT == i)),i)
print("Total accuracy of Decision Tree with BOW and LDA : %0.2f (+/- %0.2f)" % (scores_DT.mean(), scores_DT.std() * 2))



#______________________________ Decision Tree with BOW + LDA + TF-idf ____________________________

text_clf_DT = Pipeline([
    ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
    ('lda',LatentDirichletAllocation(n_components=5,random_state=0)),
    ('clf', DecisionTreeClassifier(),)
])
text_clf_DT.fit(X_train,y_train)
test_DT =X_test
predicted_DT = text_clf_DT.predict(test_DT)
print("Accuracy for DecisionTree with BOW, LDA and TF-idf:-> ",np.mean(predicted_DT == y_test))
print(metrics.classification_report(y_test, predicted_DT))
plot_confusion_matrix(text_clf_DT, X_test, y_test,xticks_rotation="vertical")
plt.title("Decision Tree Confusion matrix BOW, LDA and TF-idf")
plt.show()

print("\n[The Accuracy by using cross validation]\n")
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
kf = KFold(n_splits=10,shuffle=True,random_state=0)
scores_DT = cross_val_score(text_clf_DT, X, y, cv=kf)
for i in scores_DT:
  print("Accuracy at Iteration: %r ->"%(np.where(scores_DT == i)),i)
print("Total accuracy of Decision Tree with BOW, LDA and TF-idf : %0.2f (+/- %0.2f)" % (scores_DT.mean(), scores_DT.std() * 2))


#______________________________ Decision Tree with BOW + TF-idf ____________________________

text_clf_DT = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', DecisionTreeClassifier(),)
])
text_clf_DT.fit(X_train,y_train)
test_DT =X_test
predicted_DT = text_clf_DT.predict(test_DT)
print("Accuracy for DecisionTree with BOW and TF-idf:-> ",np.mean(predicted_DT == y_test))
print(metrics.classification_report(y_test, predicted_DT))
plot_confusion_matrix(text_clf_DT, X_test, y_test,xticks_rotation="vertical")
plt.title("Decision Tree Confusion matrix BOW and TF-idf")
plt.show()

print("\n[The Accuracy by using cross validation]\n")
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
kf = KFold(n_splits=10,shuffle=True,random_state=0)
scores_DT = cross_val_score(text_clf_DT, X, y, cv=kf)
for i in scores_DT:
  print("Accuracy at Iteration: %r ->"%(np.where(scores_DT == i)),i)
print("Total accuracy of Decision Tree with BOW and TF-idf : %0.2f (+/- %0.2f)" % (scores_DT.mean(), scores_DT.std() * 2))

#______________________________ Decision Tree with BOW ____________________________

text_clf_DT = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', DecisionTreeClassifier(),)
])
text_clf_DT.fit(X_train,y_train)
test_DT =X_test
predicted_DT = text_clf_DT.predict(test_DT)
print("Accuracy for DecisionTree with BOW :-> ",np.mean(predicted_DT == y_test))
print(metrics.classification_report(y_test, predicted_DT))
plot_confusion_matrix(text_clf_DT, X_test, y_test,xticks_rotation="vertical")
plt.title("Decision Tree Confusion matrix BOW")
plt.show()

print("\n[The Accuracy by using cross validation]\n")
cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
kf = KFold(n_splits=10,shuffle=True,random_state=0)
scores_DT = cross_val_score(text_clf_DT, X, y, cv=kf)
for i in scores_DT:
  print("Accuracy at Iteration: %r ->"%(np.where(scores_DT == i)),i)
print("Total accuracy of Decision Tree with BOW: %0.2f (+/- %0.2f)" % (scores_DT.mean(), scores_DT.std() * 2))

"""# Selec Best Model And Save It"""

import joblib 
filename = 'finalized_model.sav'
joblib.dump(text_clf_svm, filename)

"""# Save Finial Data"""

Final_Data=pd.DataFrame(([X,Y,xx,yy,zz] for X,Y,xx, yy,zz in zip(df["Dish"],clusters["ing"],df["URL"],clusters["class"],cuisine)),columns = ["Dish","Ingredients","URL","Class","Cuisine"])
Final_Data.to_csv("Final_Data.csv")
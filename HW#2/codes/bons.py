import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , f1_score,precision_score,recall_score,classification_report,confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
df = pd.read_csv('C:\\Users\\dehgh\\OneDrive\\Desktop\\HW-parttern\\HW2\\agaricus-lepiota.data.txt',names=['class',
'cap-shape',
'cap-surface',
'cap-color',
'bruises',
'odor',
'gill-attachment',
'gill-spacing',
'gill-size',
'gill-color',
'stalk-shape',
'stalk-root',
'stalk-surface-above-ring',
'stalk-surface-below-ring',
'stalk-color-above-ring',
'stalk-color-below-ring',
'veil-type',
'veil-color',
'ring-number',
'ring-type',
'spore-print-color',
'population',
'habitat'
])
df.drop('veil-type',axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder
col=df.columns
df[col]=df[col].apply(LabelEncoder().fit_transform)
x= df.drop('class',axis=1)
y =df['class']
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20)
class NaiveBayesClassifier():
    def calc_prior(self, features, target):
      
    
        self.prior = (features.groupby(target).apply(lambda x: len(x)) / self.rows).to_numpy()

        return self.prior
    
    def calc_statistics(self, features, target):
        
        self.mean = features.groupby(target).apply(np.mean).to_numpy()
        self.var = features.groupby(target).apply(np.var).to_numpy()
              
        return self.mean, self.var
    
    def gaussian_density(self, class_idx, x):     
       
        mean = self.mean[class_idx]
        var = self.var[class_idx]

        numerator = np.exp((-1/2)*((x-mean)**2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        prob = numerator / denominator
        return prob
    
    def calc_posterior(self, x):
        posteriors = []

        for i in range(self.count):
            prior = np.log(self.prior[i]) 
            conditional = np.sum(np.log(self.gaussian_density(i, x)))
            posterior = prior + conditional
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
     

    def fit(self, features, target):
        self.classes = np.unique(target)
        self.count = len(self.classes)
        self.feature_nums = features.shape[1]
        self.rows = features.shape[0]
        
        self.calc_statistics(features, target)
        self.calc_prior(features, target)
        
    def predict(self, features):
        preds = [self.calc_posterior(f) for f in features.to_numpy()]
        return preds

x = NaiveBayesClassifier()


x.fit(xtrain, ytrain)
predictions = x.predict(xtest)
pred=x.predict(xtrain)
dd=classification_report(ytrain,pred)
print(f'train metircs: {dd}')
cc=confusion_matrix(ytrain,pred)
print(cc)
dm=classification_report(ytest,predictions)
print(f'test metircs: {dm}')
cm=confusion_matrix(ytest,predictions)
print(cm)
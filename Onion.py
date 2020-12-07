import pandas
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

#Read csv
csv_data = pandas.read_csv('onion-or-not2.csv')

#Titles after preprocessing
preprocessed_titles = []
#Converts title to words,stems and removes stopwords and appends to preprocessed_titles
for title in csv_data['text']:
    
    #Removes upper letter and splits title to words
    words = title.lower().split()

    #Stem 
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(i) for i in words]

    #Stopword removal
    stops = set(stopwords.words('english'))
    preprocessed_words = [w for w in words if not w in stops]

    # Merge words to title
    preprocessed_titles.append(" ".join(preprocessed_words))

#Tf-idf vector
word_vector = TfidfVectorizer()

#Fit and transform vectorizer
vector_titles = word_vector.fit_transform(preprocessed_titles)
vector_titles = pandas.DataFrame(vector_titles.toarray(), columns=word_vector.get_feature_names())
vector_labels = csv_data['label']

#Split dataset to train and test(75-25)
titles_train, titles_test, label_train, label_test = train_test_split(vector_titles, vector_labels, test_size=0.25)

#Create and train classifier
classifier = MLPClassifier()
classifier.fit(titles_train, label_train)

#Predict
predict = classifier.predict(titles_test)
print(classification_report(label_test, predict))
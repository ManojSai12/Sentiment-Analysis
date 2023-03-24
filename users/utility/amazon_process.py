import nltk
import numpy as np
from sklearn import metrics
np.set_printoptions(precision=2, linewidth=80)


def assign_sentiment(rating):
    if float(rating) >= 4:
        # return "Positive"
        return 1
    else:
        # return "Negative"
        return 0


def start_classification_analysis():
    from django.conf import settings
    import pandas as pd
    path = settings.MEDIA_ROOT + "\\" + "amazon_reviews.csv"
    df = pd.read_csv(path, dtype='unicode')
    df.columns = ['id', 'name', 'asins', 'brand', 'categories', 'keys', 'manufacturer', 'date', 'dateAdded', 'dateSeen',
                  'isPurchase', 'isRecommended', 'reviewsId', 'numHelpful', 'rating', 'sourceURLs', 'reviewText',
                  'reviewTitle', 'city', 'userProvince', 'username']
    df = df.head(500)
    # print(df.reviewsId.dtype)
    df.reviewsId.fillna(0.0)
    # print(df.head(10))
    df = df.drop('keys', 1)
    df.drop('sourceURLs', 1, inplace=True)
    df.drop(['dateAdded', 'dateSeen'], 1, inplace=True)
    # df.head()
    df.isPurchase.fillna(False, inplace=True)
    df.reviewsId.fillna("", inplace=True)
    df.city.fillna("", inplace=True)
    df.userProvince.fillna("", inplace=True)
    # df.head()
    df.dropna(subset=['name'], inplace=True)
    df.describe(include='object')
    df['name'].value_counts()
    sdf = df[['rating', 'reviewText']]
    sdf.head(2)
    sdf['sentiment'] = sdf['rating'].apply(assign_sentiment)
    sdf.drop('rating', inplace=True, axis=1)

    # sdf = sdf['sentiment'].replace(0, 'Negative', inplace=True)
    # sdf = sdf['sentiment'].replace(1, 'Positive', inplace=True)

    print(sdf.head(10))
    import re
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpus = []
    for i in range(0, 500):
        review = re.sub('[^a-zA-Z]', ' ', sdf['reviewText'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    ## Creates the bag of wrods Model
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = sdf.iloc[:, 1].values

    # Spliting the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    # Feature Scaling
    # from sklearn.preprocessing import StandardScaler
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    # Fitting Naive baues to the training Set
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predicting the Test set Results
    y_pred = classifier.predict(X_test)

    # Make The Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    nb_accuracy = metrics.accuracy_score(y_test, y_pred)
    nb_precision = metrics.precision_score(y_test, y_pred)
    nb_recall = metrics.recall_score(y_test, y_pred)
    nb_f1_score = metrics.f1_score(y_test, y_pred)
    nb_dict = {
        'nb_accuracy': nb_accuracy,
        'nb_precision': nb_precision,
        'nb_recall': nb_recall,
        'nb_f1_score': nb_f1_score
    }
    # Fitting Logistic Regression
    from sklearn.linear_model import LogisticRegression
    lg_model = LogisticRegression()
    lg_model.fit(X_train, y_train)
    # Predicting the Test set Results
    y_pred = lg_model.predict(X_test)
    # Make The Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    lg_accuracy = metrics.accuracy_score(y_test, y_pred)
    lg_precision = metrics.precision_score(y_test, y_pred)
    lg_recall = metrics.recall_score(y_test, y_pred)
    lg_f1_score = metrics.f1_score(y_test, y_pred)
    lg_dict = {
        'lg_accuracy': lg_accuracy,
        'lg_precision': lg_precision,
        'lg_recall': lg_recall,
        'lg_f1_score': lg_f1_score
    }

    # Fitting RandomForest
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    # Predicting the Test set Results
    y_pred = rf_model.predict(X_test)
    # Make The Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    rf_accuracy = metrics.accuracy_score(y_test, y_pred)
    rf_precision = metrics.precision_score(y_test, y_pred)
    rf_recall = metrics.recall_score(y_test, y_pred)
    rf_f1_score = metrics.f1_score(y_test, y_pred)
    rf_dict = {
        'rf_accuracy': rf_accuracy,
        'rf_precision': rf_precision,
        'rf_recall': rf_recall,
        'rf_f1_score': rf_f1_score
    }

    from sklearn.naive_bayes import BernoulliNB
    bnb_model = BernoulliNB()
    bnb_model.fit(X_train, y_train)
    # Predicting the Test set Results
    y_pred = bnb_model.predict(X_test)
    # Make The Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    bnb_accuracy = metrics.accuracy_score(y_test, y_pred)
    bnb_precision = metrics.precision_score(y_test, y_pred)
    bnb_recall = metrics.recall_score(y_test, y_pred)
    bnb_f1_score = metrics.f1_score(y_test, y_pred)
    bnb_dict = {
        'bnb_accuracy': bnb_accuracy,
        'bnb_precision': bnb_precision,
        'bnb_recall': bnb_recall,
        'bnb_f1_score': bnb_f1_score
    }

    result = {
        "nb": nb_dict,
        "lg": lg_dict,
        "rf": rf_dict,
        "bnb": bnb_dict
    }
    return result



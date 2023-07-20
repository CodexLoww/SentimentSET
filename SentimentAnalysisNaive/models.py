import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, make_scorer

class Models:

    def __init__(self):
        self.name = ''
        path = 'dataset/trainingdata.csv'
        df = pd.read_csv(path)
        df = df.dropna()
        self.x = df['sentences']
        self.y = df['sentiments']

    def mnb_classifier(self):
        self.name = 'MultinomialNB classifier'
        classifier = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
        return classifier

    def mnb_stemmed_classifier(self):
        self.name = 'MultinomialNB stemmed classifier'
        self.stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
        classifier = Pipeline([('vect', self.stemmed_count_vect), ('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
        return classifier

    def evaluate_model(self, model):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        # Use class weights to handle imbalanced data
        class_weights = 'balanced'
        model.named_steps['clf'].class_prior = None  # Reset any previous class priors
        model.named_steps['clf'].class_weight = class_weights

        # Cross-validation for performance evaluation
        scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1_score': make_scorer(f1_score, average='weighted')
        }
        cv_results = cross_validate(model, self.x, self.y, cv=5, scoring=scoring)

        print(f"{self.name} cross-validation results:")
        print("Accuracy:", cv_results['test_accuracy'].mean())
        print("Precision:", cv_results['test_precision'].mean())
        print("Recall:", cv_results['test_recall'].mean())
        print("F1 Score:", cv_results['test_f1_score'].mean())

        # Train and evaluate on the test set
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"\n{self.name} test set results:")
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        print("Precision: {:.2f}%".format(precision * 100))
        print("Recall: {:.2f}%".format(recall * 100))
        print("F1 Score: {:.2f}%".format(f1 * 100))
        print("Confusion Matrix:")
        for row in conf_matrix:
            print(row)

class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


if __name__ == '__main__':
    model = Models()

    # MultinomialNB classifier without stemming
    mnb_classifier_model = model.mnb_classifier()
    model.evaluate_model(mnb_classifier_model)

    # MultinomialNB classifier with stemming
    mnb_stemmed_classifier_model = model.mnb_stemmed_classifier()
    model.evaluate_model(mnb_stemmed_classifier_model)

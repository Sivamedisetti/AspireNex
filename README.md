"# AspireNex" 


#Model Details
Data Preprocessing
Lowercasing: Converting all text to lowercase.
Tokenization: Splitting the text into individual words.
Removing Special Characters: Removing punctuation and numbers.
Removing Stopwords: Eliminating common English words that do not contribute to the meaning.
Stemming: Reducing words to their root form using Porter Stemmer.
Feature Extraction
TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert the preprocessed text data into numerical features.

#Classifiers
Several classifiers were trained and evaluated:

Naive Bayes (Gaussian, Multinomial, Bernoulli)
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Evaluation Metrics
The models were evaluated using the following metrics:

#Accuracy: The ratio of correctly predicted instances to the total instances.
#Precision: The ratio of correctly predicted positive observations to the total predicted positives.
#Recall: The ratio of correctly predicted positive observations to the all observations in actual class.
#F1 Score: The weighted average of Precision and Recall.
#Results
The performance of different models was compared based on precision and F1 score. Below is the summary of the evaluation:

![Screenshot 2024-07-04 173311](https://github.com/Sivamedisetti/AspireNex-SMS-SPAM-DETECTION/assets/96729473/7e914546-75be-4310-9fd1-6b520da750d7)
# AspireNex-SMS-Spam-Detection

## Project Overview

**SMS-Spam-Detection** is a machine learning project focused on classifying SMS messages as either "spam" or "ham" (non-spam). The objective of this project is to develop and evaluate various machine learning models for effective spam detection in SMS messages. The project includes data preprocessing, feature extraction, model training, evaluation, and a summary of results to identify the most effective classification techniques.

## Model Details

### Data Preprocessing

Data preprocessing is a fundamental step in preparing the raw SMS text data for analysis. The preprocessing steps include:

- **Lowercasing:** Converting all text to lowercase to ensure uniformity.
- **Tokenization:** Splitting the text into individual words or tokens.
- **Removing Special Characters:** Eliminating punctuation, numbers, and other special characters that do not contribute to the meaning.
- **Removing Stopwords:** Filtering out common English words that do not add significant meaning (e.g., "and", "the").
- **Stemming:** Reducing words to their root form using the Porter Stemmer to standardize variations of words.

### Feature Extraction

**TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert the preprocessed text data into numerical features. TF-IDF measures the importance of words in the messages, relative to the entire dataset, creating a suitable representation for machine learning models.

### Classifiers

Several machine learning classifiers were trained and evaluated for SMS spam detection:

- **Support Vector Machine (SVM):** A robust classifier that finds the optimal hyperplane for separating spam from ham messages.
- **Naive Bayes:** Includes Gaussian, Multinomial, and Bernoulli Naive Bayes models based on probability for text classification.
- **Random Forest Classifier:** An ensemble method using multiple decision trees to improve classification accuracy.
- **Decision Tree Classifier:** A model that uses a tree-like structure to make classification decisions based on features.
- **Logistic Regression:** A model used for binary and multi-class classification tasks.
- **K-Nearest Neighbors (KNN):** A classifier that assigns a label based on the majority vote of the nearest neighbors.

### Evaluation Metrics

The models were evaluated using the following metrics:

- **Accuracy:** The ratio of correctly predicted instances to the total instances.
- **Precision:** The ratio of correctly predicted spam messages to the total predicted spam messages.
- **Recall:** The ratio of correctly predicted spam messages to all actual spam messages.
- **F1 Score:** The weighted average of Precision and Recall, balancing both metrics.

### Results

The performance of different models was compared based on accuracy, precision, recall, and F1 score. Below is a summary of the evaluation results:

| Model                     | Accuracy | Precision | Recall | F1 Score |
|---------------------------|----------|-----------|--------|----------|
| Support Vector Machine   | 97.29    | 99.00     | 78.57  | 87.61    |
| Naive Bayes               | 97.10    | 100.00    | 76.19  | 86.49    |
| Random Forest Classifier | 96.91    | 97.96     | 76.19  | 85.71    |
| Decision Tree Classifier | 94.49    | 78.51     | 75.40  | 76.92    |
| Logistic Regression      | 95.16    | 97.50     | 61.90  | 75.73    |
| K-Nearest Neighbors (KNN) | 92.26    | 100.00    | 36.51  | 53.49    |

The results show that the Support Vector Machine achieved the highest accuracy and precision, while K-Nearest Neighbors had the lowest F1 score due to lower recall performance.

## Technologies Used

- **Programming Language:** Python
- **Libraries:** scikit-learn for machine learning algorithms and model evaluation, pandas for data manipulation.

## Conclusion

The **SMS-Spam-Detection** project highlights the development of a machine learning pipeline for text classification, demonstrating skills in data preprocessing, feature extraction, model training, and evaluation. The project reveals that the Support Vector Machine model is the most effective for spam detection in SMS messages based on accuracy and precision.

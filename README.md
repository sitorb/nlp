# Sentiment Analysis of Text Data

## Project Overview

This project aims to perform sentiment analysis on a dataset of text data to classify the sentiment expressed as either positive or negative. It leverages Natural Language Processing (NLP) techniques and machine learning algorithms to achieve this goal.

## Project Structure

The project is structured as follows:

1. **Data Loading and Exploration:**
   - The code begins by loading the dataset from a CSV file using the `pandas` library.
   - It then performs exploratory data analysis (EDA) to understand the data's characteristics, such as data types, descriptive statistics, and missing values.

2. **Text Preprocessing:**
   - **Tokenization:** The text data is tokenized into individual words using the `nltk` library's `word_tokenize` function.
   - **Cleaning:** Punctuation is removed, and text is converted to lowercase using string operations.
   - **Normalization:** Lemmatization is applied to reduce words to their base forms using `WordNetLemmatizer`.
   - **Stop Word Removal:** Common words that do not contribute much to sentiment analysis are removed using `nltk`'s `stopwords` list.

3. **Feature Extraction:**
   - **Vectorization:** The preprocessed text is converted into numerical vectors using the TF-IDF (Term Frequency-Inverse Document Frequency) technique with `TfidfVectorizer` from `sklearn`. This allows the machine learning model to understand the text data.

4. **Model Building and Evaluation:**
   - **Model Selection:** A Multinomial Naive Bayes classifier is chosen for this task.
   - **Training and Testing:** The dataset is split into training and testing sets, and the model is trained on the training data.
   - **Evaluation:** The model's performance is evaluated using metrics such as precision, recall, F1-score, and a confusion matrix.

5. **Hyperparameter Tuning and Model Improvement:**
   - **Grid Search:** `GridSearchCV` is used to find the optimal value for the `alpha` hyperparameter of the Multinomial Naive Bayes model.
   - **Model Re-training:** The model is re-trained with the best hyperparameter value.
   - **Evaluation:** The improved model's performance is re-evaluated.

6. **Visualization:**
   - **Confusion Matrix:** A confusion matrix is generated and visualized using `seaborn` and `matplotlib` to understand the model's predictions.
  
   ![image](https://github.com/user-attachments/assets/6e8a3052-a77b-4112-afb3-0410de52cc35)


## Technology and Algorithms

**Technology:**

- **Python:** The primary programming language used for this project.
- **Libraries:** `pandas`, `nltk`, `sklearn`, `matplotlib`, `seaborn`.
- **Google Colab:** The development environment used for running the code.

**Algorithms:**

- **Multinomial Naive Bayes:** A probabilistic machine learning algorithm used for text classification.
- **TF-IDF (Term Frequency-Inverse Document Frequency):** A technique for converting text into numerical vectors based on word frequencies.

## Logic

The project follows a typical NLP pipeline for sentiment analysis:

1. **Data Preparation:** Cleaning, tokenization, normalization, and stop word removal.
2. **Feature Engineering:** Vectorization using TF-IDF.
3. **Model Training and Evaluation:** Building, training, and evaluating the machine learning model.
4. **Hyperparameter Tuning:** Optimizing the model's performance.
5. **Visualization:** Presenting the results using a confusion matrix.

## Conclusion

This project demonstrates a basic approach to sentiment analysis using Python and machine learning. The results can be further improved by exploring other models, advanced preprocessing techniques, and larger datasets.

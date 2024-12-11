# ML End Sem Project Evaluation: Suicidal Sentiments Identifier

## Team Members
- **Aayush Mishra** 
- **Dhruv Prakash** 
- **Himanshu Raj** 
- **Sarthak Sharma** 

---

## Abstract
Our aim is to create a machine learning model that detects suicidal tendencies in communication, typically social media posts and messages, which are in a text form. The model classifies the users through speech patterns and surrounding circumstances using a classification model, exploring methods such as Support Vector Machines, Naive Bayes, etc. The method ensures a powerful tool to save lives and intervene in the health-threatening behaviours of individuals. 


## Dataset Details
The dataset used for this project is a publicly available Reddit dataset sourced from Kaggle, containing approximately 230,000 posts from the r/SuicideWatch subreddit. These posts span a period from December 2008 to January 2021, covering a wide range of discussions related to suicidal tendencies and mental health support. Non-suicide posts are collected from r/teenagers. 

Key Characteristics:
- **Balanced Dataset**: Evenly split between suicidal and non-suicidal posts.
- **Post Length**: Average post length is 131.5 words, ranging from 1 to 15,632 words. Most posts (80%) contain fewer than 190 words.
- **Handling Slang**: Incorporated methods to preserve the context of slang words during preprocessing.

---

## Data Preprocessing
- **Lowercasing**: Standardized all text by converting to lowercase.
- **URL Removal**: Stripped URLs to eliminate noise.
- **Numerical Values**: Removed phone numbers and underscores.
- **Punctuation**: Eliminated punctuation and special characters.
- **Tokenization**: Split text into individual words for granular analysis.
- **Stopword Removal**: Removed common stopwords to focus on meaningful words.
- **Lemmatization**: Mapped words to their root forms to preserve context.
- **Handling Conjunction Words**: Used the `wordninja` Python library to split concatenated words (e.g., "helloworld" to "hello" + "world").

---

## Methodology: Feature Extraction
Techniques used for feature extraction:
1. **Bag of Words (BoW)**: Represents text based on word frequency.
2. **TF-IDF**: Enhances BoW by assigning relative importance to words across posts.

The vocabulary consists of 164,765 unique words derived from the dataset.

---

## Methodology: Model Selection
We experimented with multiple classification algorithms to identify the best-performing model:

1. **Logistic Regression**:
   - Provides clear coefficients for interpretability.
   - Accuracy: 91.75%, F1 Score: 94%.
2. **Multinomial Naive Bayes**:
   - Probabilistic handling of word frequencies.
   - Accuracy: 86.65%, F1 Score: 86%-87%.
3. **Perceptron**:
   - Iterative learning algorithm.
   - Accuracy: 87.01%, F1 Score: 87%.
4. **MLP**:
   - Parameters: `hidden_layer_sizes=(256, 16)`, `activation='relu'`, `solver='adam'`, `learning_rate_init=0.001`.
   - Accuracy: 92.21%, F1 Score: Best-performing model.
5. **Random Forest**:
   - Accuracy: 84.19%.

---

## Results and Analysis
We evaluated models based on:
- **Accuracy**: Percentage of correct predictions.
- **Recall**: Proportion of actual positives correctly identified.
- **F1 Score**: Harmonic mean of precision and recall.

### Performance Summary:
- Logistic Regression and MLP showed superior performance.
- MLP achieved the highest accuracy (92.21%) with early stopping based on validation scores.
- Random Forest and Decision Tree models underperformed compared to others.

---

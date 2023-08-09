# Help Twitter Combat Hate Speech Using NLP and Machine Learning

### Objectives
- Use NLP and ML to make a model to identify hate speech
(racist or sexist tweets) on Twitter
- Use cleaned-up tweets to build a classification model using
NLP techniques, cleanup-specific data for all tweets,
regularization, and hyperparameter tuning using stratified
k-fold and cross validation to get the best model

### Prerequisites
- Sklearn: It includes several effective methods of statistical modeling and
machine learning, such as classification, regression, clustering, and
dimensionality reduction.
- Grid search: It is a procedure that thoroughly explores a manually chosen
portion of the targeted algorithm's hyperparameter space.
- Stratified K-fold: This cross validation is an extension of the crossvalidation technique used for classification problems.

### Dataset Description

<B>Variable     - Description</b> <br>
id           - Identifier number of the comment<br>
comment_text - The text in the comment<br>
toxic        - Status of toxicity with 0 for nontoxic and 1 for toxic

### Steps
Task to Perform
1. Load the tweets file using the read_csv function from the Pandas package
2. Upload the tweets into a list for easy text cleanup and manipulation
3. Apply the following steps to clean up the tweets:
- Normalize the casing
- Use regular expressions and remove user handles that begin with @
- Use regular expressions, and remove URLs
- Use TweetTokenizer from NLTK to tokenize the tweets into individual
terms
- Remove stop words
- Remove redundant terms like amp and rt
- Remove # from the tweets while retaining the text that follows it
Task to Perform
4. Use the cleanup process to remove terms with a length of 1
5. Check the top terms in the tweets:
- First, get all tokenized terms into one list
- Use the counter, and find the 10 most common terms
6. Format the data for predictive modeling:
- Join the tokens back to form strings, which will be required for the
vectorizers
- Assign x and y
- Perform train_test_split using sklearn
Task to Perform
7. Use TF-IDF values for the terms as a feature to get into a vector space
model
- Import TF-IDF vectorizer from sklearn
- Instantiate the model with a maximum of 5000 terms in your vocabulary
- Fit and apply the vector space model on the train set
- Apply the model on the test set
8. Model building: ordinary logistic regression
- Instantiate logistic regression from sklearn with default parameters
- Fit model on the train data
- Make predictions for the train and the test sets
Task to Perform
9. Model evaluation: accuracy, recall, and f_1 score
- Report the accuracy of the train set
- Report the recall on the train set: decent, high, or low
- Get the f_1 score on the train set
10. Adjust the class imbalance, if any
- Adjust the appropriate class in the logistic regression model
11. Train the model again with the adjustment and evaluate
- Train the model on the train set
- Evaluate the predictions on the train set: accuracy, recall, and f_1 score
Task to Perform
12. Regularization and hyperparameter tuning:
- Import GridSearch and StratifiedKFold
- Choose for C and penalty parameters under the parameters grid
- Use a balanced class weight while instantiating the logistic regression
13. Find the parameters with the best recall in cross-validation
- Choose recall as the metric for scoring
- Choose a stratified four-fold cross-validation scheme
- Fit it on the train set
14. List the best parameters
Task to Perform
15. Predict and evaluate parameters using the best estimator
- Use the best estimator from the grid search to make predictions on the
test set
- Find the recall on the test set for the toxic comments
- Find the f_1 score

### Setup and Installation:
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip list
```

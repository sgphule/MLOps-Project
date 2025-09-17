import dagshub
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from config import CONFIG
import logging
logger = logging.getLogger(__name__)


# data preprocessing
# Define text preprocessing functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text


def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text


def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def normalize_text(df):
    """Normalize the text data."""
    try:
        df['review'] = df['review'].apply(lower_case)
        df['review'] = df['review'].apply(remove_stop_words)
        df['review'] = df['review'].apply(removing_numbers)
        df['review'] = df['review'].apply(removing_punctuations)
        df['review'] = df['review'].apply(removing_urls)
        df['review'] = df['review'].apply(lemmatization)
        df.head()
        return df
    except Exception as e:
        logging.info(f'Error during text normalization: {e}')
        raise


def making_sure_sentiment_is_either_positive_or_negative(df):
    x = df['sentiment'].isin(['positive','negative'])
    df = df[x]
    return df


def map_sentiment_to_numbers(df):
    df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})
    df.head()
    return df


def are_there_any_null_values(df):
    df.isnull().sum()


def set_mlflow_experiment_using_dagshub():
    mlflow.set_tracking_uri(CONFIG['mlflow_tracking_uri'])
    dagshub.init(repo_owner=CONFIG['dagshub_repo_owner'], repo_name=CONFIG['dagshub_repo_name'], mlflow=True)
    mlflow.set_experiment(CONFIG['base_experiment'])


def perform_train_test_split(df):
    vectorizer = CountVectorizer(max_features=100)
    X = vectorizer.fit_transform(df['review'])
    y = df['sentiment']
    return train_test_split(X, y, test_size=0.25, random_state=42)


def model_training(X_train, X_test, y_train, y_test):
    logging.info("Starting MLflow run...")
    with mlflow.start_run():
        start_time = time.time()
        try:
            logging.info("Logging preprocessing parameters...")
            mlflow.log_param("vectorizer", "Bag of Words")
            mlflow.log_param("num_features", 100)
            mlflow.log_param("test_size", 0.25)

            logging.info("Initializing Logistic Regression model...")
            model = LogisticRegression(max_iter=1000)  # Increase max_iter to prevent non-convergence issues

            logging.info("Fitting the model...")
            model.fit(X_train, y_train)
            logging.info("Model training complete.")

            logging.info("Logging model parameters...")
            mlflow.log_param("model", "Logistic Regression")

            logging.info("Making predictions...")
            y_pred = model.predict(X_test)

            logging.info("Calculating evaluation metrics...")
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logging.info("Logging evaluation metrics...")
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            logging.info("Saving and logging the model...")
            mlflow.sklearn.log_model(model, "model")

            # Log execution time
            end_time = time.time()
            logging.info(f"Model training and logging completed in {end_time - start_time:.2f} seconds.")

            # Print the results for verification
            logging.info(f"Accuracy: {accuracy}")
            logging.info(f"Precision: {precision}")
            logging.info(f"Recall: {recall}")
            logging.info(f"F1 Score: {f1}")

        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)


def data_preprocessing(df):
    df = normalize_text(df)
    logger.info(df['sentiment'].value_counts())
    df = making_sure_sentiment_is_either_positive_or_negative(df)
    df = map_sentiment_to_numbers(df)
    are_there_any_null_values(df)
    return perform_train_test_split(df)


def main():
    global x_train, x_test, y_train, y_test
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv('notebooks/IMDB.csv')
    df = df.sample(500)
    df.to_csv('notebooks/data.csv', index=False)

    while True:
        logger.info("1. DISPLAY FIRST 5 RECORDS IN DATA")
        logger.info("2. DISPLAY SHAPE OF DATA")
        logger.info("3. VERIFY NULL VALUE EXISTENCE IN DATA")
        logger.info("4. DISPLAY GENERAL DATA INFORMATION")
        logger.info("5. SUMMARIZE  STATISTICS")
        logger.info("6. PERFORM DATA PREPROCESSING")
        logger.info("7. SET EXPERIMENT AT MLFLOW USING DAGSHUB")
        logger.info("8. PERFORM MODEL TRAINING AFTER DATA PREPROCESSING")
        logger.info("9. EXIT APPLICATION")
        logger.info("ENTER YOUR CHOICE: ")
        choice = int(input())
        match choice:
            case 1:
                logger.info(df.head())
            case 2:
                logger.info(df.shape)
            case 3:
                logger.info(df.isnull().sum())
            case 4:
                logger.info(df.info())
            case 5:
                logger.info(df.describe())
            case 6:
                x_train, x_test, y_train, y_test= data_preprocessing(df)
            case 7:
                set_mlflow_experiment_using_dagshub()
            case 8:
                answer = input("HAVE YOU PERFORMED DATA PREPROCESSING (YES/NO)?")
                if answer == "NO" or answer == "no":
                    x_train, x_test, y_train, y_test = data_preprocessing(df)
                answer = input("HAVE YOU MLFLOW SET EXPERIMENT USING DAGSHUB (YES/NO)?")
                if answer == "NO" or answer == "no":
                    set_mlflow_experiment_using_dagshub()
                model_training(x_train, x_test, y_train, y_test)
            case 9:
                logger.info("Exiting the loop. Goodbye!")
                break
            case _:
                logger.info("Invalid option. Try again.")

if __name__ == "__main__":
    main()

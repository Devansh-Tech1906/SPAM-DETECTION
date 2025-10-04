import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # Used for saving and loading the model

# --- File Paths ---
# Define file paths for the dataset and the saved model components
DATASET_PATH = 'spam.csv'
MODEL_PATH = 'spam_model.joblib'
VECTORIZER_PATH = 'vectorizer.joblib'

def train_and_save_model():
    """
    Loads the dataset, trains two different classifiers (Naive Bayes and Logistic Regression),
    evaluates them, and saves the best performing model and the vectorizer to disk.
    """
    print("--- Starting Model Training ---")
    
    # 1. Load Data
    try:
        # This dataset often requires a specific encoding
        df = pd.read_csv(DATASET_PATH, encoding='latin-1')
    except FileNotFoundError:
        print(f"Error: The dataset file '{DATASET_PATH}' was not found.")
        print("Please download it and place it in the same directory as the script.")
        return

    # 2. Preprocess Data
    # Keep only the necessary columns and rename them for clarity
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("Sample of the data:")
    print(df.head())
    print("\n")

    X = df['text']
    y = df['label']

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Feature Extraction (Vectorization)
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 5. Train and Evaluate Models
    
    # --- Model 1: Multinomial Naive Bayes ---
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    y_pred_nb = nb_model.predict(X_test_tfidf)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print("--- Multinomial Naive Bayes Evaluation ---")
    print(f"Accuracy: {accuracy_nb:.4f}")
    
    # Advanced Evaluation Metrics
    print("\nConfusion Matrix:")
    # A confusion matrix shows:
    # [[True Negatives, False Positives],
    #  [False Negatives, True Positives]]
    print(confusion_matrix(y_test, y_pred_nb, labels=['ham', 'spam']))
    
    print("\nClassification Report:")
    # Precision: Out of all predicted as spam, how many were correct?
    # Recall: Out of all actual spam, how many did we catch?
    print(classification_report(y_test, y_pred_nb))

    # --- Model 2: Logistic Regression ---
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    y_pred_lr = lr_model.predict(X_test_tfidf)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    print("\n--- Logistic Regression Evaluation ---")
    print(f"Accuracy: {accuracy_lr:.4f}")
    print(classification_report(y_test, y_pred_lr))

    # 6. Save the Best Model and the Vectorizer
    # In this case, both models perform very well. We'll save the Naive Bayes model.
    joblib.dump(nb_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"--- Model and Vectorizer have been saved successfully! ---")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}\n")


def predict_from_saved_model():
    """
    Loads the saved model and vectorizer from disk and uses them to classify
    new, user-provided email text.
    """
    # 1. Check if model files exist
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Model files not found. Please train a model first (Option 1).")
        return

    # 2. Load the saved model and vectorizer
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("--- Model and vectorizer loaded successfully ---")
    except Exception as e:
        print(f"An error occurred while loading files: {e}")
        return

    # 3. Get user input
    print("\nEnter the email text you want to classify.")
    print("Type 'exit' when you are done.")
    
    while True:
        email_text = input("\nYour email text: ")
        if email_text.lower() == 'exit':
            break
        
        if not email_text.strip():
            print("Please enter some text.")
            continue

        # 4. Make a prediction
        email_tfidf = vectorizer.transform([email_text])
        prediction = model.predict(email_tfidf)
        probability = model.predict_proba(email_tfidf)

        # 5. Display the result
        spam_prob = probability[0][1] # Probability of being spam
        
        print("-" * 20)
        print(f"Prediction: {prediction[0].upper()}")
        print(f"Confidence (Spam Probability): {spam_prob:.2%}")
        print("-" * 20)


def main_menu():
    """
    Displays the main menu and handles user interaction.
    """
    while True:
        print("\n--- Advanced Spam Detector Menu ---")
        print("1. Train a New Model")
        print("2. Classify an Email with a Loaded Model")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            train_and_save_model()
        elif choice == '2':
            predict_from_saved_model()
        elif choice == '3':
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# --- Main Execution ---
if __name__ == "__main__":
    main_menu()


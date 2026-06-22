import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocess import UrduPreprocessor
from src.features import create_tfidf_features
from src.train import train_models, save_model
from src.evaluate import evaluate_model

print("=" * 60)
print("URDU NEWS CLASSIFICATION")
print("=" * 60)

df = pd.read_csv('data/urdu_news_dataset.csv')

preprocessor = UrduPreprocessor()
df = preprocessor.preprocess_dataframe(df)

X = df['cleaned_text']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_vec, X_test_vec, vectorizer = create_tfidf_features(X_train, X_test)

results = train_models(X_train_vec, y_train, X_test_vec, y_test)

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"\nBest Model: {best_model_name} with {results[best_model_name]['accuracy'] * 100:.2f}% accuracy")

save_model(results[best_model_name]['model'], 'models/best_model.pkl')
save_model(vectorizer, 'models/vectorizer.pkl')

print("\nModel saved successfully!")

labels = df['category'].unique()
evaluate_model(y_test, results[best_model_name]['predictions'], labels)
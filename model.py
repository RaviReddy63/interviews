# ===== simple_modeling.py =====
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def train_simple_models(data, y_column):
    """Train logistic regression and decision tree with specified target column."""
    
    # Prepare X and y
    y = data[y_column]
    X = data.drop(columns=[y_column])
    
    print(f"Using '{y_column}' as target variable")
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train models
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
    
    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    
    # Predictions
    lr_pred = lr_model.predict(X_test)
    dt_pred = dt_model.predict(X_test)
    
    # Results
    lr_accuracy = accuracy_score(y_test, lr_pred)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    
    print("\nLOGISTIC REGRESSION")
    print(f"Accuracy: {lr_accuracy:.3f}")
    print(classification_report(y_test, lr_pred))
    
    print("\nDECISION TREE")
    print(f"Accuracy: {dt_accuracy:.3f}")
    print(classification_report(y_test, dt_pred))
    
    return lr_model, dt_model, X_test, y_test, lr_pred, dt_pred


def plot_simple_results(y_test, lr_pred, dt_pred):
    """Simple bar plot of model accuracies."""
    
    lr_accuracy = accuracy_score(y_test, lr_pred)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    
    models = ['Logistic Regression', 'Decision Tree']
    accuracies = [lr_accuracy, dt_accuracy]
    
    plt.figure(figsize=(8, 5))
    plt.bar(models, accuracies, color=['blue', 'green'], alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1)
    
    # Add accuracy labels on bars
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.show()


def predict_new_party(model, X_test):
    """Simple prediction for new party using test data example."""
    
    # Use first row of test data as example
    new_features = X_test.iloc[0].values
    
    prediction = model.predict([new_features])
    probability = model.predict_proba([new_features])
    
    print(f"\nPREDICTING NEW PARTY:")
    print(f"Features: {new_features}")
    print(f"Prediction: {prediction[0]}")
    
    # Show probabilities for all classes
    classes = model.classes_
    for i, prob in enumerate(probability[0]):
        print(f"Probability {classes[i]}: {prob:.3f}")
    
    return prediction[0]


# ===== Complete Example =====
def run_modeling_with_target_column(data, y_column):
    """Run complete modeling with specified target column."""
    
    # Check if column exists
    if y_column not in data.columns:
        print(f"Error: Column '{y_column}' not found in data")
        print(f"Available columns: {list(data.columns)}")
        return
    
    # Train models
    lr_model, dt_model, X_test, y_test, lr_pred, dt_pred = train_simple_models(data, y_column)
    
    # Plot results
    plot_simple_results(y_test, lr_pred, dt_pred)
    
    # Predict for new party
    predict_new_party(lr_model, X_test)


# ===== Usage Examples =====
def example_with_real_data():
    """Example using the political party data."""
    
    # Load your data
    from political_party_analysis.loader import DataLoader
    
    data_loader = DataLoader()
    df = data_loader.preprocess_data()
    
    # Add a simple target column for demo
    # You can replace this with any actual column name from your data
    if 'lrgen' in df.columns:
        # Use existing left-right column
        target_column = 'lrgen'
    else:
        # Create a dummy target based on first component
        first_comp = df.iloc[:, 0]
        df['political_leaning'] = (first_comp <= first_comp.median()).map({True: 'Left', False: 'Right'})
        target_column = 'political_leaning'
    
    print(f"Available columns: {list(df.columns)}")
    
    # Run modeling
    run_modeling_with_target_column(df, target_column)


def example_with_dummy_data():
    """Simple example with dummy data."""
    
    # Create dummy dataset
    np.random.seed(42)
    dummy_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'target_variable': np.random.choice(['Left', 'Right'], 100)
    })
    
    print("Dummy data created with columns:", list(dummy_data.columns))
    
    # Run modeling with specified target
    run_modeling_with_target_column(dummy_data, 'target_variable')


# ===== Test Function =====
def test_modeling_with_target():
    """Simple test."""
    
    # Create test data
    test_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [2, 3, 1, 5, 4, 6],
        'my_target': ['A', 'B', 'A', 'B', 'A', 'B']
    })
    
    # Test that it works
    try:
        run_modeling_with_target_column(test_data, 'my_target')
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    print("Running dummy data example...")
    example_with_dummy_data()
    
    print("\n" + "="*50)
    print("Running test...")
    test_modeling_with_target()
    
    # Uncomment to run with real data
    # print("\nRunning with real data...")
    # example_with_real_data()

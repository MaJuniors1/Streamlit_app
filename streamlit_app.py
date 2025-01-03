import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
st.title("Water Potability Analysis")
st.write("This application allows you to explore and analyze the water potability dataset.")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    data_clean = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data_clean.head())
    
    # Display correlation heatmap
    if st.button("Show Correlation Heatmap"):
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 8))
        corr = data_clean.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        st.pyplot(plt)
    
    # Train-test split
    X = data_clean.drop('Potability', axis=1)
    y = data_clean['Potability']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models and display accuracy comparison
    if st.button("Train Models"):
        # Train models
        models = {
            "KNN": KNeighborsClassifier(),
            "Random Forest": RandomForestClassifier(random_state=42),
            "GaussianNB": GaussianNB()
        }
        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append({"Model": name, "Accuracy": accuracy})
        
        results_df = pd.DataFrame(results)
        st.write("### Model Accuracy Comparison")
        st.bar_chart(results_df.set_index("Model"))


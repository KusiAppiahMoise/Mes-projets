# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model
model_path = r"my_model.pkl"
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Create a Streamlit app
st.title("Détection des fraudes à la carte de crédit")
st.sidebar.header("Fonctionnalités d’entrée de l'utilisateur")

# Dictionary containing the types of transaction
transaction_type_dic = {"CASH_OUT": 1,"PAYMENT": 2,"CASH_IN": 3,"TRANSFER": 4,"DEBIT": 5}

# Collect user input features (e.g., transaction amount, type, etc.)
type = st.sidebar.selectbox("Transaction Type", list(transaction_type_dic.keys()))
amount = st.sidebar.number_input("Montant de la transaction", min_value=0.0, step=1.0)
oldbalanceOrg = st.sidebar.number_input("Solde de l'expediteur avant transaction", min_value=0.0, step=1.0)
newbalanceOrig = st.sidebar.number_input("Solde de l'expediteur après transaction ", min_value=0.0, step=1.0)
oldbalanceDest = st.sidebar.number_input("Solde du destinataire avant transaction", min_value=0.0, step=1.0)
newbalanceDest = st.sidebar.number_input("Solde du destinataire après transaction", min_value=0.0, step=1.0)
# Add more relevant features here...

if st.sidebar.button("Vérifier"):
    # Create a feature vector from user input
    user_features = np.array([transaction_type_dic.get(type), amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest])  # Add other features as needed

    # Make predictions using the pre-trained model
    fraud_probability = model.predict_proba(user_features.reshape(1, -1))[:, 1]

    # Display the prediction result
    st.write(f"Probabilité de fraude: {fraud_probability[0]:.0%}")

    # Add more Streamlit components (charts, plots, etc.) as desired

    # Display a bar chart of fraud probabilities
    st.bar_chart({"Probabilité de fraude": [fraud_probability[0], 1 - fraud_probability[0]]})

    # Display additional information or insights
    st.write("Sur la base des caractéristiques fournies, la transaction est classée comme frauduleuse si la probabilité dépasse un certain seuil.")
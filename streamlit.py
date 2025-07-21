import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
import os

def load_saved_model(model_path, format_type='auto'):
    """
    Load a previously saved model.

    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    format_type : str, default='auto'
        Format of the saved model ('joblib', 'pickle', or 'auto' to detect)

    Returns:
    --------
    Loaded model pipeline
    """
    try:
        if format_type == 'auto':
            # Auto-detect format based on file extension
            if model_path.endswith('.joblib'):
                return joblib.load(model_path)
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            else:
                # Try joblib first, then pickle
                try:
                    return joblib.load(model_path)
                except Exception:
                    with open(model_path, 'rb') as f:
                        return pickle.load(f)
        elif format_type == 'joblib':
            return joblib.load(model_path)
        else:  # pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def make_predictions_with_saved_model(model_path, X_new, return_proba=False):
    """
    Make predictions using a saved model.

    Parameters:
    -----------
    model_path : str
        Path to the saved model file
    X_new : array-like
        New data to make predictions on
    return_proba : bool, default=False
        Whether to return prediction probabilities

    Returns:
    --------
    array: Predictions (and probabilities if requested)
    """
    # Load the model
    model = load_saved_model(model_path)

    if model is None:
        return None

    try:
        # Make predictions
        predictions = model.predict(X_new)

        if return_proba:
            try:
                probabilities = model.predict_proba(X_new)
                return predictions, probabilities
            except AttributeError:
                st.warning("Model doesn't support probability predictions")
                return predictions

        return predictions

    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

# Streamlit UI
st.title("🔍 Credit Card Fraud Detection")
st.markdown("---")

# Sidebar for model selection
st.sidebar.header("Model Configuration")

# Available models in the directory
available_models = [
    item for item in os.listdir("./") if item.endswith(".joblib") or item.endswith(".pkl")
]

selected_model = st.sidebar.selectbox(
    "Select a pre-trained model:",
    available_models,
    index=0
)

# Load the selected model
model_path = f"./{selected_model}"

df = pd.read_csv("./creditcard.csv")
print(df.head())
fraud_row = df[df["Class"] == 1].iloc[0, :].tolist()[:-1]
normal_row = df[df["Class"] == 0].iloc[0, :].tolist()[:-1]
print(fraud_row)
print(normal_row)

feature_set_selection = [
    fraud_row,
    normal_row
]

selected_feature_set = st.sidebar.selectbox(
    "Select a example feature set:",
    feature_set_selection,
    index=0
)

# Create feature array in the correct order
if st.button("🔍 Predict Fraud", type="primary"):
    try:

        # Convert to numpy array and reshape for prediction
        X_new = np.array(selected_feature_set).reshape(1, -1)
        
        # Make prediction with probabilities
        result = make_predictions_with_saved_model(model_path, X_new, return_proba=True)
        
        if result is not None:
            if isinstance(result, tuple):
                predictions, probabilities = result
                prediction = predictions[0]
                prob_fraud = probabilities[0][1] if len(probabilities[0]) > 1 else 0
                prob_normal = probabilities[0][0] if len(probabilities[0]) > 0 else 0
            else:
                prediction = result[0]
                prob_fraud = None
                prob_normal = None
            
            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Results")
            
            if prediction == 1:
                st.error("🚨 **FRAUD DETECTED** 🚨")
                st.markdown("This transaction appears to be **fraudulent**.")
            else:
                st.success("✅ **LEGITIMATE TRANSACTION** ✅")
                st.markdown("This transaction appears to be **legitimate**.")
            
            # Show probabilities if available
            if prob_fraud is not None:
                st.subheader("Confidence Scores")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        label="Probability of Fraud",
                        value=f"{prob_fraud:.4f}",
                        delta=f"{prob_fraud*100:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        label="Probability of Legitimate",
                        value=f"{prob_normal:.4f}",
                        delta=f"{prob_normal*100:.2f}%"
                    )
                
                # Progress bars for visual representation
                st.subheader("Visual Confidence")
                st.write("Fraud Risk:")
                st.progress(prob_fraud)
                st.write("Legitimate Confidence:")
                st.progress(prob_normal)
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page title and icon
st.set_page_config(page_title="Model Prediction App", page_icon="🤖")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# App title and description
st.title("Model Prediction App")
st.write("This app uses a pre-trained model to make predictions based on your input.")

# Check if model loaded successfully
if model is None:
    st.error("Failed to load the model. Please ensure 'model.pkl' exists in the same directory.")
    st.stop()

# Get feature names (assuming the model has this attribute)
try:
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    elif hasattr(model, 'feature_importances_'):
        # If feature names aren't available, create generic ones
        num_features = len(model.feature_importances_)
        feature_names = [f'Feature_{i+1}' for i in range(num_features)]
    else:
        # For models without feature_importances_, ask user for number of features
        num_features = st.number_input("Enter number of features in your model:", 
                                     min_value=1, value=5, step=1)
        feature_names = [f'Feature_{i+1}' for i in range(num_features)]
except:
    st.warning("Couldn't determine number of features. Using default feature names.")
    feature_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4', 'Feature_5']

# Input form for features
st.header("Input Features")
input_data = {}

# Create input widgets for each feature
cols = st.columns(2)  # Split inputs into 2 columns for better layout
for i, feature in enumerate(feature_names):
    with cols[i % 2]:  # Alternate between columns
        input_data[feature] = st.number_input(
            f"{feature}:",
            value=0.0,
            step=0.01,
            format="%.2f"
        )

# Prediction button
if st.button("Make Prediction"):
    try:
        # Convert input to DataFrame (model expects 2D array)
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Display results
        st.success("Prediction made successfully!")
        st.subheader("Results")
        
        if len(prediction.shape) == 1 and prediction.shape[0] == 1:
            # Single output
            st.metric(label="Prediction", value=f"{prediction[0]:.2f}")
        else:
            # Multiple outputs
            for i, pred in enumerate(prediction[0]):
                st.metric(label=f"Output {i+1}", value=f"{pred:.2f}")
                
        # Show input data
        st.subheader("Input Summary")
        st.write(input_df)
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")

# Add some info about the model
st.sidebar.header("Model Information")
st.sidebar.write(f"Model type: {type(model).__name__}")
try:
    if hasattr(model, 'feature_importances_'):
        st.sidebar.subheader("Feature Importances")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.sidebar.dataframe(importance_df)
except:
    pass

# Add footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit")

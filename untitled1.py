import streamlit as st
import pickle

# Set the title of the Streamlit app
st.title('Model Prediction App')

# Add a description
st.write('This app loads a pre-trained model and makes predictions.')

# Load the model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    st.success('Model loaded successfully!')
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found. Please make sure the model file is in the same directory as 'app.py'.")
    st.stop() # Stop the app if the model isn't found
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Example: How to use the loaded model (replace with your actual model's input) ---

st.header('Make a Prediction')

# Assuming your model expects numerical inputs, create input fields
# Replace these with the actual features your model expects
feature1 = st.slider('Input Feature 1', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
feature2 = st.number_input('Input Feature 2', min_value=0.0, max_value=100.0, value=50.0)
feature3 = st.checkbox('Input Feature 3 (True/False)')

# Create a button to trigger prediction
if st.button('Predict'):
    # Prepare input data for the model
    # This part heavily depends on how your model expects input.
    # For example, if it's a scikit-learn model, it might expect a 2D array.
    
    # Example: If your model expects a list of features
    input_data = [feature1, feature2, 1 if feature3 else 0] # Convert boolean to 0 or 1

    try:
        # Make a prediction
        # You might need to reshape your input_data depending on your model
        # For a single prediction with a scikit-learn model, it's often model.predict([[feature1, feature2, ...]])
        prediction = model.predict([input_data])
        
        st.subheader('Prediction Result:')
        st.write(f"The predicted value is: **{prediction[0]}**") # Assuming prediction returns an array
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("""
---
**Instructions:**
1.  Save this code as `app.py`.
2.  Make sure your `model.pkl` file is in the same directory as `app.py`.
3.  Install Streamlit: `pip install streamlit`
4.  Run the app from your terminal: `streamlit run app.py`

**Note:** You will need to adjust the "Make a Prediction" section (input fields and `input_data` preparation) to match the exact input requirements of your `model.pkl`.
""")

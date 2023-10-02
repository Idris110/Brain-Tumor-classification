import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score

# Load the saved models
rf = joblib.load('random_forest_model.pkl')
lg = joblib.load('logistic_regression_model.pkl')
sv = joblib.load('support_vector_model.pkl')
dt = joblib.load('decision_tree_model.pkl')

# Define the class labels
dec = {0: 'No Tumor', 1: 'Positive Tumor'}

# Streamlit app code
st.title("Brain Tumor Classifier")

# Navigation bar
nav_choice = st.sidebar.selectbox("Navigation", ["Home", "Model Selection", "Model Accuracy"])

if nav_choice == "Home":
    st.write("Welcome to the Brain Tumor Classifier!")
    st.markdown(
    f'<img style="left: -145px; width: 500px; position: absolute;" src="https://i.imgur.com/6ePVWRj.gif" alt="cat gif">',
    unsafe_allow_html=True,
)
elif nav_choice == "Model Selection":
    st.header("Model Selection")
    selected_model = st.selectbox("Select a model", ["Random Forest", "Logistic Regression", "Support Vector", "Decision Tree"])
    uploaded_file = st.file_uploader("Upload a black and white image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            # Read the uploaded image using OpenCV
            image_bytes = uploaded_file.read()
            if len(image_bytes) == 0:
                st.error("Empty image file. Please upload a valid image.")
            else:
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

                # Check if the image has valid dimensions
                if image is not None and image.shape[0] > 0 and image.shape[1] > 0:
                    # Resize the image to the desired size (e.g., 200x200)
                    image = cv2.resize(image, (200, 200))
                    img = cv2.resize(image, (200, 200))

                    # Reshape and normalize the image
                    image = image.reshape(1, -1) / 255.0

                    # Make predictions based on the selected model
                    if selected_model == "Random Forest":
                        prediction = rf.predict(image)
                    elif selected_model == "Logistic Regression":
                        prediction = lg.predict(image)
                    elif selected_model == "Support Vector":
                        prediction = sv.predict(image)
                    elif selected_model == "Decision Tree":
                        prediction = dt.predict(image)

                    # Display the result
                    st.image(img, caption=f"Predicted Class: {dec[prediction[0]]}", width=100)

                else:
                    st.error("Invalid image dimensions. Please upload a valid image.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

elif nav_choice == "Model Accuracy":
    st.header("Model Accuracy")
    st.subheader("Model Accuracy Scores:")

    # Create a DataFrame for accuracy scores
    models = ["Random Forest", "Logistic Regression", "Support Vector", "Decision Tree"]
    training_scores = [1.0, 1.0, 0.9108910891089109, 1.0]
    testing_scores = [0.8235294117647058, 0.8235294117647058, 0.7450980392156863, 0.7450980392156863]

    accuracy_df = pd.DataFrame({"Model": models, "Training Score": training_scores, "Testing Score": testing_scores})

    # Display accuracy scores as a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Model", y="Training Score", data=accuracy_df, color="b", label="Training Score", ax=ax)
    sns.barplot(x="Model", y="Testing Score", data=accuracy_df, color="r", alpha=0.6, label="Testing Score", ax=ax)
    plt.title("Model Accuracy Scores")
    plt.xlabel("Model")
    plt.ylabel("Accuracy Score")
    plt.legend(loc="upper left")

    # Display the Matplotlib figure using st.pyplot
    st.pyplot(fig)

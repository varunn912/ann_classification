# README.md

# 🚀 Bank Customer Churn Prediction

This is a web application that predicts customer churn for a bank using an Artificial Neural Network (ANN). The app also provides explanations for its predictions using SHAP (SHapley Additive exPlanations).

![App Screenshot](https://i.imgur.com/your-screenshot-url.png)  ---

## ✨ Features

- **Single Prediction**: Input a single customer's details through a user-friendly form to get an instant churn prediction.
- **Batch Prediction**: Upload a CSV file with multiple customers to get predictions for all of them at once.
- **Prediction Explanation**: For single predictions, a SHAP force plot visualizes which features contributed most to the prediction, making the model's decision transparent.
- **Improved UI**: A clean, tabbed interface built with Streamlit and styled with custom CSS.

---

## 🛠️ Tech Stack

- **Backend**: Python
- **Machine Learning**: TensorFlow/Keras, Scikit-learn
- **Model Explainability**: SHAP
- **Web Framework**: Streamlit
- **Data Manipulation**: Pandas, NumPy

---

## 📦 Setup & Installation

Follow these steps to run the project locally.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/ann-classification.git](https://github.com/your-username/ann-classification.git)
    cd ann-classification
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser.

---

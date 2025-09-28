# Retinal AI: A Non-Invasive Approach to Early Cardiovascular Risk Screening

This project explores the use of deep learning to screen for cardiovascular risk non-invasively by analyzing retinal fundus images. The core of this repository is a **Convolutional Neural Network (CNN)** trained to detect **Hypertensive Retinopathy**, a key indicator of chronic high blood pressure and a major risk factor for heart disease.

The model serves as a proof-of-concept for using the eye as a "window" to systemic health, creating a powerful proxy for cardiovascular risk assessment.

---

## ## Key Features

* **High-Accuracy Detection:** The model achieves **96% test accuracy** in classifying Hypertensive Retinopathy vs. Normal retinas.
* **Transfer Learning:** Leverages the **MobileNetV2** architecture, pre-trained on ImageNet, for fast and effective training.
* **Data Augmentation:** Implements a robust data augmentation pipeline to ensure high model generalization and prevent overfitting.
* **Scalable Architecture:** The project is designed as the core "engine" for a larger health screening application.

---

## ## Technology Stack

* **Language:** Python
* **Libraries:** TensorFlow, Keras, Scikit-learn, NumPy
* **Environment:** Google Colab (with GPU acceleration)
* **Dataset:** A curated binary dataset from the Kaggle "Ocular Disease Dataset".

---

## ## Project Workflow

The project was executed in four main phases:

1.  **Data Curation:** Engineered a focused binary dataset by isolating 'Hypertensive Retinopathy' and 'Normal' images from a larger multi-class collection.
2.  **Data Preparation:** Structured a rigorous 80/10/10 train/validation/test split and applied a data augmentation pipeline to the training set.
3.  **Model Training:** Built and trained a CNN using Transfer Learning (MobileNetV2). The pre-trained base was frozen, and a custom classification head was trained on the curated dataset.
4.  **Performance Evaluation:** Validated the model's real-world viability using a confusion matrix, precision, and recall on the unseen test set.



---

## ## Model Performance

* **Test Accuracy:** 96%
* **Loss:** Low (approx. 0.18)
* **Key Insight:** The model demonstrates excellent generalization, with validation and training accuracies remaining closely aligned throughout the training process.

---

## ## Setup and Usage

To replicate this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)<Your-Username>/Retinal-Cardio-Risk-AI.git
    cd Retinal-Cardio-Risk-AI
    ```

2.  **Set up the environment:**
    The project is designed to run in a Google Colab notebook. Upload the `.ipynb` file to Colab and ensure the runtime is set to `T4 GPU`.

3.  **Download the Dataset:**
    The notebook includes commands to download the required dataset directly from Kaggle using the Kaggle API. You will need to upload your `kaggle.json` key when prompted.

4.  **Run the Notebook:**
    Execute the cells in the notebook sequentially to perform data preparation, model training, and evaluation.

---

## ## Future Work

This model serves as the foundational component for a larger, user-facing application. The architected future path includes:

* **Integration with a Conversational AI:** Develop a chatbot interface where a user can upload their retinal scan.
* **Framingham Risk Score Calculation:** Use the AI's output (Hypertension: Yes/No) as an input for the medically-validated **Framingham Risk Score**, combining it with user-provided data (e.g., age, cholesterol) to calculate a final 10-year cardiovascular risk percentage.
* **Building "Model B":** Train a similar model to detect Diabetic Retinopathy to add another critical risk factor to the calculation.

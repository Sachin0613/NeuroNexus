# Credit Card Fraud Detection using Machine Learning

This project focuses on building a machine learning model to detect fraudulent credit card transactions using the dataset provided by [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The dataset is highly imbalanced, and the project addresses this issue through various data preprocessing and resampling techniques.

## Dataset

* **Source**: [Credit Card Fraud Detection | Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Description**: This dataset contains transactions made by European credit card holders in September 2013. It includes 284,807 transactions, where only 492 are fraudulent (0.172%).

## Objectives

* Preprocess and normalize transaction data.
* Handle class imbalance using techniques like SMOTE (oversampling) or undersampling.
* Split data into training and test sets.
* Train classification algorithms (Logistic Regression, Random Forest).
* Evaluate models using metrics such as **Precision**, **Recall**, and **F1-Score**.


## Setup Instructions

1. **Clone the Repository**

   ```
   git clone https://github.com/Sachin0613/NeuroNexus.git
   ```

2. **Install Dependencies**

   ```
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook**

   ```
   jupyter notebook notebooks/EDA_and_Model_Training.ipynb
   ```

## Key Features

* **Data Preprocessing**:

  * Scaling with StandardScaler.
  * Feature transformation.
  * Train-test split (stratified).

* **Handling Imbalance**:

  * Random Undersampling.
  * SMOTE Oversampling.

* **Models Trained**:

  * Logistic Regression.
  * Random Forest Classifier.


* **Evaluation Metrics**:

  * Confusion Matrix
  * Precision, Recall, F1-Score
  * ROC-AUC Score

## Results

[DataScienceWordFile.docx](https://github.com/user-attachments/files/20039855/DataScienceWordFile.docx)

| Model               | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.88      | 0.76   | 0.82     | 0.97    |
| Random Forest       | 0.93      | 0.84   | 0.88     | 0.99    |

## Future Improvements

* Try ensemble models like XGBoost or LightGBM.
* Tune hyperparameters using GridSearchCV.
* Deploy the model using Flask/FastAPI for real-time detection.

## License

This project is licensed under the MIT License.

## Author

* **Your Name** – [GitHub](https://github.com/Sachin0613)



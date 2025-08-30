# ML Attack in A Probabilistic Circuit with Phase Change Memory Nanodevices & Pass Transistor Logic Selectors

## Description
This repository contains the verification of **randomness** of a **probabilistic circuit** by investigating the capability of different machine learning algorithms capability to predict the **output of the circuit** i.e., **Machine Learning Attacks**, focusing on binary classification of circuit simulation data. The pipeline evaluates multiple machine learning models to predict output behavior based on resistance values extracted from PCM devices in the custom-designed circuit proposed in our work.

The pipeline includes:
- Data loading, preprocessing, and transformation of hexadecimal resistance values into integers.  
- Training and evaluation of various machine learning models (classical ML and deep learning).  
- Automated hyperparameter tuning using `RandomizedSearchCV`.  
- Visualization of performance metrics (AUC-ROC, Precision-Recall curves, Confusion Matrices).  
- Saving results and plots for each dataset and train-test split.  

---

## Dataset

The datasets were generated using the **LTspice circuit simulator**.  

- **Number of datasets:** 12 (named `Set1`, `Set2`, …, `Set12`)  
- **Samples per dataset:** 3,136  
- **Features:**  
  - `R1`, `R2`, `R3`, `R4` → resistances of four PCM devices (originally in **hexadecimal**, converted to **integers**)  
- **Target:**  
  - `Out_simulation_bin` → binary output voltage (0 or 1)  
- **Preprocessing steps:**  
  - Conversion of resistance values from hexadecimal to integers  
  - Train-test splitting with varying ratios (10%–90% test size)  
- **Evaluation Metric:**  
  - **AUC-ROC (Area Under the Receiver Operating Characteristic Curve)**, as it is robust for binary classification.  

---

## Code Overview

### **main_classification.ipynb** or **2. main_classification.py**
- Jupyter Notebook with detailed step-by-step implementation:
  - Data exploration & preprocessing
  - Feature transformation (hex → integer)
  - Training multiple classifiers
  - Hyperparameter tuning with `RandomizedSearchCV`
  - Model evaluation (Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR)
  - Visualization of results
  - Saving results to CSV

---

## Models Implemented
The following models were implemented and compared:  

- **Ensemble Methods**:  
  - Random Forest Classifier  
  - Gradient Boosting  
  - XGBoost 
  - LightGBM  

- **Linear Models**:  
  - Logistic Regression  

- **Support Vector Machines**:  
  - SVM with linear & RBF kernels  

- **Instance-based Learning**:  
  - k-Nearest Neighbors (KNN)  

- **Probabilistic Models**:  
  - Naïve Bayes  

- **Tree-based Methods**:  
  - Decision Tree  

- **Basic Neural Networks**:  
  - Feedforward Deep Neural Network (Keras/TensorFlow)  

---

## Input & Output

### **Inputs**
- Excel file: `ML_data_set_Statistical_Analysis_0V_0V.xlsx`  
  - Contains 12 sheets (`Set1` → `Set12`) with 3,136 rows each.  
- Configuration parameters (train-test split ratio, model type, tuning flag).  

### **Outputs**
- **Evaluation metrics** (Accuracy, AUC-ROC, AUC-PR, Precision, Recall, F1).  
- **Confusion Matrix heatmaps** for Training sets.  
- **Precision-Recall curves** for Training sets.  
- **AUC-ROC vs Train Size plots** (individual and combined).  
- **CSV file**: `auc_roc_results.csv` storing AUC-ROC scores for each dataset and train size.  
- **Saved PNG plots** (per dataset & combined).  

---

## How to Run the Code

### **Option 1: Run in Jupyter Notebook**
1. Open `ML_security_final_v2.ipynb` in **Google Colab** or Jupyter.  
2. Upload your dataset (`ML_data_set_Statistical_Analysis_0V_0V.xlsx`).  
3. Execute cells step by step to preprocess, train, evaluate, and visualize results.  

### **Option 2: Run as Python Script**
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
2. python main_classification.py

## 📦 Dependencies

Make sure the following libraries are installed:  

- **Core Libraries**:  
  - numpy, pandas, scipy, matplotlib, seaborn, os  

- **Scikit-Learn**:  
  - sklearn (RandomForest, GradientBoosting, SVM, Logistic Regression, KNN, Naïve Bayes, Decision Tree, metrics, preprocessing, model_selection)  

- **Boosting Libraries**:  
  - xgboost  
  - lightgbm  

- **Deep Learning**:  
  - tensorflow (Keras Sequential, Dense layers, Adam optimizer)  

- **Google Colab (optional)**:  
  - google.colab for file download support  

## Results
- Models were evaluated across **12 datasets** and **10 train-test splits (10%–90%)**.  
- AUC-ROC was used as the primary performance metric.  
- Plots show **how AUC-ROC varies with training size across datasets**, highlighting model robustness and generalization.  

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.  

---

## 📬 Contact
For questions, issues, or contributions, feel free to open an **Issue** or submit a **Pull Request**.  

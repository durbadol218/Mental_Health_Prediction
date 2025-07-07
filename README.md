# 🧠 Mental Health Treatment Prediction in the Tech Industry

This project analyzes trends in mental health within the tech industry and uses machine learning to predict whether an individual will seek treatment for mental health issues, based on survey data.

---

## 📌 Objectives

- Analyze patterns related to mental health support in tech workplaces.
- Predict whether an individual will seek mental health treatment (binary classification).
- Identify key features that influence treatment-seeking behavior.

---

## 🧾 Dataset

- **Source**: [OSMI Mental Health in Tech Survey – Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- **Rows**: ~1,400  
- **Columns**: ~27  
- **Target**: `treatment` (`Yes` / `No`)

---

## 📁 Project Structure

```
mental_health_prediction/
├── data/
│   └── survey.csv                  # Original dataset
├── models/
│   └── model.pkl                   # Trained model
├── notebooks/
│   └── analysis.ipynb              # Visualizations and analysis
├── main.py                         # Main training script
├── visualize.py                    # Helper functions for plotting
└── README.md                       # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/durbadol218/mental_health_prediction.git
   cd mental_health_prediction
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate       # For Linux/macOS
   .venv\Scripts\activate        # For Windows
   ```

3. **Install the required packages**
   ```bash
   pip install -r requirements.txt
   ```

   Or manually install:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

---

## 🚀 How to Run

1. **Train the model**
   ```bash
   python main.py
   ```

2. **View visualizations**
   - Open the notebook:
     ```bash
     jupyter notebook notebooks/analysis.ipynb
     ```
   - Or use `visualize.py` functions directly in scripts.

---

## 📊 Models Used

- **Logistic Regression** – Simple baseline
- **Perceptron** – Neural-inspired binary classifier
- **Decision Tree** – Interpretable model
- **Random Forest** ✅ – Best performance
- **MLP Classifier** – Multi-layer neural network

---

## 🧪 Evaluation

| Class                       | Precision | Recall | F1-Score | Support |
| --------------------------- | --------- | ------ | -------- | ------- |
| **0** (No Treatment)        | 0.87      | 0.78   | 0.82     | 129     |
| **1** (Will Seek Treatment) | 0.79      | 0.88   | 0.83     | 123     |
| **Accuracy**                | –         | –      | **0.83** | 252     |
| **Macro Avg**               | 0.83      | 0.83   | 0.83     | 252     |
| **Weighted Avg**            | 0.83      | 0.83   | 0.83     | 252     |

- **ROC-AUC**: `0.894`

---

## 📊 Visualizations Include

- Confusion Matrix  
- Classification Report Heatmap  
- ROC Curve  
- Model Comparison Bar Chart (F1-Score & ROC-AUC)

All visualizations are available in `notebooks/analysis.ipynb`.

---

## 🛠️ Technologies Used

- Python 3
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

---

## 📬 Contact

If you have any questions or suggestions, feel free to reach out via GitHub issues or email at `goswamidurbadol@example.com`.

---

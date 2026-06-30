# 🎓 Student Dropout Prediction — Early Warning System for Education

An end-to-end machine learning project that predicts students at risk of **dropping out** of higher education, enabling institutions to intervene early. Built with a focus on **social impact** and **imbalanced classification**.

> Goal: identify at-risk students *before* they drop out, so support (tutoring, counseling, financial aid) can be provided in time.

---

## 📌 Problem Statement

Academic dropout is a major challenge in higher education. If students at risk can be identified early — based on data available in their first year — institutions can act before it's too late.

This project answers: **"Which students are likely to drop out, and what factors drive that risk?"**

---

## 📊 Dataset

- **4,424 students**, **35 features** (UCI Machine Learning Repository)
- Features: demographics, socio-economic background, parents' education/occupation, and **academic performance (1st & 2nd semester)**
- Target (3 classes): **Dropout**, **Enrolled**, **Graduate**
- **Imbalanced:** Graduate 49.9% · Dropout 32.1% · Enrolled 17.9%
- No missing values (clean, research-grade dataset)

---

## 🔧 Workflow

### 1. Exploratory Data Analysis (EDA)
Key patterns found:
- Students who **drop out** have far **fewer passed credits** in semester 2
- **Unpaid tuition fees** are associated with higher dropout
- **Scholarship holders** graduate at much higher rates → financial aid helps prevent dropout

### 2. Modeling — Multiclass Classification
- Compared **Decision Tree, SVM, Random Forest** — all with `class_weight='balanced'` to handle imbalance from the start
- **Random Forest** performed best

| Metric (Dropout class) | Score |
|---|---|
| Precision | 0.80 |
| **Recall** | **0.79** |

> Recall on the Dropout class is the priority metric — catching as many at-risk students as possible matters more than raw accuracy, since a missed at-risk student means a missed intervention.

### 3. Hyperparameter Tuning (GridSearchCV)
- Ran GridSearchCV (with cross-validation) to find the optimal configuration
- **Key finding & trade-off:** optimizing for `f1_macro` improved the weak *Enrolled* class (recall 0.32 → 0.54) but *lowered* Dropout recall (0.79 → 0.73).
- **Decision:** since the project's goal is dropout detection, the model maximizing **Dropout recall** was selected — demonstrating that the optimization metric must align with the business/social objective.

### 4. Feature Importance
Top drivers of dropout:
1. **Curricular units 2nd sem (approved)** — 12%
2. **Curricular units 2nd sem (grade)** — 10%
3. **Curricular units 1st sem (approved)** — 8%
4. **Curricular units 1st sem (grade)** — 7.5%

---

## 🎯 Key Insight

**Early academic performance is the strongest predictor of dropout** — the top 4 factors are all academic (passed credits & grades in the first two semesters). This is consistent across both EDA and feature importance.

**Recommendation for institutions:** build an **early-warning system** based on first-semester grades and passed credits. Flag low-performing students early and provide fast intervention (tutoring, counseling, academic support) — plus expand scholarship programs, since financial aid correlates with higher graduation rates.

---

## 🛠️ Tech Stack
`Python` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn`

**Techniques:** multiclass classification · imbalanced handling (`class_weight`) · cross-validation · hyperparameter tuning (GridSearchCV) · feature importance

## 📂 Files
- `student_dropout_analysis.ipynb` — full analysis notebook
- `dataset.csv` — dataset
- `README.md` — this file

---

## 🚀 How to Run
```bash
pip install pandas scikit-learn matplotlib seaborn
jupyter notebook student_dropout_analysis.ipynb
```

---

## 📈 Possible Improvements
- Custom scorer to optimize specifically for Dropout recall
- SMOTE for the minority (Enrolled) class
- Deploy as a Streamlit app for interactive risk scoring

"""
Bankruptcy Prediction Analysis - Reproduces Tables 5.1, 5.2, Fig 5.3, Table 5.3
Source: Database_Alis.xlsx (Dataset sheet)
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

RATIOS = ['Current_Ratio','Quick_Ratio','Debt_to_Equity','Interest_Coverage',
          'ROA','Net_Profit_Margin','Asset_Turnover','Firm_Size_lnTA']

# Vendose Database_Alis.xlsx ne te njejtin folder me kete skript
df = pd.read_excel('Database_Alis.xlsx', sheet_name='Dataset')
print("Loaded:", df.shape)

# ---- Table 5.1: Descriptive statistics ----
print("\n=== Table 5.1: Descriptive Statistics ===")
desc = df[RATIOS].agg(['count','mean','std','min','max']).T
print(desc.round(3))

# ---- Table 5.2: Welch t-test by Status ----
print("\n=== Table 5.2: Welch t-Test ===")
for col in RATIOS:
    g0 = df.loc[df['Status']==0, col]
    g1 = df.loc[df['Status']==1, col]
    t, p = stats.ttest_ind(g0, g1, equal_var=False)
    print(f"{col:20s} t={t:7.3f}  p={p:.4f}  {'Sig' if p<0.05 else 'n.s.'}")

# ---- Figure 5.3: Pearson correlation matrix ----
print("\n=== Figure 5.3: Correlation Matrix ===")
print(df[RATIOS].corr().round(2))

# ---- Table 5.3: ML models on Train/Test split (as defined in 'Sample' column) ----
print("\n=== Table 5.3: Model Performance (Test Set) ===")
X_train = df.loc[df['Sample']=='Train', RATIOS]
y_train = df.loc[df['Sample']=='Train', 'Status']
X_test  = df.loc[df['Sample']=='Test',  RATIOS]
y_test  = df.loc[df['Sample']=='Test',  'Status']

models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
    'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42),
    'SVM (RBF)': make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"{name:22s} TN={tn} FP={fp} FN={fn} TP={tp} | Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")

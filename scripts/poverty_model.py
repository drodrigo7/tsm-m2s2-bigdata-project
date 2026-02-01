import warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

# Raw Data
# ==================================================
path = '../data/raw/enaho/'

df_raw = []
for y in range(2004, 2025):
    _df_chunk = pd.read_stata(
        '{}/sumaria-{}.dta'.format(path, y, y),
        convert_categoricals=False,
        convert_missing=False,
        convert_dates=False,
        columns=[
            'conglome',
            'vivienda',
            'hogar',
            'ubigeo',
            'dominio',
            'estrato',
            'percepho',
            'totmieho',
            'mieperho',
            'ga04hd',
            'ia01hd',
            'ingmo1hd',
            'ingmo2hd',
            'inghog1d',
            'inghog2d',
            'gashog1d',
            'gashog2d',
            'pobreza'
        ]
    )
    _df_chunk['year'] = y
    df_raw.append(_df_chunk)

df = pd.concat(df_raw, axis=0)
df.shape
df.head(10)

# Feature Engineering
# ==================================================
# Log para variables monetarias proxy
for c in [
    'ia01hd',
    'ga04hd',
    'ingmo1hd',
    'ingmo2hd',
    'inghog1d',
    'inghog2d',
    'gashog1d',
    'gashog2d'
]:
    if c in df.columns:
        df[f'log_{c}'] = np.log1p(df[c])

# Tamaño relativo
df['dependencia'] = df['totmieho'] / df['percepho'].clip(lower=1)

# Area rural o urbano
df['area'] = np.where(df['estrato'] <= 4, 'urban', 'rural')
df['area'] = df['area'].astype('category')

# Categóricas
dominio_labels = {
    1: 'north_coast',
    2: 'central_coast',
    3: 'south_coast',
    4: 'north_highlands',
    5: 'central_highlands',
    6: 'south_highlands',
    7: 'jungle',
    8: 'metropolitan_lima'
}
df['dominio_cat'] = df['dominio'].map(dominio_labels).astype('category')

# Pobreza
pobreza_labels = {
    1: 'poverty',
    2: 'poverty',
    3: 'no poverty'
}
df['pobreza_cat'] = df['pobreza'].map(pobreza_labels).astype('category')

# Renaming
rename_dict = {
    # Household identifiers (you may keep or drop later)
    'conglome': 'cluster_id',
    'vivienda': 'dwelling_id',
    'hogar': 'household_id',
    'ubigeo': 'geo_code',
    # Geography
    'dominio': 'domain_code',
    'dominio_cat': 'domain',
    'estrato': 'stratum',
    'area': 'area_type',
    # Household composition
    'percepho': 'income_receivers',
    'totmieho': 'household_total_size',
    'mieperho': 'household_members',
    'dependencia': 'dependency_ratio',
    # Log-transformed
    'log_ia01hd': 'log_imputed_rent_income',
    'log_ga04hd': 'log_imputed_rent_expense',
    'log_ingmo1hd': 'log_monetary_income_gross',
    'log_ingmo2hd': 'log_monetary_income_net',
    'log_inghog1d': 'log_household_income_gross',
    'log_inghog2d': 'log_household_income_net',
    'log_gashog1d': 'log_household_expense_monetary',
    'log_gashog2d': 'log_household_expense_total',
    # Poverty
    'pobreza': 'poverty_code',
    'pobreza_cat': 'poverty_status',
    # Time
    'year': 'year'
}
df = df.rename(columns=rename_dict)

# Target balance
df['poverty_status'].value_counts(dropna=False)
df['poverty_status'].value_counts(normalize=True)

# Modeling
# ==================================================
df_model = df.drop(
    columns=['cluster_id', 'dwelling_id', 'household_id', 'geo_code']
)
df_model['poverty_binary'] = (df_model['poverty_status'] == 'poverty').astype(int)

model_columns = [
    'income_receivers',
    'household_total_size',
    'household_members',
    'dependency_ratio',
    'log_imputed_rent_income',
    'log_imputed_rent_expense',
    'log_household_income_gross',
    'log_household_expense_total',
    'area_type',
    'domain',
    'poverty_binary'
]
df_model = df_model[model_columns]

X, y = df_model.drop(columns='poverty_binary'), df_model['poverty_binary']
X = pd.get_dummies(X, drop_first=True)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

print(rf.score(X_test, y_test))
print(classification_report(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_prob))

cm = confusion_matrix(y_test, y_pred)
labels = ('no poverty', 'poverty')
ConfusionMatrixDisplay(cm, display_labels=labels).plot()


import pandas as pd

feature_importance = (
    pd.Series(rf.feature_importances_, index=X.columns)
      .sort_values(ascending=False)
)

feature_importance.head(10)
feature_importance.plot()


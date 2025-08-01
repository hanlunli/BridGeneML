import pandas as pd
import re
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

df1 = pd.read_csv("ipo_file1.csv", dtype=str)
df2 = pd.read_csv("ipo_file2.csv", header=0, index_col=False, dtype=str)

df2 = df2.loc[:, ~df2.columns.str.startswith("Unnamed")]

if len(df2.columns) == len(df1.columns):
    df2.columns = df1.columns
    print("✓ Aligned df2 columns with df1")

def clean_cols(df):
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.replace(r"[\n,]", " ", regex=True)
          .str.replace(r"\s+", " ", regex=True)
          .str.replace(" ", "_")
    )
    return df

df1 = clean_cols(df1)
df2 = clean_cols(df2)

rename_map = {
    "Company_Name": "Company",
    "Company": "Company",
    "Ticker": "Ticker",
    "Offer_Price_(USD)": "Offer_Price",
    "Offer_Price": "Offer_Price",
    "Stock_Price_May_1_2025_(USD)": "Price_May1",
    "Stock_Price_Jun_2_2025_(USD)": "Price_Jun2",
    "Stock_Price_Jul_1_2025_(USD)": "Price_Jul1",
    "Current_Stock_Price_as_of_May_1_2025": "Price_May1",
    "Current_Stock_Price_as_of_June_2_2025": "Price_Jun2",
    "Current_Stock_Price_as_of_July_1_2025": "Price_Jul1",
    "Gross_Proceeds_(USD_M)": "Gross_Proceeds",
    "Gross_Proceeds_(USD_in_million)": "Gross_Proceeds",
    "Market_Cap_at_IPO_(USD_M)": "MarketCap_IPO",
    "Market_Cap_as_of_May_1_2025_(USD_M)": "MarketCap_May1",
    "Market_Capitalization_at_the_IPO_Date": "MarketCap_IPO",
    "Stage_at_IPO": "Clinical_Stage",
    "Lead_Candidate_Clinical_Stage_at_Time_of_IPO": "Clinical_Stage",
    "Therapeutic_Sector": "Therapeutic_Sector",
    "IPO_Date": "IPO_Date"
}

df1 = df1.rename(columns=rename_map)
df2 = df2.rename(columns=rename_map)

df = pd.concat([df1, df2], ignore_index=True)

def parse_ipo_date(date_str):
    if pd.isna(date_str): 
        return pd.NaT
    if not isinstance(date_str, str):
        date_str = str(date_str)
    for fmt in ("%b %d, %Y", "%m/%d/%Y"):
        try:
            return pd.to_datetime(date_str.strip(), format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(date_str, errors="coerce")

df['IPO_Date_Parsed'] = df['IPO_Date'].apply(parse_ipo_date)
today = datetime.now()
df['Days_Since_IPO'] = (today - df['IPO_Date_Parsed']).dt.days

to_numeric_cols = [
    "Offer_Price", "Gross_Proceeds", "MarketCap_IPO",
    "MarketCap_May1", "Price_May1", "Price_Jun2", "Price_Jul1"
]
for c in to_numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(
            df[c].astype(str)
                 .str.replace(r"[\$,~]", "", regex=True)
                 .str.replace("nan", ""),
            errors="coerce"
        )

def extract_clinical_phase(stage):
    if pd.isna(stage):
        return None
    s = str(stage).lower()
    if 'preclinical' in s:
        return 0
    if 'marketed' in s or 'approved' in s:
        return 4
    if 'pivotal' in s:
        return 3.5
    if 'ind' in s and 'enabling' in s:
        return 0.5
    for token, val in [('2b', 2.5), ('1b', 1.5)]:
        if token in s:
            return val
    match = re.search(r"\d+", s)
    return float(match.group()) if match else None

df['Clinical_Phase_Numeric'] = df['Clinical_Stage'].apply(extract_clinical_phase)

df.dropna(subset=['Offer_Price', 'Price_Jul1'], inplace=True)
features_all = [
    'Gross_Proceeds', 'MarketCap_IPO', 'Price_May1',
    'Clinical_Phase_Numeric', 'Days_Since_IPO'
]
df.dropna(subset=features_all, how='all', inplace=True)

df['y_up'] = (df['Price_Jul1'] > df['Offer_Price']).astype(int)

features = [f for f in features_all if df[f].notna().any()]
X = df[features].copy()
y = df['y_up']
mask = X.notna().all(axis=1)
X, y = X[mask], y[mask]

print("Training class distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42))
])
pipeline.fit(X_train, y_train)

print(classification_report(y_test, pipeline.predict(X_test), zero_division=0))

def predict_from_csv(file_path):
    df_new = pd.read_csv(file_path, dtype=str)
    df_new = clean_cols(df_new)
    df_new = df_new.rename(columns=rename_map)

    for c in to_numeric_cols:
        if c in df_new.columns:
            df_new[c] = pd.to_numeric(
                df_new[c].astype(str)
                        .str.replace(r"[\$,~]", "", regex=True)
                        .str.replace("nan", ""),
                errors="coerce"
            )
    df_new['IPO_Date_Parsed'] = df_new['IPO_Date'].apply(parse_ipo_date)
    df_new['Days_Since_IPO'] = (datetime.now() - df_new['IPO_Date_Parsed']).dt.days

    df_new['Clinical_Phase_Numeric'] = df_new['Clinical_Stage'].apply(extract_clinical_phase)

    X_new = df_new[features].copy()

    mask_new = X_new.notna().all(axis=1)
    X_new = X_new[mask_new]
    if X_new.empty:
        print("No valid rows to predict after cleaning.")
        return

    preds = pipeline.predict(X_new)
    probs = pipeline.predict_proba(X_new)[:, 1]

    for idx, company in enumerate(df_new.loc[mask_new, 'Company']):
        direction = 'increase' if preds[idx] == 1 else 'decrease'
        print(f"{company}: Predicted stock price will {direction} (probability of increase: {probs[idx]:.2%})")

if __name__ == '__main__':
    predict_from_csv("input.csv")

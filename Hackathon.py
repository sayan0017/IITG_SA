# 📦 Import required libraries
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 📂 Load datasets
train_df = pd.read_csv("/kaggle/input/summer-analytics-mid-hackathon/hacktrain.csv")
test_df = pd.read_csv("/kaggle/input/summer-analytics-mid-hackathon/hacktest.csv")

# 🧹 Drop unnecessary column
train_df.drop(columns=['Unnamed: 0'], inplace=True)
test_df.drop(columns=['Unnamed: 0'], inplace=True)

# 🏷️ Separate target and features
X_raw = train_df.drop(columns=['class', 'ID'])
y = train_df['class']
test_raw = test_df.drop(columns=['ID'])

train_ids = train_df['ID']
test_ids = test_df['ID']

# 🧼 Handle missing values by imputing column mean
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)
test_imputed = pd.DataFrame(imputer.transform(test_raw), columns=test_raw.columns)

# 🧠 Feature engineering function
def extract_features(df):
    return pd.DataFrame({
        'ndvi_mean': df.mean(axis=1),
        'ndvi_std': df.std(axis=1),
        'ndvi_min': df.min(axis=1),
        'ndvi_max': df.max(axis=1),
        'ndvi_median': df.median(axis=1),
        'ndvi_range': df.max(axis=1) - df.min(axis=1),
        'ndvi_first': df.iloc[:, 0],
        'ndvi_last': df.iloc[:, -1],
        'ndvi_trend': df.iloc[:, -1] - df.iloc[:, 0],
    })

# 🧪 Extract features
X_features = extract_features(X_imputed)
test_features = extract_features(test_imputed)

# 🔢 Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 🧪 Optional: Train-validation split for local testing
X_train, X_val, y_train, y_val = train_test_split(
    X_features, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 🔁 Train logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
model.fit(X_train, y_train)

# 📊 Evaluate model on validation set
val_preds = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("\nClassification Report:\n", classification_report(y_val, val_preds, target_names=label_encoder.classes_))

# ✅ Retrain on full data
model_final = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
model_final.fit(X_features, y_encoded)

# 🔮 Predict on test set
test_preds = model_final.predict(test_features)
test_preds_labels = label_encoder.inverse_transform(test_preds)

# 📄 Create submission DataFrame
submission_df = pd.DataFrame({
    'ID': test_ids,
    'class': test_preds_labels
})

# 💾 Save to CSV
submission_df.to_csv("submission.csv", index=False)

print("✅ submission.csv generated!")

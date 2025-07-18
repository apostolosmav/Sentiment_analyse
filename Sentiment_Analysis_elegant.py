import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# ----------------------------
# 1. Dateneinlesung 
# ----------------------------

# CSV-Datei lessen
df = pd.read_csv("C:\\Users\\velpTEC edutainment\\Downloads\\sentiment_analysis.csv")

# ----------------------------
# 2. Daten Vorbereiten mit Vectorizer
# ----------------------------
# Nans aus Dataframe entfernen
df_clean = df.dropna(subset=['text','sentiment'])
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000,lowercase=True) # stopwords entfernen
X = vectorizer.fit_transform(df_clean['text'])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_clean['sentiment']) # in numerische Werte konvertieren

# ----------------------------
# 3. Daten trainieren & modellieren
# ----------------------------

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
lr = LogisticRegression()
lr.fit(X_train,y_train) 

# ----------------------------
# 4. Vorhersage & Model bewerten
# ----------------------------
y_pred = lr.predict(X_test)
accuracy = lr.score(X_test,y_test)
print("Vorhersagen:", y_pred)
print("Genauigkeit:", accuracy)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# ----------------------------
# 4. Ergebnisse ploten
# ----------------------------
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

ConfusionMatrixDisplay.from_estimator(lr, X_test, y_test, display_labels=label_encoder.classes_, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
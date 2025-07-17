import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# CSV-Datei laden
df = pd.read_csv("C:\\Users\\velpTEC edutainment\\Downloads\\sentiment_analysis.csv")

#  Vokabular erstellen 
def get_vokabular(df):
    vokabular = []
    for satz in df['text']: 
        worte = re.findall(r'\b\w+\b', satz.lower()) 
        vokabular.extend(worte)
    vokabular = sorted(set(vokabular))
    return vokabular

# BoW-Matrix erzeugen
def get_bow(df, vokabular):
    bow = []
    for satz in df['text']:
        tokens = re.findall(r'\b\w+\b', satz.lower())
        satz_bow = [1 if wort in tokens else 0 for wort in vokabular]
        bow.append(satz_bow)
    return bow

# Bag-of-Words vorbereiten
df = df.dropna(subset=['text']) # Nans entfernen sont kommt AttributeError: 'float' object has no attribute 'lower'
vokabular = get_vokabular(df)
bow = get_bow(df, vokabular)
X = pd.DataFrame(bow, columns=vokabular)
print(X)

#  Labels encodieren 
y = df['sentiment']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Modelltraining 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Modellbewertung 
score = clf.score(X_test, y_test)
print(f"Genauigkeit: {score:.2f}")
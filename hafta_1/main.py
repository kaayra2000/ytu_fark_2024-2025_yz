import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Veri kümesini yükleme
df = pd.read_csv('turkish_movie_sentiment_dataset.csv')

# Puan sütunundaki virgülleri nokta ile değiştirme ve float'a çevirme
df['point'] = df['point'].str.replace(',', '.').astype(float)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(df['comment'], df['point'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer'ı oluşturma (trigram tabanlı)
vectorizer = TfidfVectorizer(ngram_range=(1, 3))

# Eğitim verilerini vektöre dönüştürme
X_train_vectorized = vectorizer.fit_transform(X_train)

# Test verilerini vektöre dönüştürme
X_test_vectorized = vectorizer.transform(X_test)

# Linear Regression modelini oluşturma ve eğitme
model = LinearRegression()
model.fit(X_train_vectorized, y_train)

# Test seti üzerinde tahminler yapma
y_pred = model.predict(X_test_vectorized)

# Model performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Ortalama Kare Hata: {mse:.4f}')
print(f'R-kare Skoru: {r2:.4f}')

# Yeni bir yorum için tahmin yapma
new_comment = "Bu film gerçekten harikaydı, kesinlikle tavsiye ederim!"
new_comment_vectorized = vectorizer.transform([new_comment])
predicted_score = model.predict(new_comment_vectorized)

print(f'Tahmini Puan: {predicted_score[0]:.2f}')

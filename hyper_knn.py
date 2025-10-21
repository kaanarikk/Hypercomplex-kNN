import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Hiperkompleks sayı sınıfımızı kendi dosyasından import ediyoruz
from hypercomplex_class import HypercomplexNumber

# --- UZAKLIK FONKSİYONLARI ---

def euclidean_distance(point1, point2):
    """İki nokta arasındaki Öklid uzaklığını hesaplar."""
    return np.sqrt(np.sum((point1 - point2)**2))

def hypercomplex_distance(point1, point2, p):
    """İki nokta arasındaki hiperkompleks normu uzaklık olarak kullanır."""
    h1 = HypercomplexNumber(list(point1), p)
    h2 = HypercomplexNumber(list(point2), p)
    distance = (h1 - h2).norm()
    if isinstance(distance, complex):
        return abs(distance)
    return distance

# --- MANUEL KNN ALGORİTMASI ---

def predict_single_point(X_train, y_train, test_point, k, distance_func, p=None):
    """Tek bir test noktası için en yakın k komşuyu bulur ve sınıfını tahmin eder."""
    distances = []
    for i, train_point in enumerate(X_train):
        if p is not None:
            dist = distance_func(train_point, test_point, p)
        else:
            dist = distance_func(train_point, test_point)
        distances.append((dist, y_train[i]))
    
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    neighbor_labels = [neighbor[1] for neighbor in neighbors]
    most_common = Counter(neighbor_labels).most_common(1)
    
    return most_common[0][0]

def knn_predict(X_train, y_train, X_test, k, distance_func, p=None):
    """Tüm test seti için KNN tahminleri yapar."""
    predictions = []
    for test_point in X_test:
        pred = predict_single_point(X_train, y_train, test_point, k, distance_func, p)
        predictions.append(pred)
    return np.array(predictions)


# --- ANA TEST KODU ---
if __name__ == '__main__':
    # Ayarlar
    K = 5  # En yakın komşu sayısı

    # 1. VERİ SETİ YÜKLEME VE İŞLEME
    print("--- Veri Seti Yükleniyor ve İşleniyor ---")
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
    column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    df = pd.read_csv(url, names=column_names)
    X = df.drop('class', axis=1)
    y = df['class']
    X_encoded = OrdinalEncoder().fit_transform(X)
    y_encoded = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("Veri seti hazır.\n")
    
    # 2. ÖKLİD UZAKLIĞI İLE KNN TESTİ (SABİT KARŞILAŞTIRMA NOKTASI)
    print(f"--- Öklid Uzaklığı ile KNN Testi (k={K}) ---")
    euclidean_preds = knn_predict(X_train, y_train, X_test, K, euclidean_distance)
    euclidean_accuracy = accuracy_score(y_test, euclidean_preds)
    print(f"Öklid KNN Başarımı: {euclidean_accuracy:.4f}\n")

    # 3. FARKLI p DEĞERLERİ İLE HİPERKOMPLEKS NORMU TESTİ
    print(f"--- Farklı p Değerleri için Hiperkompleks KNN Testi (k={K}) ---")
    
    p_values = np.linspace(-1.1, -0.9, 10)
    accuracies = []
    
    # Optimum değeri bulmak için değişkenler
    best_accuracy = -1.0
    optimal_p = None

    print("p değerleri test ediliyor...")
    for p_val in p_values:
        hypercomplex_preds = knn_predict(X_train, y_train, X_test, K, hypercomplex_distance, p=p_val)
        accuracy = accuracy_score(y_test, hypercomplex_preds)
        accuracies.append(accuracy)
        
        # En iyi sonucu kontrol et ve güncelle
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            optimal_p = p_val
    
    print("Test tamamlandı.\n")

    # En iyi sonucu yazdır
    print("--- Optimum Sonuç ---")
    print(f"En Yüksek Başarım: {best_accuracy:.4f}")
    print(f"Bu Başarımı Sağlayan p Değeri: {optimal_p:.4f}\n")

    # 4. SONUÇLARI GÖRSELLEŞTİRME
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    
    plt.plot(p_values, accuracies, marker='o', linestyle='-', markersize=5, label='Hiperkompleks Norm Başarımı')
    
    # Karşılaştırma için Öklid başarımını sabit bir çizgi olarak ekle
    plt.axhline(y=euclidean_accuracy, color='r', linestyle='--', label=f'Öklid Normu Başarımı ({euclidean_accuracy:.4f})')
    
    plt.title('p Parametresinin Değişiminin KNN Başarımına Etkisi', fontsize=16)
    plt.xlabel('p Değeri', fontsize=12)
    plt.ylabel('Doğruluk (Accuracy)', fontsize=12)
    plt.legend(fontsize=10)
    plt.ylim(bottom=min(accuracies) - 0.02, top=max(accuracies) + 0.02)
    plt.gca().invert_xaxis()
    
    print("Grafik oluşturuluyor. Lütfen grafik penceresini kontrol edin.")
    plt.show()


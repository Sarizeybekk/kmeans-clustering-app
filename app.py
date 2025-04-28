
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score
import plotly.graph_objects as go

# Başlık
st.title("Müşteri Segmentasyonu ve Davies-Bouldin Analizi")

# 1. CSV Yükleme
uploaded_file = st.file_uploader("CSV dosyası yükle", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Yüklenen Veri:")
    st.dataframe(data.head())

    # 2. Kullanılacak Özellik Seçimi
    columns = st.multiselect("Kullanılacak Özellikleri Seçin:", data.columns, default=['Annual Income (k$)', 'Spending Score (1-100)'])

    if len(columns) >= 2:
        X = data[columns]

        # 3. Veriyi Ölçekleme
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 4. K Seçimi
        k = st.slider("Kaç Küme (K) seçmek istersin?", min_value=2, max_value=10, value=5)

        # 5. KMeans Kümeleme
        if st.button("Kümeleri Oluştur"):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X_scaled)

            # Davies-Bouldin Skoru
            db_index = davies_bouldin_score(X_scaled, labels)

            st.success(f"Davies-Bouldin Skoru (K={k}): {db_index:.3f}")

            # 6. Kümeleme Grafiği (2D Scatter)
            plt.figure(figsize=(8,6))
            plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels, cmap='viridis', alpha=0.6)
            plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=300, c='red', marker='X')
            plt.xlabel(columns[0])
            plt.ylabel(columns[1])
            plt.title('K-Means Kümeleme Sonucu')
            st.pyplot(plt)

            # 7. Davies-Bouldin Gauge Grafiği
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = db_index,
                title = {'text': f"Davies-Bouldin Skoru (K={k})"},
                gauge = {
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "green"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgreen"},
                        {'range': [0.3, 0.6], 'color': "yellow"},
                        {'range': [0.6, 1], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': db_index
                    }
                }
            ))
            st.plotly_chart(fig)

    else:
        st.warning("Lütfen en az 2 özellik seçin!")

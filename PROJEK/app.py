import streamlit as st
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

# =========================
# KONFIGURASI AWAL
# =========================
st.set_page_config(
    page_title="PCA Image Compression",
    layout="wide"
)

st.title("📉 PCA untuk Kompresi Citra (Grayscale)")
st.write("Aplikasi EDA dan kompresi citra menggunakan **Principal Component Analysis (PCA)**")

# =========================
# UPLOAD GAMBAR
# =========================
uploaded_file = st.file_uploader(
    "Upload gambar (jpg / png / jpeg)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # =========================
    # BACA & PREPROCESS GAMBAR
    # =========================
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = img_gray.shape

    # =========================
    # EDA AWAL
    # =========================
    st.header("🔍 EDA Awal")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img_gray, caption="Citra Grayscale", clamp=True)

    with col2:
        fig, ax = plt.subplots()
        ax.hist(img_gray.flatten(), bins=256, range=(0, 255))
        ax.set_title("Histogram Intensitas Pixel")
        ax.set_xlabel("Intensitas")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

    st.subheader("📊 Statistik Dasar")
    st.write(f"**Ukuran citra** : {h} x {w}")
    st.write(f"**Rentang pixel** : {img_gray.min()} – {img_gray.max()}")
    st.write(f"**Mean (Kecerahan)** : {np.mean(img_gray):.2f}")
    st.write(f"**Std Dev (Kontras)** : {np.std(img_gray):.2f}")

    st.info(
        "📌 Interpretasi: "
        "Mean menunjukkan tingkat kecerahan rata-rata, "
        "sedangkan standar deviasi menunjukkan kontras citra."
    )

    # =========================
    # NORMALISASI / CENTERING
    # =========================
    st.header("➗ Normalisasi / Centering Data")

    mean_vector = np.mean(img_gray, axis=0)
    X_centered = img_gray - mean_vector

    st.latex(r"X_{centered} = X - \mu")

    # =========================
    # KOVARIANS
    # =========================
    st.header("📐 Matriks Kovarians")

    cov_matrix = np.cov(X_centered, rowvar=False)

    st.latex(r"\Sigma = \frac{1}{n-1} X^T X")

    # =========================
    # EIGENVALUE & EIGENVECTOR
    # =========================
    st.header("🔢 Eigenvalue & Eigenvector")

    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    explained_variance = eigenvalues / np.sum(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance)

    # =========================
    # TABEL PC1 - PC10
    # =========================
    st.subheader("📋 Tabel Komponen Utama (PC1–PC10)")

    makna = []
    for ev in explained_variance[:10]:
        if ev > 0.2:
            makna.append("Informasi sangat besar")
        elif ev > 0.1:
            makna.append("Informasi besar")
        elif ev > 0.05:
            makna.append("Informasi sedang")
        else:
            makna.append("Informasi kecil")

    df_pc = pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(10)],
        "Eigenvalue": eigenvalues[:10],
        "Explained Variance (%)": explained_variance[:10] * 100,
        "Makna": makna
    })

    st.dataframe(df_pc)

    # =========================
    # GRAFIK EIGENVALUE
    # =========================
    st.subheader("📈 Top 10 Eigenvalue")

    fig, ax = plt.subplots()
    ax.bar(range(1, 11), explained_variance[:10] * 100)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    st.pyplot(fig)

    # =========================
    # SCREE & CUMULATIVE
    # =========================
    st.header("📊 Scree Plot & Cumulative Explained Variance")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.plot(eigenvalues[:100])
        ax.set_title("Scree Plot (Top 100)")
        ax.set_xlabel("PC")
        ax.set_ylabel("Eigenvalue")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.plot(cumulative_variance * 100)
        ax.set_title("Cumulative Explained Variance")
        ax.set_xlabel("Jumlah PC (k)")
        ax.set_ylabel("Kumulatif Informasi (%)")
        st.pyplot(fig)

    # =========================
    # PILIH K
    # =========================
    st.header("🎯 Pilih Jumlah Komponen Utama (k)")
    k = st.slider("Nilai k", 1, min(200, w), 50)

    # =========================
    # PROYEKSI & REKONSTRUKSI
    # =========================
    Wk = eigenvectors[:, :k]
    Z = X_centered @ Wk
    X_reconstructed = (Z @ Wk.T) + mean_vector
    X_reconstructed = np.clip(X_reconstructed, 0, 255)

    # =========================
    # EVALUASI
    # =========================
    st.header("📏 Evaluasi Kualitas Kompresi")

    mse = mean_squared_error(img_gray, X_reconstructed)
    psnr = peak_signal_noise_ratio(img_gray, X_reconstructed, data_range=255)
    ssim = structural_similarity(img_gray, X_reconstructed, data_range=255)

    size_original = h * w
    size_compressed = h * k + k * w
    compression_ratio = (1 - size_compressed / size_original) * 100

    st.write(f"**Explained Variance** : {cumulative_variance[k-1]*100:.2f}%")
    st.write(f"**MSE** : {mse:.2f}")
    st.write(f"**PSNR** : {psnr:.2f} dB")
    st.write(f"**SSIM** : {ssim:.4f}")
    st.write(f"**Penghematan ukuran** : {compression_ratio:.2f}%")

    # =========================
    # TABEL EVALUASI MULTI-K
    # =========================
    st.subheader("📋 Tabel Evaluasi Beberapa Nilai k")

    k_values = [5, 10, 20, 50, 100]
    rows = []

    for kv in k_values:
        Wk = eigenvectors[:, :kv]
        Z = X_centered @ Wk
        Xr = (Z @ Wk.T) + mean_vector
        Xr = np.clip(Xr, 0, 255)

        rows.append([
            kv,
            cumulative_variance[kv-1] * 100,
            mean_squared_error(img_gray, Xr),
            peak_signal_noise_ratio(img_gray, Xr, data_range=255),
            structural_similarity(img_gray, Xr, data_range=255)
        ])

    df_eval = pd.DataFrame(
        rows,
        columns=["k", "Explained Variance (%)", "MSE", "PSNR", "SSIM"]
    )

    st.dataframe(df_eval)

    # =========================
    # PERBANDINGAN VISUAL
    # =========================
    st.header("🖼️ Perbandingan Visual")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img_gray, caption="Citra Asli", clamp=True)

    with col2:
        st.image(X_reconstructed, caption=f"Rekonstruksi (k={k})", clamp=True)

    with col3:
        st.image(np.abs(img_gray - X_reconstructed), caption="Error Image", clamp=True)

    # =========================
    # RINGKASAN
    # =========================
    st.header("📌 Ringkasan PCA & EDA")

    st.success(
        "PCA memanfaatkan eigenvalue dan eigenvector untuk "
        "menyimpan informasi terbesar citra. "
        "EDA membantu memahami distribusi data sebelum dan sesudah kompresi."
    )
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title="PCA Image Compression", layout="wide")

# =========================================================
# FUNGSI BANTUAN
# =========================================================
def psnr(original, reconstructed):
    mse = mean_squared_error(original.flatten(), reconstructed.flatten())
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def interpret_eigen(pct):
    if pct >= 20:
        return "Informasi sangat besar"
    elif pct >= 10:
        return "Informasi besar"
    elif pct >= 5:
        return "Informasi sedang"
    else:
        return "Informasi kecil"

# =========================================================
# JUDUL
# =========================================================
st.title("📉 PCA untuk Kompresi Citra Digital")
st.write("Exploratory Data Analysis (EDA), PCA, dan Evaluasi Kompresi")

# =========================================================
# UPLOAD GAMBAR
# =========================================================
uploaded_file = st.file_uploader(
    "Upload gambar (jpg/png/jpeg)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # =====================================================
    # BACA GAMBAR
    # =====================================================
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)

    h, w = img_gray.shape

    # =====================================================
    # EDA AWAL
    # =====================================================
    st.header("1️⃣ EDA Awal")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_color, caption="Citra Asli", use_column_width=True)
    with col2:
        st.image(img_gray, caption="Grayscale", use_column_width=True)

    st.subheader("📊 Informasi Statistik Citra")
    st.write(f"**Ukuran Citra:** {h} x {w}")
    st.write(f"**Rentang Pixel:** {img_gray.min()} - {img_gray.max()}")
    st.write(f"**Mean (Kecerahan):** {img_gray.mean():.2f}")
    st.write(f"**STD Dev (Kontras):** {img_gray.std():.2f}")

    st.info(
        "Citra memiliki distribusi intensitas tertentu. "
        "Mean menunjukkan tingkat kecerahan, sedangkan standar deviasi "
        "menunjukkan kontras citra."
    )

    # Histogram
    fig, ax = plt.subplots()
    ax.hist(img_gray.flatten(), bins=256)
    ax.set_xlabel("Intensitas Pixel")
    ax.set_ylabel("Frekuensi")
    st.pyplot(fig)

    # =====================================================
    # NORMALISASI / CENTERING DATA
    # =====================================================
    st.header("2️⃣ Normalisasi / Centering Data")

    X = img_gray.astype(np.float64)
    mean_vector = np.mean(X, axis=0)
    X_centered = X - mean_vector

    st.latex(r"X_{centered} = X - \mu")
    st.write("Data dikurangi rata-rata setiap kolom (centering).")

    # =====================================================
    # PCA
    # =====================================================
    st.header("3️⃣ PCA dan Eigen Analysis")

    pca = PCA()
    pca.fit(X_centered)

    eigenvalues = pca.explained_variance_
    explained_var = pca.explained_variance_ratio_ * 100
    eigenvectors = pca.components_

    # =====================================================
    # TABEL EIGEN (PC1 - PC10)
    # =====================================================
    st.subheader("📋 Tabel Eigenvalue & Eigenvector (PC1 - PC10)")

    data_eigen = []
    for i in range(10):
        data_eigen.append([
            f"PC{i+1}",
            eigenvalues[i],
            explained_var[i],
            interpret_eigen(explained_var[i])
        ])

    df_eigen = pd.DataFrame(
        data_eigen,
        columns=["Komponen", "Eigenvalue", "Persentase Informasi (%)", "Makna"]
    )
    st.dataframe(df_eigen)

    # =====================================================
    # SCREE PLOT & CUMULATIVE
    # =====================================================
    st.header("4️⃣ Scree Plot & Cumulative Explained Variance")

    cumulative = np.cumsum(explained_var)

    fig, ax = plt.subplots()
    ax.plot(explained_var[:100], marker='o')
    ax.set_xlabel("Komponen Utama (PC)")
    ax.set_ylabel("Eigenvalue (%)")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(cumulative[:100], marker='o')
    ax.set_xlabel("Jumlah PC (K)")
    ax.set_ylabel("Kumulatif Informasi (%)")
    st.pyplot(fig)

    # =====================================================
    # PILIH K
    # =====================================================
    st.header("5️⃣ Rekonstruksi & Kompresi")
    k = st.slider("Pilih jumlah komponen utama (K)", 1, min(200, w), 50)

    pca_k = PCA(n_components=k)
    Z = pca_k.fit_transform(X_centered)
    X_reconstructed = pca_k.inverse_transform(Z) + mean_vector

    # =====================================================
    # EVALUASI
    # =====================================================
    mse_val = mean_squared_error(X, X_reconstructed)
    psnr_val = psnr(X, X_reconstructed)
    ssim_val = ssim(X, X_reconstructed, data_range=255)

    info_retained = np.sum(pca_k.explained_variance_ratio_) * 100
    compression_ratio = (k * (h + w)) / (h * w) * 100

    st.subheader("📊 Evaluasi Kualitas Kompresi")
    st.write(f"**Informasi Dipertahankan:** {info_retained:.2f}%")
    st.write(f"**MSE:** {mse_val:.2f}")
    st.write(f"**PSNR:** {psnr_val:.2f} dB")
    st.write(f"**SSIM:** {ssim_val:.4f}")
    st.write(f"**Rasio Kompresi:** {compression_ratio:.2f}%")

    # =====================================================
    # TABEL EVALUASI BEBERAPA K
    # =====================================================
    st.header("6️⃣ Tabel Evaluasi Beberapa Nilai K")

    k_values = [5, 10, 20, 50, 100]
    eval_data = []

    for kv in k_values:
        if kv < w:
            pca_temp = PCA(n_components=kv)
            Zt = pca_temp.fit_transform(X_centered)
            Xt = pca_temp.inverse_transform(Zt) + mean_vector

            mse_t = mean_squared_error(X, Xt)
            psnr_t = psnr(X, Xt)
            ssim_t = ssim(X, Xt, data_range=255)
            info_t = np.sum(pca_temp.explained_variance_ratio_) * 100
            ratio_t = (kv * (h + w)) / (h * w) * 100

            eval_data.append([kv, info_t, mse_t, psnr_t, ssim_t, ratio_t])

    df_eval = pd.DataFrame(
        eval_data,
        columns=["K", "Explained Variance (%)", "MSE", "PSNR", "SSIM", "Rasio Kompresi (%)"]
    )
    st.dataframe(df_eval)

    # =====================================================
    # EDA SETELAH KOMPRESI
    # =====================================================
    st.header("7️⃣ EDA Setelah Kompresi")

    error_img = np.abs(X - X_reconstructed)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(X, caption="Citra Asli", use_column_width=True, clamp=True)
    with col2:
        st.image(X_reconstructed, caption=f"Rekonstruksi (K={k})", use_column_width=True, clamp=True)
    with col3:
        st.image(error_img, caption="Error Image", use_column_width=True, clamp=True)

    fig, ax = plt.subplots()
    ax.hist(X.flatten(), bins=256, alpha=0.5, label="Asli")
    ax.hist(X_reconstructed.flatten(), bins=256, alpha=0.5, label="Rekonstruksi")
    ax.legend()
    st.pyplot(fig)

    # =====================================================
    # RINGKASAN
    # =====================================================
    st.header("8️⃣ Ringkasan")
    st.write("""
    - PCA mereduksi dimensi citra dengan memilih eigenvalue terbesar  
    - Eigenvalue menunjukkan seberapa besar informasi yang dibawa  
    - Eigenvector adalah arah komponen utama  
    - EDA membantu memahami distribusi data sebelum & sesudah kompresi  
    - Nilai K menentukan trade-off antara kualitas dan ukuran data  
    """)
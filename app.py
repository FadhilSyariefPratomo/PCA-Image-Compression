import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
import math

st.set_page_config(page_title="PCA Image Compression", layout="wide")

st.title("📉 PCA Image Compression & EDA")

# =========================
# UPLOAD IMAGE
# =========================
uploaded_file = st.file_uploader(
    "Upload gambar (jpg / png / jpeg)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img_rgb = Image.open(uploaded_file).convert("RGB")
    img_gray = img_rgb.convert("L")
    img = np.array(img_gray).astype(np.float64)

    h, w = img.shape

    # =========================
    # EDA AWAL
    # =========================
    st.header("📊 EDA Awal")

    pixel_min, pixel_max = img.min(), img.max()
    mean_val, std_val = img.mean(), img.std()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ukuran Citra", f"{h} x {w}")
    col2.metric("Rentang Piksel", f"{int(pixel_min)} – {int(pixel_max)}")
    col3.metric("Mean (Kecerahan)", f"{mean_val:.2f}")
    col4.metric("Std Dev (Kontras)", f"{std_val:.2f}")

    if std_val < 40:
        st.info("Kontras rendah")
    elif std_val < 80:
        st.info("Kontras sedang")
    else:
        st.success("Kontras tinggi")

    # Visual + Histogram
    fig1, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Citra Grayscale")
    ax[0].axis("off")

    ax[1].hist(img.flatten(), bins=256, color="gray")
    ax[1].set_title("Histogram Citra Asli")
    ax[1].set_xlabel("Intensitas Piksel")
    ax[1].set_ylabel("Frekuensi")

    st.pyplot(fig1)

    # =========================
    # CENTERING DATA
    # =========================
    st.header("📐 Normalisasi / Centering Data")

    mean_vector = np.mean(img, axis=0)
    X_centered = img - mean_vector

    st.latex(r"X_{centered} = X - \mu")

    # =========================
    # KOVARIANS & EIGEN
    # =========================
    st.header("🧮 Matriks Kovarians & Eigen")

    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    explained_var = eigenvalues / np.sum(eigenvalues) * 100

    # =========================
    # TABEL PC1–PC10
    # =========================
    makna = []
    for val in explained_var[:10]:
        if val > 10:
            makna.append("Informasi sangat besar")
        elif val > 5:
            makna.append("Informasi besar")
        elif val > 1:
            makna.append("Informasi sedang")
        else:
            makna.append("Informasi kecil")

    df_pc = pd.DataFrame({
        "Komponen": [f"PC{i+1}" for i in range(10)],
        "Eigenvalue": eigenvalues[:10],
        "Persentase Informasi (%)": explained_var[:10],
        "Makna": makna
    })

    st.subheader("Tabel Eigenvalue & Makna Informasi")
    st.dataframe(df_pc, use_container_width=True)

    # =========================
    # GRAFIK PCA
    # =========================
    st.subheader("Visualisasi PCA")

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4))

    ax2[0].bar(range(1, 11), explained_var[:10])
    ax2[0].set_title("Top 10 Explained Variance")
    ax2[0].set_xlabel("PC")
    ax2[0].set_ylabel("% Informasi")

    ax2[1].plot(np.cumsum(explained_var[:100]), marker="o")
    ax2[1].set_title("Cumulative Explained Variance")
    ax2[1].set_xlabel("Jumlah PC (k)")
    ax2[1].set_ylabel("Informasi Kumulatif (%)")

    st.pyplot(fig2)

    # =========================
    # PILIH K
    # =========================
    st.header("🎯 Rekonstruksi PCA")

    k = st.slider("Pilih jumlah komponen utama (k)", 1, min(200, w), 50)

    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X_centered)
    X_recon = pca.inverse_transform(X_pca) + mean_vector

    # =========================
    # EVALUASI
    # =========================
    mse_val = mean_squared_error(img, X_recon)
    psnr_val = 10 * math.log10((255 ** 2) / mse_val)
    ssim_val = ssim(img, X_recon, data_range=255)

    info_retained = np.sum(explained_var[:k])

    original_size = img.nbytes / 1024
    compressed_size = X_pca.nbytes / 1024
    saving = (1 - compressed_size / original_size) * 100

    st.subheader("Evaluasi Kualitas Kompresi")

    col1, col2, col3 = st.columns(3)
    col1.metric("Informasi Dipertahankan", f"{info_retained:.2f}%")
    col2.metric("PSNR", f"{psnr_val:.2f} dB")
    col3.metric("SSIM", f"{ssim_val:.4f}")

    st.write(f"**MSE:** {mse_val:.4f}")
    st.write(f"**Ukuran Asli:** {original_size:.2f} KB")
    st.write(f"**Ukuran Terkompresi:** {compressed_size:.2f} KB")
    st.write(f"**Penghematan Memori:** {saving:.2f}%")

    # =========================
    # ERROR IMAGE (FIX)
    # =========================
    error_img = np.abs(img - X_recon)
    error_img_norm = (error_img / error_img.max()) * 255
    error_img_norm = error_img_norm.astype(np.uint8)

    st.subheader("🖼️ Perbandingan Visual")

    c1, c2, c3 = st.columns(3)
    c1.image(img.astype(np.uint8), caption="Citra Asli", width=250)
    c2.image(X_recon.astype(np.uint8), caption="Rekonstruksi PCA", width=250)
    c3.image(error_img_norm, caption="Error Image", width=250)

    # =========================
    # KESIMPULAN
    # =========================
    st.header("📌 Ringkasan")

    st.markdown("""
    - PCA mereduksi dimensi citra dengan mempertahankan variansi terbesar.
    - Eigenvalue menunjukkan seberapa besar informasi tiap komponen utama.
    - Semakin besar nilai k, kualitas rekonstruksi meningkat namun kompresi berkurang.
    - Error image menampilkan selisih lokal antara citra asli dan hasil PCA.
    """)

    // test
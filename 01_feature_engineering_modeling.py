# """
# ==============================================================================
#  01_feature_engineering_modeling.py
# ==============================================================================
# Tugas Besar - Rekayasa Fitur
# Studi Kasus: Prediksi Harga Rumah (House Price Prediction)
# Dataset acuan: Kaggle "House Prices - Advanced Regression Techniques" (Ames Housing)

# FILE INI ADALAH FILE UTAMA (yang benar-benar mengerjakan tugas besar).
# Ia membutuhkan satu file input bernama "train.csv" di folder yang sama.

# DARI MANA train.csv BERASAL? (hubungan antar file)
#     Opsi A (tanpa dataset asli / untuk sekadar menguji kode):
#         jalankan dulu -> python3 00_generate_dataset.py
#         skrip itu akan MEMBUAT file train.csv (data simulasi) di folder ini.

#     Opsi B (dengan dataset asli, direkomendasikan untuk laporan final):
#         unduh train.csv dari
#         https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
#         lalu upload file "train.csv".
#         (Skema kolomnya sama, jadi tidak perlu ubah kode apa pun di file ini.)

# Setelah train.csv tersedia (lewat opsi A ATAU B), baru jalankan file ini:
# OPSI 1 :
#     python3 01_feature_engineering_modeling.py
# OPSI 2 : (Streamlit agar bisa dibuka di web)
#     python -m streamlit run 01_feature_engineering_modeling.py 

# TAHAPAN YANG DILAKUKAN FILE INI:
#     1. Memuat data (load_data)
#     2. Pra-pemrosesan / penanganan missing value (preprocess)
#     3. Rekayasa fitur: feature creation, binning, encoding (engineer_features)
#     4. Split data + scaling + feature selection (prepare_model_data)
#     5. Pemodelan: Linear Regression & Random Forest (train_and_evaluate)
#     6. Visualisasi hasil (make_plots)

# OUTPUT YANG DIHASILKAN (semua tersimpan di folder yang sama):
#     - model_results.csv         -> tabel perbandingan RMSE/MAE/R2 tiap model
#     - feature_importances.csv   -> skor pentingnya setiap fitur (dari Random Forest)
#     - plot_actual_vs_predicted.png
#     - plot_model_comparison.png
#     - plot_feature_importance.png
# ==============================================================================
# """


# KODE PROGRAM OPSI B (Streamlit)
# UNTUK KODE PROGRAM OPSI A ADA DIBAWAH KODE INI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Pengaturan halaman Streamlit
st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide")
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# FUNGSI CACHE UNTUK EFISIENSI STREAMLIT
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

@st.cache_data
def preprocess(df):
    df = df.copy()
    if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
        df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )
    if "GarageCars" in df.columns:
        df["HasGarage"] = (df["GarageCars"].fillna(0) > 0).astype(int)
    if "GarageYrBlt" in df.columns:
        df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].fillna("NA")

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

    return df

@st.cache_data
def engineer_features(df):
    df = df.copy()
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
    df["WasRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
    df["TotalSF"] = df["TotalBsmtSF"] + df["GrLivArea"]
    df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"]
    df["TotalPorchSF"] = df["WoodDeckSF"] + df["OpenPorchSF"]
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["QualCondScore"] = df["OverallQual"] * df["OverallCond"]

    df["HouseAgeBin"] = pd.cut(
        df["HouseAge"],
        bins=[-1, 5, 15, 30, 60, 200],
        labels=["Baru(0-5th)", "Muda(6-15th)", "Sedang(16-30th)", "Tua(31-60th)", "SangatTua(60+th)"],
    )

    qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
    quality_cols = ["ExterQual", "ExterCond", "KitchenQual", "BsmtQual", "BsmtCond",
                     "HeatingQC", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
    for col in quality_cols:
        if col in df.columns:
            df[col + "_enc"] = df[col].map(qual_map).fillna(0)
            df.drop(columns=[col], inplace=True)

    nominal_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    sisa_non_numerik = df_encoded.select_dtypes(exclude=[np.number, bool]).columns.tolist()
    sisa_non_numerik = [c for c in sisa_non_numerik if c not in ("Id", "SalePrice")]
    if sisa_non_numerik:
        df_encoded = df_encoded.drop(columns=sisa_non_numerik)

    drop_cols = ["Id", "SalePrice", "YearBuilt", "YearRemodAdd", "MoSold", "YrSold"]
    feature_cols = [c for c in df_encoded.columns if c not in drop_cols]
    X = df_encoded[feature_cols].copy()
    y = df_encoded["SalePrice"].copy()

    return X, y

@st.cache_data
def prepare_model_data(X, y, k=20):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_train_scaled, y_train)
    selected_features = X.columns[selector.get_support()].tolist()

    X_train_sel = selector.transform(X_train_scaled)
    X_test_sel = selector.transform(X_test_scaled)

    return X_train_sel, X_test_sel, y_train, y_test, selected_features

@st.cache_resource
def train_and_evaluate(X_train_sel, X_test_sel, y_train, y_test, selected_features):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }

    results = []
    predictions = {}
    for name, model in models.items():
        model.fit(X_train_sel, y_train)
        pred = model.predict(X_test_sel)
        predictions[name] = pred
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

    results_df = pd.DataFrame(results)

    rf_model = models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=selected_features)
    importances = importances.sort_values(ascending=False)

    return results_df, predictions, importances


# ---------------------------------------------------------------------------
# FUNGSI BANTU UNTUK UNDUH
# ---------------------------------------------------------------------------
def df_to_excel(df, sheet_name="Sheet1"):
    """Konversi DataFrame ke file Excel (byte) menggunakan openpyxl."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output

def fig_to_png(fig):
    """Konversi figure matplotlib ke PNG (byte)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# ANTARMUKA STREAMLIT UTAMA
# ---------------------------------------------------------------------------
def main():
    st.title("🏡 Pipeline Prediksi Harga Rumah")
    st.markdown("Aplikasi web ini mengeksekusi pipeline Feature Engineering dan Machine Learning untuk memprediksi harga rumah (Studi Kasus: Ames Housing).")

    # Bagian Unggah File
    st.sidebar.header("1. Unggah Dataset")
    uploaded_file = st.sidebar.file_uploader("Pilih file train.csv", type=["csv"])

    if uploaded_file is not None:
        with st.spinner("Memuat dan memproses data..."):
            # 1. Memuat Data
            df = load_data(uploaded_file)
            st.subheader("📊 Pratinjau Data Mentah")
            st.write(f"Ukuran Data: {df.shape[0]} baris, {df.shape[1]} kolom")
            st.dataframe(df.head())

            # 2. Pra-pemrosesan
            df_clean = preprocess(df)

            # 3. Rekayasa Fitur
            X, y = engineer_features(df_clean)

            # 4. Persiapan Data (K=20 fitur)
            k_features = st.sidebar.slider("Pilih jumlah fitur (KBest)", min_value=5, max_value=50, value=20)
            X_train_sel, X_test_sel, y_train, y_test, selected_features = prepare_model_data(X, y, k=k_features)

        with st.spinner("Melatih model..."):
            # 5. Pelatihan Model
            results_df, predictions, importances = train_and_evaluate(
                X_train_sel, X_test_sel, y_train, y_test, selected_features
            )

        st.success("Pipeline Selesai Dieksekusi!")

        # 6. Menampilkan Hasil Evaluasi
        st.subheader("📈 Hasil Evaluasi Model")
        st.dataframe(results_df, use_container_width=True)

        # --- PENJELASAN KINERJA MODEL TERBAIK ---
        # Mengambil baris model dengan RMSE terendah dan R2 tertinggi
        best_row = results_df.loc[results_df['RMSE'].idxmin()]
        worst_row = results_df.loc[results_df['RMSE'].idxmax()]

        best_name = best_row['Model']
        worst_name = worst_row['Model']

        best_rmse = best_row['RMSE']
        best_mae = best_row['MAE']
        best_r2 = best_row['R2']

        worst_rmse = worst_row['RMSE']
        worst_r2 = worst_row['R2']

        # Teks penjelasan otomatis
        st.info(f"📝 **Analisis Kinerja Model Terbaik**\n\n"
                f"Berdasarkan hasil evaluasi, model **{best_name}** menunjukkan kinerja terbaik dibandingkan dengan **{worst_name}**. "
                f"Berikut adalah rincian analisisnya:\n\n"
                f"- **RMSE (Root Mean Squared Error)**: {best_name} memiliki nilai RMSE sebesar **{best_rmse:,.2f}**, lebih rendah dibandingkan {worst_name} ({worst_rmse:,.2f}). "
                f"Nilai yang lebih rendah menunjukkan bahwa {best_name} memiliki tingkat kesalahan prediksi yang lebih kecil.\n\n"
                f"- **R² (R-Squared)**: {best_name} mencapai skor R² sebesar **{best_r2:.4f}** ({best_r2*100:.2f}%), sedangkan {worst_name} hanya mendapatkan {worst_r2:.4f} ({worst_r2*100:.2f}%). "
                f"Skor R² yang lebih tinggi (mendekati 1) berarti {best_name} jauh lebih baik dalam menjelaskan variasi data harga rumah berdasarkan fitur-fitur yang digunakan.\n\n"
                f"👉 **Kesimpulan:** Model **{best_name}** adalah pilihan paling optimal untuk dataset ini karena memberikan tingkat error yang paling minimal dan akurasi prediksi yang paling tinggi.")
        
        # Tombol unduh hasil evaluasi (Excel)
        st.download_button(
            label="⬇️ Unduh Hasil Evaluasi (Excel)",
            data=df_to_excel(results_df, sheet_name="Hasil Evaluasi"),
            file_name="hasil_evaluasi_model.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 7. Visualisasi (Dibagi dalam Tab)
        st.subheader("📉 Visualisasi Analisis")
        tab1, tab2, tab3 = st.tabs(["Aktual vs Prediksi", "Perbandingan Metrik", "Feature Importance"])

        best_pred = predictions[best_name]

        # ---------------- TAB 1: Aktual vs Prediksi ----------------
        with tab1:
            fig1, ax1 = plt.subplots(figsize=(8, 6))
            ax1.scatter(y_test, best_pred, alpha=0.5, color="#2563eb", edgecolor="none")
            lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
            ax1.plot(lims, lims, "r--", linewidth=1.5, label="Prediksi Sempurna")
            ax1.set_xlabel("Harga Aktual (SalePrice)")
            ax1.set_ylabel("Harga Prediksi")
            ax1.set_title(f"Aktual vs Prediksi - Model Terbaik ({best_name})")
            ax1.legend()
            st.pyplot(fig1)

            st.download_button(
                label="⬇️ Unduh Gambar Aktual vs Prediksi (PNG)",
                data=fig_to_png(fig1),
                file_name="aktual_vs_prediksi.png",
                mime="image/png"
            )

        # ---------------- TAB 2: Perbandingan Metrik ----------------
        with tab2:
            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
            axes2[0].bar(results_df["Model"], results_df["RMSE"], color=["#93c5fd", "#2563eb"])
            axes2[0].set_title("Perbandingan RMSE (Lebih rendah lebih baik)")
            axes2[0].set_ylabel("RMSE")

            axes2[1].bar(results_df["Model"], results_df["R2"], color=["#93c5fd", "#2563eb"])
            axes2[1].set_title("Perbandingan R² Score (Mendekati 1 lebih baik)")
            axes2[1].set_ylabel("R²")
            fig2.tight_layout()
            st.pyplot(fig2)

            st.download_button(
                label="⬇️ Unduh Gambar Perbandingan Metrik (PNG)",
                data=fig_to_png(fig2),
                file_name="perbandingan_metrik.png",
                mime="image/png"
            )

        # ---------------- TAB 3: Feature Importance ----------------
        with tab3:
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            top10 = importances.head(10).sort_values()
            ax3.barh(top10.index, top10.values, color="#2563eb")
            ax3.set_title("Top 10 Fitur Terpenting (Random Forest)")
            ax3.set_xlabel("Nilai Kepentingan (Feature Importance)")
            fig3.tight_layout()
            st.pyplot(fig3)

            # Tombol unduh gambar feature importance
            st.download_button(
                label="⬇️ Unduh Gambar Feature Importance (PNG)",
                data=fig_to_png(fig3),
                file_name="feature_importance.png",
                mime="image/png"
            )

            with st.expander("Lihat Semua Skor Feature Importance"):
                importance_df = importances.reset_index().rename(
                    columns={"index": "Fitur", 0: "Skor Kepentingan"}
                )
                st.dataframe(importance_df, use_container_width=True)

                # Tombol unduh tabel feature importance (Excel)
                st.download_button(
                    label="⬇️ Unduh Tabel Feature Importance (Excel)",
                    data=df_to_excel(importance_df, sheet_name="Feature Importance"),
                    file_name="feature_importance.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    else:
        st.info("Silakan unggah dataset train.csv pada bilah di sebelah kiri untuk memulai eksekusi.")

if __name__ == "__main__":
    main()


# KODE PROGRAM UNTUK OPSI A

# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import os

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_selection import SelectKBest, f_regression
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# RANDOM_STATE = 42
# pd.set_option("display.width", 120)


# # ---------------------------------------------------------------------------
# # 1. MEMUAT DATA
# # ---------------------------------------------------------------------------
# def load_data(path="train.csv"):
#     df = pd.read_csv(path)
#     print(f"[1/6] Data dimuat dari '{path}' -> ukuran: {df.shape}")
#     return df


# # ---------------------------------------------------------------------------
# # 2. PRA-PEMROSESAN DATA (penanganan missing value)
# # ---------------------------------------------------------------------------
# def preprocess(df):
#     # --- Penanganan khusus (bermakna) untuk kolom yang paling penting ---
#     # LotFrontage (numerik): diisi median PER Neighborhood, karena lebar lahan
#     # sangat dipengaruhi lokasi/lingkungan perumahan.
#     if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
#         df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
#             lambda s: s.fillna(s.median())
#         )
#     # GarageYrBlt: NaN berarti rumah tidak punya garasi -> diisi 0, ditandai
#     # lewat fitur biner HasGarage yang dibuat di tahap rekayasa fitur.
#     if "GarageCars" in df.columns:
#         df["HasGarage"] = (df["GarageCars"].fillna(0) > 0).astype(int)
#     if "GarageYrBlt" in df.columns:
#         df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

#     # --- Penanganan generik untuk SEMUA kolom lain ---
#     # Dataset asli Kaggle punya puluhan kolom kategorikal yang NaN-nya justru
#     # bermakna "fasilitas ini tidak ada" (mis. Alley, PoolQC, Fence, FireplaceQu,
#     # GarageType/Finish/Qual/Cond, BsmtCond/Exposure/FinType1/2, MiscFeature),
#     # jadi diisi kategori "NA" alih-alih dihapus barisnya.
#     obj_cols = df.select_dtypes(include=["object", "string"]).columns
#     if len(obj_cols) > 0:
#         df[obj_cols] = df[obj_cols].fillna("NA")

#     # Kolom numerik yang tersisa (mis. MasVnrArea) diisi median kolom tsb.
#     num_cols = df.select_dtypes(include=[np.number]).columns
#     if len(num_cols) > 0:
#         df[num_cols] = df[num_cols].fillna(df[num_cols].median(numeric_only=True))

#     sisa_missing = df.isna().sum()
#     sisa_missing = sisa_missing[sisa_missing > 0]
#     print("[2/6] Pra-pemrosesan selesai. Missing value tersisa:",
#           "tidak ada" if sisa_missing.empty else f"\n{sisa_missing}")
#     return df


# # ---------------------------------------------------------------------------
# # 3. REKAYASA FITUR (FEATURE ENGINEERING)
# # ---------------------------------------------------------------------------
# def engineer_features(df):
#     # 3.1 Feature creation - fitur baru yang lebih informatif dari fitur mentah.
#     df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
#     df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
#     df["WasRemodeled"] = (df["YearBuilt"] != df["YearRemodAdd"]).astype(int)
#     df["TotalSF"] = df["TotalBsmtSF"] + df["GrLivArea"]
#     df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"]
#     df["TotalPorchSF"] = df["WoodDeckSF"] + df["OpenPorchSF"]
#     df["HasPool"] = (df["PoolArea"] > 0).astype(int)
#     df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
#     df["QualCondScore"] = df["OverallQual"] * df["OverallCond"]

#     # 3.2 Binning - usia rumah (kontinu) -> kategori interval.
#     df["HouseAgeBin"] = pd.cut(
#         df["HouseAge"],
#         bins=[-1, 5, 15, 30, 60, 200],
#         labels=["Baru(0-5th)", "Muda(6-15th)", "Sedang(16-30th)", "Tua(31-60th)", "SangatTua(60+th)"],
#     )

#     # 3.3 Ordinal encoding untuk SEMUA kolom bertipe rating kualitas
#     #     (Ex=Excellent, Gd=Good, TA=Average, Fa=Fair, Po=Poor, NA=tidak ada).
#     #     Dicek otomatis kolom mana saja dari daftar ini yang benar-benar ada
#     #     di dataset, supaya kode ini tetap jalan baik untuk data simulasi
#     #     (33 kolom) maupun data asli Kaggle (81 kolom).
#     qual_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}
#     quality_cols = ["ExterQual", "ExterCond", "KitchenQual", "BsmtQual", "BsmtCond",
#                      "HeatingQC", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
#     for col in quality_cols:
#         if col in df.columns:
#             df[col + "_enc"] = df[col].map(qual_map).fillna(0)
#             df.drop(columns=[col], inplace=True)  # kolom teks asli sudah diwakili _enc

#     # 3.4 One-hot encoding - deteksi OTOMATIS semua kolom kategorikal yang
#     #     tersisa (object/category), bukan daftar tetap. Ini penting karena
#     #     dataset asli Kaggle punya puluhan kolom nominal (MSZoning, Street,
#     #     Exterior1st, Foundation, SaleType, dst.) yang jumlah & namanya
#     #     berbeda dari data simulasi.
#     nominal_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
#     df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

#     # 3.5 Jaring pengaman: kalau masih ada kolom non-numerik yang lolos
#     #     (mis. format data tak terduga), buang saja daripada bikin program
#     #     crash saat scaling. Beri tahu lewat pesan supaya tetap ketahuan.
#     sisa_non_numerik = df_encoded.select_dtypes(exclude=[np.number, bool]).columns.tolist()
#     sisa_non_numerik = [c for c in sisa_non_numerik if c not in ("Id", "SalePrice")]
#     if sisa_non_numerik:
#         print(f"[3/6] Peringatan: {len(sisa_non_numerik)} kolom non-numerik "
#               f"tidak terduga dan dihapus: {sisa_non_numerik}")
#         df_encoded = df_encoded.drop(columns=sisa_non_numerik)

#     # 3.6 Menentukan fitur final yang dipakai untuk pemodelan
#     drop_cols = ["Id", "SalePrice", "YearBuilt", "YearRemodAdd", "MoSold", "YrSold"]
#     feature_cols = [c for c in df_encoded.columns if c not in drop_cols]
#     X = df_encoded[feature_cols].copy()
#     y = df_encoded["SalePrice"].copy()

#     print(f"[3/6] Rekayasa fitur selesai. Jumlah fitur: {X.shape[1]} "
#           f"(dari {df.shape[1] - 1} fitur mentah)")
#     return X, y


# # ---------------------------------------------------------------------------
# # 4. SPLIT DATA + SCALING + FEATURE SELECTION
# # ---------------------------------------------------------------------------
# def prepare_model_data(X, y, k=20):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=RANDOM_STATE
#     )

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     selector = SelectKBest(score_func=f_regression, k=k)
#     selector.fit(X_train_scaled, y_train)
#     selected_features = X.columns[selector.get_support()].tolist()

#     X_train_sel = selector.transform(X_train_scaled)
#     X_test_sel = selector.transform(X_test_scaled)

#     print(f"[4/6] Split 80/20 + scaling + feature selection selesai. "
#           f"{k} fitur terpilih:")
#     print(selected_features)

#     return X_train_sel, X_test_sel, y_train, y_test, selected_features


# # ---------------------------------------------------------------------------
# # 5. PEMODELAN & EVALUASI
# # ---------------------------------------------------------------------------
# def train_and_evaluate(X_train_sel, X_test_sel, y_train, y_test, selected_features):
#     output_dir = "hasil_grafik"
#     os.makedirs(output_dir, exist_ok=True)

#     models = {
#         "Linear Regression": LinearRegression(),
#         "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
#     }

#     results = []
#     predictions = {}
#     for name, model in models.items():
#         model.fit(X_train_sel, y_train)
#         pred = model.predict(X_test_sel)
#         predictions[name] = pred
#         rmse = np.sqrt(mean_squared_error(y_test, pred))
#         mae = mean_absolute_error(y_test, pred)
#         r2 = r2_score(y_test, pred)
#         results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

#     results_df = pd.DataFrame(results)
#     results_df.to_csv(os.path.join(output_dir, "model_results.csv"), index=False)

#     print("\n[5/6] === HASIL EVALUASI MODEL === (disimpan ke hasil_grafik/model_results.csv)")
#     print(results_df.to_string(index=False))

#     rf_model = models["Random Forest"]
#     importances = pd.Series(rf_model.feature_importances_, index=selected_features)
#     importances = importances.sort_values(ascending=False)
#     importances.to_csv(os.path.join(output_dir, "feature_importances.csv"))

#     print("\nTop 10 fitur terpenting (Random Forest), disimpan ke hasil_grafik/feature_importances.csv:")
#     print(importances.head(10))

#     return results_df, predictions, importances


# # ---------------------------------------------------------------------------
# # 6. VISUALISASI
# # ---------------------------------------------------------------------------
# def make_plots(y_test, predictions, results_df, importances):
#     plt.style.use("seaborn-v0_8-whitegrid")

#     # Pastikan folder hasil_grafik ada
#     output_dir = "hasil_grafik"
#     os.makedirs(output_dir, exist_ok=True)

#     best_name = results_df.sort_values("RMSE").iloc[0]["Model"]
#     best_pred = predictions[best_name]

#     # 6.1 Actual vs Predicted
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.scatter(y_test, best_pred, alpha=0.5, color="#2563eb", edgecolor="none")
#     lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
#     ax.plot(lims, lims, "r--", linewidth=1.5, label="Prediksi Sempurna")
#     ax.set_xlabel("Harga Aktual (SalePrice)")
#     ax.set_ylabel("Harga Prediksi")
#     ax.set_title(f"Aktual vs Prediksi - {best_name}")
#     ax.legend()
#     fig.tight_layout()
#     fig.savefig(os.path.join(output_dir, "plot_actual_vs_predicted.png"), dpi=150)
#     plt.close(fig)

#     # 6.2 Perbandingan RMSE & R2 antar model
#     fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
#     axes[0].bar(results_df["Model"], results_df["RMSE"], color=["#93c5fd", "#2563eb"])
#     axes[0].set_title("Perbandingan RMSE")
#     axes[0].set_ylabel("RMSE")
#     axes[0].tick_params(axis="x", rotation=10)

#     axes[1].bar(results_df["Model"], results_df["R2"], color=["#93c5fd", "#2563eb"])
#     axes[1].set_title("Perbandingan R² Score")
#     axes[1].set_ylabel("R²")
#     axes[1].tick_params(axis="x", rotation=10)
#     fig.tight_layout()
#     fig.savefig(os.path.join(output_dir, "plot_model_comparison.png"), dpi=150)
#     plt.close(fig)

#     # 6.3 Feature importance (top 10)
#     fig, ax = plt.subplots(figsize=(7, 5))
#     top10 = importances.head(10).sort_values()
#     ax.barh(top10.index, top10.values, color="#2563eb")
#     ax.set_title("Top 10 Fitur Terpenting (Random Forest)")
#     ax.set_xlabel("Feature Importance")
#     fig.tight_layout()
#     fig.savefig(os.path.join(output_dir, "plot_feature_importance.png"), dpi=150)
#     plt.close(fig)

#     print(f"\n[6/6] Grafik selesai dibuat di folder '{output_dir}': "
#           "plot_actual_vs_predicted.png, plot_model_comparison.png, plot_feature_importance.png")


# # ---------------------------------------------------------------------------
# # MAIN - menjalankan seluruh pipeline berurutan
# # ---------------------------------------------------------------------------
# def main():
#     df = load_data("train.csv")
#     df = preprocess(df)
#     X, y = engineer_features(df)
#     X_train_sel, X_test_sel, y_train, y_test, selected_features = prepare_model_data(X, y)
#     results_df, predictions, importances = train_and_evaluate(
#         X_train_sel, X_test_sel, y_train, y_test, selected_features
#     )
#     make_plots(y_test, predictions, results_df, importances)
#     print("\nSELESAI. Semua file hasil tersimpan di folder ini.")


# if __name__ == "__main__":
#     main()

"""
==============================================================================
 00_generate_dataset.py
==============================================================================
Skrip ini membuat dataset SIMULASI (bukan data asli) yang meniru dataset
Kaggle "House Prices - Advanced Regression Techniques" (Ames Housing).

KENAPA ADA FILE INI?
Lingkungan pengerjaan tidak punya akses internet, sehingga file asli dari
Kaggle tidak bisa langsung diunduh di sini. Skrip ini membuat pengganti
sementara agar seluruh pipeline (01_feature_engineering_modeling.py) tetap
bisa dijalankan dan diuji dari awal sampai akhir.

HUBUNGAN DENGAN FILE LAIN:
    00_generate_dataset.py  --(membuat file)-->  train.csv  --(dibaca oleh)--> 01_feature_engineering_modeling.py

Jadi urutan menjalankannya WAJIB:
    1) python3 00_generate_dataset.py          -> menghasilkan train.csv
    2) python3 01_feature_engineering_modeling.py  -> membaca train.csv lalu memprosesnya

Jika kamu sudah punya train.csv ASLI dari Kaggle, cukup taruh file itu di
folder yang sama (dengan nama persis "train.csv") dan LEWATI langkah 1;
langsung jalankan langkah 2.

Referensi dataset asli (WAJIB dicantumkan di laporan):
- Kaggle Competition: "House Prices - Advanced Regression Techniques"
  https://www.kaggle.com/c/house-prices-advanced-regression-techniques
- De Cock, D. (2011). "Ames, Iowa: Alternative to the Boston Housing Data
  as an End of Semester Regression Project." Journal of Statistics
  Education, 19(3).
==============================================================================
"""

import numpy as np
import pandas as pd


def main():
    rng = np.random.default_rng(42)   # seed tetap -> hasil selalu sama tiap dijalankan
    N = 1200                          # jumlah baris (rumah) yang disimulasikan

    neighborhoods = ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst",
                      "Gilbert", "NridgHt", "Sawyer", "BrkSide", "Mitchel"]
    neigh_base_price = {
        "NAmes": 145000, "CollgCr": 197000, "OldTown": 123000, "Edwards": 128000,
        "Somerst": 225000, "Gilbert": 190000, "NridgHt": 315000, "Sawyer": 136000,
        "BrkSide": 124000, "Mitchel": 158000,
    }

    house_style = rng.choice(["1Story", "2Story", "1.5Fin", "SLvl"], size=N, p=[0.5, 0.3, 0.12, 0.08])
    neighborhood = rng.choice(neighborhoods, size=N)
    overall_qual = rng.integers(2, 11, size=N)          # 1-10 rating kualitas keseluruhan
    overall_cond = rng.integers(2, 10, size=N)          # 1-10 rating kondisi keseluruhan
    year_built = rng.integers(1900, 2011, size=N)
    year_remod = np.maximum(year_built, year_built + rng.integers(0, 40, size=N))
    year_remod = np.minimum(year_remod, 2010)

    lot_area = rng.normal(10500, 4200, size=N).clip(1500, 45000)
    gr_liv_area = rng.normal(1500, 480, size=N).clip(500, 4500) + overall_qual * 25
    total_bsmt_sf = (gr_liv_area * rng.uniform(0.4, 0.95, size=N)).clip(0, 3000)
    first_flr_sf = gr_liv_area * rng.uniform(0.55, 1.0, size=N)
    garage_cars = rng.integers(0, 4, size=N)
    garage_area = garage_cars * rng.normal(280, 40, size=N).clip(0, None)
    full_bath = rng.integers(1, 4, size=N)
    half_bath = rng.integers(0, 2, size=N)
    bedroom_abvgr = rng.integers(1, 6, size=N)
    kitchen_abvgr = rng.integers(1, 3, size=N)
    tot_rms_abvgrd = bedroom_abvgr + kitchen_abvgr + rng.integers(1, 4, size=N)
    fireplaces = rng.integers(0, 3, size=N)
    wood_deck_sf = rng.choice([0, 0, 0, 100, 150, 200, 250], size=N)
    open_porch_sf = rng.integers(0, 200, size=N)
    pool_area = rng.choice([0]*97 + [200, 400, 550], size=N)

    exter_qual = rng.choice(["Ex", "Gd", "TA", "Fa"], size=N, p=[0.08, 0.35, 0.5, 0.07])
    kitchen_qual = rng.choice(["Ex", "Gd", "TA", "Fa"], size=N, p=[0.1, 0.4, 0.42, 0.08])
    bsmt_qual = rng.choice(["Ex", "Gd", "TA", "Fa", "NA"], size=N, p=[0.08, 0.32, 0.4, 0.1, 0.1])
    central_air = rng.choice(["Y", "N"], size=N, p=[0.93, 0.07])
    sale_condition = rng.choice(["Normal", "Abnorml", "Partial", "Family"], size=N, p=[0.82, 0.07, 0.08, 0.03])
    mo_sold = rng.integers(1, 13, size=N)
    yr_sold = rng.integers(2006, 2011, size=N)

    # missing values buatan (menyerupai data asli yang punya banyak NA)
    lot_frontage = rng.normal(70, 22, size=N).clip(20, 200)
    miss_idx = rng.choice(N, size=int(N * 0.17), replace=False)
    lot_frontage[miss_idx] = np.nan

    garage_yr_blt = year_built + rng.integers(0, 5, size=N)
    no_garage_idx = np.where(garage_cars == 0)[0]
    garage_yr_blt = garage_yr_blt.astype(float)
    garage_yr_blt[no_garage_idx] = np.nan

    qual_map = {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "NA": 0}
    base_neigh = np.array([neigh_base_price[n] for n in neighborhood])

    sale_price = (
        base_neigh
        + overall_qual * 9200
        + overall_cond * 1800
        + (gr_liv_area) * 46
        + (total_bsmt_sf) * 22
        + garage_cars * 9800
        + full_bath * 6200
        + fireplaces * 3200
        + np.array([qual_map[q] for q in exter_qual]) * 4100
        + np.array([qual_map[q] for q in kitchen_qual]) * 5200
        + (year_built - 1900) * 210
        + rng.normal(0, 16000, size=N)
    ).clip(34900, None)

    df = pd.DataFrame({
        "Id": np.arange(1, N + 1),
        "MSSubClass": rng.choice([20, 30, 40, 50, 60, 70, 80, 90], size=N),
        "Neighborhood": neighborhood,
        "HouseStyle": house_style,
        "LotFrontage": lot_frontage.round(1),
        "LotArea": lot_area.round(0).astype(int),
        "OverallQual": overall_qual,
        "OverallCond": overall_cond,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "ExterQual": exter_qual,
        "BsmtQual": bsmt_qual,
        "TotalBsmtSF": total_bsmt_sf.round(0).astype(int),
        "1stFlrSF": first_flr_sf.round(0).astype(int),
        "GrLivArea": gr_liv_area.round(0).astype(int),
        "FullBath": full_bath,
        "HalfBath": half_bath,
        "BedroomAbvGr": bedroom_abvgr,
        "KitchenAbvGr": kitchen_abvgr,
        "KitchenQual": kitchen_qual,
        "TotRmsAbvGrd": tot_rms_abvgrd,
        "Fireplaces": fireplaces,
        "GarageYrBlt": garage_yr_blt.round(0),
        "GarageCars": garage_cars,
        "GarageArea": garage_area.round(0).astype(int),
        "WoodDeckSF": wood_deck_sf,
        "OpenPorchSF": open_porch_sf,
        "PoolArea": pool_area,
        "CentralAir": central_air,
        "MoSold": mo_sold,
        "YrSold": yr_sold,
        "SaleCondition": sale_condition,
        "SalePrice": sale_price.round(0).astype(int),
    })

    # Disimpan di folder yang SAMA dengan tempat skrip ini dijalankan (folder aktif).
    df.to_csv("train.csv", index=False)

    print("=" * 60)
    print(f"BERHASIL: train.csv dibuat -> {df.shape[0]} baris, {df.shape[1]} kolom")
    print("=" * 60)
    print(df.head())
    print("\nJumlah missing value per kolom (yang > 0):")
    print(df.isna().sum()[df.isna().sum() > 0])
    print("\nLangkah selanjutnya: jalankan")
    print("    python3 01_feature_engineering_modeling.py")


if __name__ == "__main__":
    main()

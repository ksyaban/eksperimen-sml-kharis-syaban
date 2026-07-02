#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# ======================================================
# LOAD DATASET
# ======================================================

def load_dataset(file_path):
    """Memuat dataset dari file CSV."""
    print(f"\n[INFO] Membaca dataset: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {file_path}")

    df = pd.read_csv(file_path)

    print(f"[INFO] Dataset berhasil dimuat ({df.shape[0]} baris, {df.shape[1]} kolom)")

    return df


# ======================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ======================================================

def data_information(df):

    print("\n========== INFORMASI DATA ==========")

    print("\nShape Dataset")
    print(df.shape)

    print("\nNama Kolom")
    print(df.columns.tolist())

    print("\nTipe Data")
    print(df.dtypes)

    print("\nMissing Value")
    print(df.isnull().sum())

    print("\nJumlah Data Duplikat")
    print(df.duplicated().sum())


# ======================================================
# HANDLE MISSING VALUE
# ======================================================

def handle_missing_value(df):

    print("\n[INFO] Menangani Missing Value...")

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# ======================================================
# REMOVE DUPLICATE
# ======================================================

def remove_duplicate(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"[INFO] Menghapus {before-after} data duplikat.")

    return df


# ======================================================
# ENCODE CATEGORICAL FEATURES
# ======================================================

def encode_categorical(df):

    print("[INFO] Encoding fitur kategorikal...")

    encoder = LabelEncoder()

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df


# ======================================================
# FEATURE SCALING
# ======================================================

def feature_scaling(df):

    print("[INFO] Melakukan Standard Scaling...")

    target_candidates = [
        "Outcome",
        "target",
        "Target",
        "Class",
        "label",
        "Label"
    ]

    target_column = None

    for col in target_candidates:
        if col in df.columns:
            target_column = col
            break

    scaler = StandardScaler()

    if target_column is not None:

        feature_cols = df.drop(columns=[target_column]).columns

        df[feature_cols] = scaler.fit_transform(df[feature_cols])

    else:

        df[df.columns] = scaler.fit_transform(df)

    return df


# ======================================================
# SAVE DATASET
# ======================================================

def save_dataset(df, output_path):

    folder = os.path.dirname(output_path)

    if folder != "":
        os.makedirs(folder, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\n[INFO] Dataset hasil preprocessing disimpan pada:")
    print(output_path)


# ======================================================
# MAIN
# ======================================================

def main():

    parser = argparse.ArgumentParser(
        description="Automasi Data Preprocessing"
    )

    parser.add_argument(
        "input",
        help="Lokasi dataset input (.csv)"
    )

    parser.add_argument(
        "output",
        help="Lokasi dataset hasil preprocessing (.csv)"
    )

    args = parser.parse_args()

    df = load_dataset(args.input)

    data_information(df)

    df = handle_missing_value(df)

    df = remove_duplicate(df)

    df = encode_categorical(df)

    df = feature_scaling(df)

    save_dataset(df, args.output)

    print("\n===================================")
    print("PREPROCESSING BERHASIL")
    print("===================================")


if __name__ == "__main__":
    main()
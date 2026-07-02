import argparse
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# =====================================================
# LOAD DATASET
# =====================================================

def load_dataset(path):
    print(f"Memuat dataset: {path}")
    return pd.read_csv(path)


# =====================================================
# EDA
# =====================================================

def data_information(df):

    print("\n===== INFORMASI DATA =====")

    print("Shape :", df.shape)

    print("\nTipe Data")
    print(df.dtypes)

    print("\nMissing Value")
    print(df.isnull().sum())

    print("\nDuplicate")
    print(df.duplicated().sum())


# =====================================================
# HANDLE MISSING VALUE
# =====================================================

def handle_missing(df):

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())

    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


# =====================================================
# REMOVE DUPLICATE
# =====================================================

def remove_duplicate(df):
    return df.drop_duplicates()


# =====================================================
# ENCODING
# =====================================================

def encode_data(df):

    encoder = LabelEncoder()

    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    return df


# =====================================================
# SCALING
# =====================================================

def scaling(df):

    scaler = StandardScaler()

    df = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns
    )

    return df


# =====================================================
# SPLIT
# =====================================================

def split_dataset(df):

    train, test = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    return train, test


# =====================================================
# SAVE
# =====================================================

def save_dataset(train, test, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print("\nDataset preprocessing berhasil disimpan pada:")
    print(output_dir)


# =====================================================
# MAIN
# =====================================================

def main():

    parser = argparse.ArgumentParser(
        description="Automasi Data Preprocessing"
    )

    parser.add_argument(
        "dataset",
        help="Lokasi dataset CSV"
    )

    parser.add_argument(
        "-o",
        "--output",
        default="dataset_preprocessing",
        help="Folder output preprocessing"
    )

    args = parser.parse_args()

    # Load dataset
    df = load_dataset(args.dataset)

    # EDA
    data_information(df)

    # Preprocessing
    df = handle_missing(df)
    df = remove_duplicate(df)
    df = encode_data(df)
    df = scaling(df)

    # Split dataset
    train, test = split_dataset(df)

    # Save
    save_dataset(train, test, args.output)

    print("\nPreprocessing selesai.")


if __name__ == "__main__":
    main()
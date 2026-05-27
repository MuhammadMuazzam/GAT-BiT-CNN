"""
=============================================================================
GAC-BiT-CNN: Fuse Multiple Protein Embedding CSVs into One Feature Set
=============================================================================
Paper: Enhancing Functional Prediction of Methyltransferases Using a Hybrid
       GAC-BiT-CNN Deep Learning Model with Protein Language Model Embeddings

Authors: Muhammad Muazzam, Farman Ali, Muhammad Asif
         Department of Computer Science, Bahria University Islamabad, Pakistan

Description:
    Concatenates ProtGen, ESM-2, ProtT5-BFD, and ProtT5-XL-UniRef50
    embedding CSV files into one fused feature set (4352-dim).
    All CSV files must have the same number of rows in the same order.

    Final fused dimension: 1024 + 1280 + 1024 + 1024 = 4352 features

Usage:
    python code/fuse_features.py \
        --protgen  embeddings/data_training_ProtGen.csv \
        --esm2     embeddings/data_training_ESM2.csv \
        --bfd      embeddings/data_training_ProtT5_BFD.csv \
        --xl       embeddings/data_training_ProtT5_XL.csv \
        --output   embeddings/data_training_Fused.csv
=============================================================================
"""

import argparse
import pandas as pd


def fuse_features(csv_files, output_csv):
    """
    Concatenate multiple embedding CSV files into one fused feature set.

    The first CSV keeps its Sequence_ID column.
    All subsequent CSVs have their Sequence_ID column dropped to avoid
    duplicate ID columns in the final output.

    Args:
        csv_files (list): Ordered list of CSV file paths to fuse.
                          Expected order: [ProtGen, ESM2, ProtT5-BFD, ProtT5-XL]
        output_csv (str): Path to save the fused output CSV.
    """
    print("\n" + "="*60)
    print("  Fusing Protein Language Model Embedding CSVs")
    print("="*60)

    dfs = []
    total_features = 0

    for i, csv_path in enumerate(csv_files):
        df = pd.read_csv(csv_path)
        n_features = df.shape[1] - (1 if "Sequence_ID" in df.columns else 0)
        total_features += n_features
        print(f"  [{i+1}] {csv_path}")
        print(f"       Shape: {df.shape}   Features: {n_features}")

        if "Sequence_ID" in df.columns:
            if i == 0:
                dfs.append(df)                                  # keep ID column once
            else:
                dfs.append(df.drop(columns=["Sequence_ID"]))   # drop duplicate ID
        else:
            dfs.append(df)

    fused_df = pd.concat(dfs, axis=1)
    fused_df.to_csv(output_csv, index=False)

    print(f"\n  Fused CSV saved: {output_csv}")
    print(f"  Final shape    : {fused_df.shape}")
    print(f"  Total features : {total_features}")
    print(f"  Expected dim   : 1024 (ProtGen) + 1280 (ESM2) + 1024 (BFD) + 1024 (XL) = 4352")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fuse ProtGen, ESM-2, ProtT5-BFD, ProtT5-XL embeddings."
    )
    parser.add_argument("--protgen", type=str, required=True,
                        help="Path to ProtGen embedding CSV")
    parser.add_argument("--esm2",    type=str, required=True,
                        help="Path to ESM-2 embedding CSV")
    parser.add_argument("--bfd",     type=str, required=True,
                        help="Path to ProtT5-BFD embedding CSV")
    parser.add_argument("--xl",      type=str, required=True,
                        help="Path to ProtT5-XL-UniRef50 embedding CSV")
    parser.add_argument("--output",  type=str, required=True,
                        help="Output path for the fused CSV")

    args = parser.parse_args()

    fuse_features(
        csv_files=[args.protgen, args.esm2, args.bfd, args.xl],
        output_csv=args.output
    )

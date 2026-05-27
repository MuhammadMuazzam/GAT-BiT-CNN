"""
=============================================================================
GAC-BiT-CNN: Feature Extraction using Protein Language Models
=============================================================================
Paper: Enhancing Functional Prediction of Methyltransferases Using a Hybrid
       GAC-BiT-CNN Deep Learning Model with Protein Language Model Embeddings

Authors: Muhammad Muazzam, Farman Ali, Muhammad Asif
         Department of Computer Science, Bahria University Islamabad, Pakistan

Description:
    This script extracts protein embeddings from four state-of-the-art
    protein language models:
        1. ProtGen  (progen2-medium)        -> 1024-dim
        2. ProtT5-BFD (prot_t5_xl_bfd)     -> 1024-dim
        3. ProtT5-XL  (prot_t5_xl_uniref50)-> 1024-dim
        4. ESM-2    (esm2_t33_650M_UR50D)  -> 1280-dim

Usage:
    python feature_extraction.py --input data/data_training.txt \
                                  --output_dir embeddings/ \
                                  --model all
=============================================================================
"""

import os
import gc
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# UTILITY: FASTA Parser
# ─────────────────────────────────────────────────────────────

def load_fasta(file_path):
    """
    Parse a FASTA file and return headers and sequences.

    Args:
        file_path (str): Path to the FASTA file.

    Returns:
        headers (list): List of sequence IDs.
        sequences (list): List of amino acid sequences.
    """
    sequences, headers = [], []
    with open(file_path, 'r') as f:
        seq, header = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    sequences.append(seq.replace(" ", "").upper())
                    headers.append(header)
                    seq = ""
                header = line[1:].split()[0]  # Keep only the ID
            else:
                seq += line
        if seq:
            sequences.append(seq.replace(" ", "").upper())
            headers.append(header)
    return headers, sequences


# ─────────────────────────────────────────────────────────────
# MODULE 1: ProtGen (progen2-medium) Embeddings
# ─────────────────────────────────────────────────────────────

def extract_protgen(input_fasta, output_csv):
    """
    Extract 1024-dim embeddings using ProtGen (progen2-medium).

    Args:
        input_fasta (str): Path to input FASTA file.
        output_csv (str): Path to save the embedding CSV.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("\n" + "="*60)
    print("  MODULE 1: ProtGen Embedding Extraction")
    print("="*60)

    model_name = "hugohrban/progen2-medium"
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device).eval()

    headers, sequences = load_fasta(input_fasta)
    print(f"Sequences loaded: {len(sequences)}")

    def get_embedding(sequence):
        tokens = tokenizer(
            sequence,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=1024
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1].squeeze(0)

        # Masked mean pooling
        pooled = (
            hidden_states * attention_mask.squeeze(0).unsqueeze(-1)
        ).sum(0) / attention_mask.sum()
        return pooled.cpu().numpy()

    all_ids, embeddings = [], []
    for header, seq in tqdm(zip(headers, sequences), total=len(sequences),
                             desc="Extracting ProtGen embeddings"):
        try:
            emb = get_embedding(seq)
            all_ids.append(header)
            embeddings.append(emb)
        except Exception as e:
            print(f"  [SKIP] {header}: {e}")

    df = pd.DataFrame(embeddings, columns=[f"ProtGen_F{i+1}" for i in range(len(embeddings[0]))])
    df.insert(0, "Sequence_ID", all_ids)
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}  Shape: {df.shape}")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
# MODULE 2: ProtT5-BFD Embeddings
# ─────────────────────────────────────────────────────────────

def extract_prott5_bfd(input_fasta, output_csv, batch_size=8):
    """
    Extract 1024-dim embeddings using ProtT5-XL-BFD.

    Args:
        input_fasta (str): Path to input FASTA file.
        output_csv (str): Path to save the embedding CSV.
        batch_size (int): Number of sequences per batch.
    """
    from transformers import T5Tokenizer, T5EncoderModel

    print("\n" + "="*60)
    print("  MODULE 2: ProtT5-BFD Embedding Extraction")
    print("="*60)

    model_name = "Rostlab/prot_t5_xl_bfd"
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device).eval()

    headers, sequences = load_fasta(input_fasta)
    print(f"Sequences loaded: {len(sequences)}")

    def embed_batch(seqs):
        # Space-separate amino acids (required by ProtT5)
        spaced = [" ".join(list(s)) for s in seqs]
        tokens = tokenizer(
            spaced,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            embeddings = model(**tokens).last_hidden_state

        mask = tokens.attention_mask
        mask_exp = mask.unsqueeze(-1).expand(embeddings.size()).float()
        summed = torch.sum(embeddings * mask_exp, 1)
        counts = torch.clamp(mask_exp.sum(1), min=1e-9)
        return (summed / counts).cpu().numpy()

    all_ids, all_embeddings = [], []
    for i in tqdm(range(0, len(sequences), batch_size),
                  desc="Extracting ProtT5-BFD embeddings", unit="batch"):
        batch_seqs = sequences[i:i + batch_size]
        batch_hdrs = headers[i:i + batch_size]
        try:
            embs = embed_batch(batch_seqs)
            all_embeddings.extend(embs)
            all_ids.extend(batch_hdrs)
        except Exception as e:
            print(f"  [SKIP] Batch {i // batch_size}: {e}")

        if i % (batch_size * 10) == 0:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            gc.collect()

    df = pd.DataFrame(all_embeddings,
                      columns=[f"ProtT5_BFD_F{i+1}" for i in range(len(all_embeddings[0]))])
    df.insert(0, "Sequence_ID", all_ids)
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}  Shape: {df.shape}")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
# MODULE 3: ProtT5-XL-UniRef50 Embeddings
# ─────────────────────────────────────────────────────────────

def extract_prott5_xl(input_fasta, output_csv):
    """
    Extract 1024-dim embeddings using ProtT5-XL-UniRef50.

    Args:
        input_fasta (str): Path to input FASTA file.
        output_csv (str): Path to save the embedding CSV.
    """
    from transformers import T5Tokenizer, T5EncoderModel

    print("\n" + "="*60)
    print("  MODULE 3: ProtT5-XL-UniRef50 Embedding Extraction")
    print("="*60)

    model_name = "Rostlab/prot_t5_xl_uniref50"
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device).eval()

    headers, sequences = load_fasta(input_fasta)
    print(f"Sequences loaded: {len(sequences)}")

    def embed_sequence(seq):
        spaced = ' '.join(list(seq))
        tokens = tokenizer(
            spaced,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024
        ).input_ids.to(device)

        with torch.no_grad():
            embedding = model(input_ids=tokens).last_hidden_state

        # Remove special tokens and mean-pool
        embedding = embedding[0][1:-1]
        return torch.mean(embedding, dim=0).cpu().numpy()

    all_ids, embeddings = [], []
    for header, seq in tqdm(zip(headers, sequences), total=len(sequences),
                             desc="Extracting ProtT5-XL embeddings"):
        try:
            emb = embed_sequence(seq)
            embeddings.append(emb)
            all_ids.append(header)
        except Exception as e:
            print(f"  [SKIP] {header}: {e}")

    df = pd.DataFrame(embeddings,
                      columns=[f"ProtT5_XL_F{i+1}" for i in range(len(embeddings[0]))])
    df.insert(0, "Sequence_ID", all_ids)
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}  Shape: {df.shape}")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
# MODULE 4: ESM-2 Embeddings
# ─────────────────────────────────────────────────────────────

def extract_esm2(input_fasta, output_csv, batch_size=4):
    """
    Extract 1280-dim embeddings using ESM-2 (esm2_t33_650M_UR50D).

    Args:
        input_fasta (str): Path to input FASTA file.
        output_csv (str): Path to save the embedding CSV.
        batch_size (int): Number of sequences per batch.
    """
    from transformers import AutoTokenizer, AutoModel
    from Bio import SeqIO

    print("\n" + "="*60)
    print("  MODULE 4: ESM-2 Embedding Extraction")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"Loading tokenizer and model: {model_name}")
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)

    sequences = {rec.id: str(rec.seq) for rec in SeqIO.parse(input_fasta, "fasta")}
    print(f"Sequences loaded: {len(sequences)}")

    processed = [(seq, seq_id) for seq_id, seq in sequences.items() if seq]
    all_ids, all_features = [], []

    for i in tqdm(range(0, len(processed), batch_size), desc="Extracting ESM-2 embeddings"):
        batch = processed[i:i + batch_size]
        seqs, ids = zip(*batch)
        try:
            inputs = tokenizer(
                list(seqs),
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=1024
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            emb = outputs.last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1)
            pooled = (torch.sum(emb * mask, 1) / torch.sum(mask, 1)).cpu().numpy()
            all_features.extend(pooled)
            all_ids.extend(ids)
        except Exception as e:
            print(f"  [SKIP] Batch {i // batch_size}: {e}")

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    feats = np.array(all_features)
    df = pd.DataFrame(feats, columns=[f"ESM2_F{i+1}" for i in range(feats.shape[1])])
    df.insert(0, "Sequence_ID", all_ids)
    df.to_csv(output_csv, index=False)
    print(f"  Saved: {output_csv}  Shape: {df.shape}")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────
# MODULE 5: Fuse All Feature Sets
# ─────────────────────────────────────────────────────────────

def fuse_features(csv_files, output_csv):
    """
    Concatenate multiple embedding CSV files into one fused feature set.
    All CSV files must have the same number of rows in the same order.

    Args:
        csv_files (list): List of paths to embedding CSV files.
        output_csv (str): Path to save the fused CSV.
    """
    print("\n" + "="*60)
    print("  MODULE 5: Fusing Feature Sets")
    print("="*60)

    dfs = []
    for i, csv in enumerate(csv_files):
        df = pd.read_csv(csv)
        print(f"  Loaded: {csv}  Shape: {df.shape}")
        if "Sequence_ID" in df.columns:
            if i == 0:
                dfs.append(df)
            else:
                dfs.append(df.drop(columns=["Sequence_ID"], errors="ignore"))
        else:
            dfs.append(df)

    fused = pd.concat(dfs, axis=1)
    fused.to_csv(output_csv, index=False)
    print(f"\n  Fused feature set saved: {output_csv}")
    print(f"  Final shape: {fused.shape}")
    print(f"  Total features: {fused.shape[1] - 1} (excluding Sequence_ID)")


# ─────────────────────────────────────────────────────────────
# MAIN: Command-line interface
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract protein language model embeddings for methyltransferase prediction."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input FASTA file (e.g., data/data_training.txt)")
    parser.add_argument("--output_dir", type=str, default="embeddings",
                        help="Directory to save embedding CSV files")
    parser.add_argument("--model", type=str, default="all",
                        choices=["protgen", "prott5_bfd", "prott5_xl", "esm2", "fuse", "all"],
                        help="Which model(s) to run")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for embedding extraction")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prefix = os.path.splitext(os.path.basename(args.input))[0]
    out = {
        "protgen":    os.path.join(args.output_dir, f"{prefix}_ProtGen.csv"),
        "prott5_bfd": os.path.join(args.output_dir, f"{prefix}_ProtT5_BFD.csv"),
        "prott5_xl":  os.path.join(args.output_dir, f"{prefix}_ProtT5_XL.csv"),
        "esm2":       os.path.join(args.output_dir, f"{prefix}_ESM2.csv"),
        "fused":      os.path.join(args.output_dir, f"{prefix}_Fused.csv"),
    }

    if args.model in ("protgen", "all"):
        extract_protgen(args.input, out["protgen"])

    if args.model in ("prott5_bfd", "all"):
        extract_prott5_bfd(args.input, out["prott5_bfd"], batch_size=args.batch_size)

    if args.model in ("prott5_xl", "all"):
        extract_prott5_xl(args.input, out["prott5_xl"])

    if args.model in ("esm2", "all"):
        extract_esm2(args.input, out["esm2"], batch_size=args.batch_size)

    if args.model in ("fuse", "all"):
        fuse_features(
            [out["protgen"], out["esm2"], out["prott5_bfd"], out["prott5_xl"]],
            out["fused"]
        )

    print("\n" + "="*60)
    print("  All embeddings extracted successfully!")
    print("="*60)

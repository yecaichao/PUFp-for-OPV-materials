#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path

import pandas as pd


def classify_ring_type(smiles):
    num_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "%"]
    nums = [ch for ch in smiles if ch in num_list]
    if len(nums) == 0:
        return "bratch"
    if len(nums) == 2:
        return "single"
    if len(nums) == 4:
        return "double"
    if len(nums) > 4:
        return "fused"
    return "unknow"


def contains_sulfur(smiles):
    if "s" in smiles:
        s_num = smiles.count("s")
        if "se" in smiles:
            return "YES" if s_num > smiles.count("se") else "NO"
        return "YES"
    if "S" in smiles:
        s_num = smiles.count("S")
        if "Si" in smiles:
            return "YES" if s_num > smiles.count("i") else "NO"
        return "YES"
    return "NO"


def yes_no(flag):
    return "YES" if flag else "NO"


def type_judge(row):
    element = []
    if row["sulfur"] == "YES":
        element.append("S")
    if row["nitrogen"] == "YES":
        element.append("N")
    if row["fluorine"] == "YES":
        element.append("F")
    if row["oxygen"] == "YES":
        element.append("O")
    if row["selenium"] == "YES":
        element.append("se")
    if row["silicon"] == "YES":
        element.append("Si")
    if row["chlorine"] == "YES":
        element.append("Cl")

    type_name = row["ring_type"]
    for item in element:
        type_name = type_name + "-" + item
    return type_name


def build_ring_dataframe(ring_total_list_path):
    ring_smiles = pd.read_csv(ring_total_list_path, index_col=0).iloc[:, 0].astype(str)
    ring_df = pd.DataFrame({"smiles": ring_smiles})
    ring_df.index.name = "index"

    ring_df["ring_type"] = ring_df["smiles"].apply(classify_ring_type)
    ring_df["sulfur"] = ring_df["smiles"].apply(contains_sulfur)
    ring_df["selenium"] = ring_df["smiles"].apply(lambda x: yes_no("se" in x))
    ring_df["silicon"] = ring_df["smiles"].apply(lambda x: yes_no("Si" in x))
    ring_df["oxygen"] = ring_df["smiles"].apply(lambda x: yes_no(("o" in x) or ("O" in x)))
    ring_df["nitrogen"] = ring_df["smiles"].apply(lambda x: yes_no(("n" in x) or ("N" in x)))
    ring_df["chlorine"] = ring_df["smiles"].apply(lambda x: yes_no("Cl" in x))
    ring_df["fluorine"] = ring_df["smiles"].apply(lambda x: yes_no("F" in x))
    ring_df["polymer_type"] = ring_df.apply(type_judge, axis="columns")
    return ring_df


def build_type_frame(index_data_path, ring_df):
    index_frame = pd.read_csv(index_data_path, index_col=0, dtype=str).fillna("none")

    def map_polymer_type(value):
        if value == "none":
            return None
        return ring_df.loc[int(value), "polymer_type"]

    return index_frame.applymap(map_polymer_type)


def main(ring_total_list_path="ring_total_list.csv", index_data_path="index_data.csv", output_dir="."):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not Path(ring_total_list_path).exists():
        raise FileNotFoundError(f"Missing required input file: {ring_total_list_path}")
    if not Path(index_data_path).exists():
        raise FileNotFoundError(f"Missing required input file: {index_data_path}")

    ring_df = build_ring_dataframe(ring_total_list_path)
    ring_df.to_csv(output_dir / "ring_df.csv")

    type_frame = build_type_frame(index_data_path, ring_df)
    type_frame.to_csv(output_dir / "type_frame.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify polymer units into PUFp categories.")
    parser.add_argument("--ring-total-list", default="ring_total_list.csv", help="Path to ring_total_list.csv")
    parser.add_argument("--index-data", default="index_data.csv", help="Path to index_data.csv")
    parser.add_argument("--output-dir", default=".", help="Directory used to store generated csv files.")
    args = parser.parse_args()
    main(args.ring_total_list, args.index_data, args.output_dir)

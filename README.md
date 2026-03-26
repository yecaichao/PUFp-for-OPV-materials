<img width="1099" height="402" alt="image" src="https://github.com/user-attachments/assets/70645608-88b9-4129-9f66-f270b480d4a9" />

# PUFp for OPV materials

This repository contains a lightweight workflow for polymer-unit recognition,
polymer-unit fingerprint (PUFp) generation, and OPV-oriented downstream
machine-learning experiments.

## Included files

- `get-polymer-unit.py`
- `polymer-unit-classify.py`
- `structure_identity_tool.py`
- `OPV_exp_data.csv`
- `con_smile.py`
- `RF.py`
- `KRR.py`
- `SVM.py`

## Input dataset

The packaged OPV dataset is:

- `OPV_exp_data.csv`

Important columns in this dataset include:

- `No.`
- `n(SMILES)`
- `p(SMILES)`
- `PCE_max(%)`

## Main workflow

The recognition workflow now supports explicit column selection and output
directories.

Example: process the donor column `n(SMILES)`:

```bash
python get-polymer-unit.py OPV_exp_data.csv --name-column "No." --smiles-column "n(SMILES)" --output-dir output_n
python polymer-unit-classify.py --ring-total-list output_n/ring_total_list.csv --index-data output_n/index_data.csv --output-dir output_n
```

Example: process the acceptor column `p(SMILES)`:

```bash
python get-polymer-unit.py OPV_exp_data.csv --name-column "No." --smiles-column "p(SMILES)" --output-dir output_p
python polymer-unit-classify.py --ring-total-list output_p/ring_total_list.csv --index-data output_p/index_data.csv --output-dir output_p
```

## Output files

Running `get-polymer-unit.py` produces:

- `ring_total_list.csv`
- `one_hot.csv`
- `number.csv`
- `adjacent_matrix.csv`
- `node_matrix.csv`
- `index_data.csv`

Running `polymer-unit-classify.py` produces:

- `ring_df.csv`
- `type_frame.csv`

## Downstream ML models

The repository also includes baseline regression scripts for OPV prediction:

- `RF.py`
- `KRR.py`
- `SVM.py`

These scripts can be run on:

- generated PUFp feature tables such as `output_n/number.csv`, or
- a custom mixed-feature table that already combines PUFp with other
  descriptors (for example RDKit/topological/electronic descriptors), as long
  as the feature table can be aligned to the target table through a shared id
  column.

Example using the donor-side PUFp table `output_n/number.csv`:

```bash
python RF.py --feature-csv output_n/number.csv --target-csv OPV_exp_data.csv --id-column "No." --target-column "PCE_max(%)" --quick
python KRR.py --feature-csv output_n/number.csv --target-csv OPV_exp_data.csv --id-column "No." --target-column "PCE_max(%)" --quick
python SVM.py --feature-csv output_n/number.csv --target-csv OPV_exp_data.csv --id-column "No." --target-column "PCE_max(%)" --quick
```

If you want to reproduce an older workflow based on a mixed descriptor table,
replace `output_n/number.csv` with your prepared feature table. The current
scripts no longer hard-code repository-specific filenames such as
`T_15_fea_descriptor_HOMO_LUMO.csv`, `1.csv`, or `data.csv`.

## Notes

- The current repository now makes input-column selection explicit for OPV data.
- The workflow preserves the original recognition logic and output structure.
- The ML scripts now use a consistent input interface and are not restricted to pure PUFp inputs.
- The original project history suggests that some downstream experiments may
  have used mixed feature tables rather than PUFp alone, so users should choose
  the feature table that matches their intended experiment design.

## Citation

If you use this repository in your research, please cite:

[1] Xinyue Zhang, Ye Sheng, Xiumin Liu, Jiong Yang, William A. Goddard III, Caichao Ye*, Wenqing Zhang*. Polymer-unit Graph: Advancing Interpretability in Graph Neural Network Machine Learning for Organic Polymer Semiconductor Materials. J. Chem. Theory Comput., 2024, 20(7), 2908-2920.  
[2] Xinyue Zhang, Genwang Wei, Ye Sheng, Wenjun Bai, Jiong Yang, Wenqing Zhang*, Caichao Ye*. Polymer-Unit Fingerprint (PUFp): An Accessible Expression of Polymer Organic Semiconductors for Machine Learning. ACS Appl. Mater. Interfaces, 2023, 15(17), 21537-21548.  
[3] Xiumin Liu, Xinyue Zhang, Ye Sheng, Zihe Zhang, Pan Xiong*, Xuehai Ju*, Junwu Zhu, Caichao Ye*. Advancing Organic Photovoltaic Materials by Machine Learning-Driven Design with Polymer-Unit Fingerprints. npj Comput. Mater., 2025, 11, 107.  
[4] Caichao Ye, Tao Feng, Weishu Liu*, Wenqing Zhang*. Functional Unit: A New Perspective on Materials Science Research Paradigms. Acc. Mater. Res., 2025, 6(8), 914-920.

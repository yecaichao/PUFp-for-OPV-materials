# Installation and Deployment

This repository was reviewed with the packaged OPV dataset
`OPV_exp_data.csv`.

## Create the environment

```bash
conda env create -f environment_purs.yml
conda activate purs-review-py39
```

## Recommended workflow

Run the recognition workflow separately for the SMILES column you want to
analyze.

Example for `n(SMILES)`:

```bash
python get-polymer-unit.py OPV_exp_data.csv --name-column "No." --smiles-column "n(SMILES)" --output-dir output_n
python polymer-unit-classify.py --ring-total-list output_n/ring_total_list.csv --index-data output_n/index_data.csv --output-dir output_n
```

Example for `p(SMILES)`:

```bash
python get-polymer-unit.py OPV_exp_data.csv --name-column "No." --smiles-column "p(SMILES)" --output-dir output_p
python polymer-unit-classify.py --ring-total-list output_p/ring_total_list.csv --index-data output_p/index_data.csv --output-dir output_p
```

## Run the downstream ML scripts

After generating a PUFp feature table, for example `output_n/number.csv`, you
can run the packaged regression baselines:

```bash
python RF.py --feature-csv output_n/number.csv --target-csv OPV_exp_data.csv --id-column "No." --target-column "PCE_max(%)" --quick
python KRR.py --feature-csv output_n/number.csv --target-csv OPV_exp_data.csv --id-column "No." --target-column "PCE_max(%)" --quick
python SVM.py --feature-csv output_n/number.csv --target-csv OPV_exp_data.csv --id-column "No." --target-column "PCE_max(%)" --quick
```

You can also use a custom mixed-feature table instead of pure PUFp, provided
that:

- the feature table contains one row per sample,
- the sample ids can be aligned to `OPV_exp_data.csv`, and
- the target column is available in the target table.

This is important because the original project scripts appear to have used
different prepared feature tables in some experiments, not only raw PUFp
outputs.

## Notes

- The workflow preserves the original recognition logic and output structure.
- `get-polymer-unit.py` now supports `--name-column`, `--smiles-column`, and `--output-dir`.
- `polymer-unit-classify.py` now supports explicit input paths and an output directory.
- `RF.py`, `KRR.py`, and `SVM.py` now share the same dataset-loading interface.
- For historical experiments, users may still prefer to provide a mixed
  descriptor table instead of `output_n/number.csv` or `output_p/number.csv`.

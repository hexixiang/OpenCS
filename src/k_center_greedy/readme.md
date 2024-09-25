# k_center_greedy

## Overview

The `k_center_greedy` module provides tools for performing k-center greedy sampling, which is a method used to select a subset of data points that minimizes the maximum distance of any point to a center. In this project, the module is used for simple clustering to compare with tag-based clustering methods.

## `run.py`

The `run.py` script is designed to perform k-center greedy sampling on a given dataset using BERT embeddings. It includes functions to compute embeddings, sample data points, and determine the optimal number of clusters (K) using the elbow method.

### Usage

1. **Prepare the Data**: Prepare a CSV file with the data you want to process.
2. **Set Paths**: You can now provide the paths for the BERT model, embedding file, input file, and output file prefix as command-line arguments:

   ```bash
   python run.py --bert_model_path <path_to_bert_model> --bert_embedding_path <path_to_bert_embedding> --input_file <path_to_input_csv> --output_file_prefix <output_file_prefix>
   ```
3. **Run the Script**: Execute the script to perform k-center greedy sampling and determine the optimal K value.

   ```bash
   python run.py --bert_model_path res/bert-base-uncased --bert_embedding_path data/alpaca/bert_embedding.npy --input_file data/alpaca/alpaca.csv --output_file_prefix results/kcenter_greedy/alpaca_kcenter_greedy
   ```

For more details on the command-line arguments, refer to the `parse_args` function in the script.

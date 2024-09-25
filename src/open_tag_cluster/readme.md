# Open-tag Clustering

## Overview

This folder contains the experimental code related to Open-tag Clustering, mainly divided into three parts: tag generation, tag clustering, and tag post-processing. The corresponding files are:

* Tag Generation: tag_generate.py
* Tag Clustering: tag_cluster.py
* Tag Post-Processing: tag_process.py

## tag_cluster

This script is designed to cluster generated tags in order to reduce the number of tags. It reads input data, processes the tags, and uses the specified clustering algorithm to cluster the tags, finally outputting the results to a specified JSON file.

### Usage

### Running the Script

1. Prepare the input files:

* Input JSONL file: A file containing JSON lines with tags.

2. Run the script using the following command:

```shell
python src/tag_based_cluster/tag_cluster.py --dataset_type <dataset_type> --dataset <dataset> --bert_model_path <path_to_bert_model> --phrasebert_model_path <path_to_phrasebert_model> --alpha <alpha_value> --embedding_type <embedding_type> --methods <clustering_method> --n_clusters <number_of_clusters> --eps <epsilon_value> --min_samples <minimum_samples>
```

Replace `<dataset_type>`, `<dataset>`, `<path_to_bert_model>`, `<path_to_phrasebert_model>`, `<alpha_value>`, `<embedding_type>`, `<clustering_method>`, `<number_of_clusters>`, `<epsilon_value>`, and `<minimum_samples>` with the actual parameter values.

### Output

The script will generate a JSON file containing the clustered tag results. Each line in the output file corresponds to a processed tag from the input file.


## tag_generate

This script is designed to generate tags for each instruction using a large language model (here we use gpt-4o-mini). It reads input data, processes each instruction through the LLM, and outputs the results in a specified JSON file.

### Usage

### Running the Script

1. Prepare the input files:

* Input JSONL file: A file containing JSON lines with instructions.
* Prompt file: A file containing the prompt template to be used by the LLM.

2. Run the script using the following command:

```shell
python src/tag_based_cluster/tag_generate.py --input_file <path_to_input_file> --output_file <path_to_output_file> --prompt_file <path_to_prompt_file>
```

Replace <path\_to\_input\_file>, <path\_to\_output\_file>, and <path\_to\_prompt\_file> with the actual paths to your files.

### Output

The script will generate a JSON file with the results of the LLM processing. Each line in the output file corresponds to a processed instruction from the input file.

### Error Handling

If an error occurs during the LLM processing, the script will retry up to three times. If all attempts fail, the problematic instruction will be saved to a bad\_case.jsonl file for further inspection.

## tag_process

This script is designed to process the clustered tags and generate a mapping of labels to their corresponding instructions. It reads input data, extracts the longest instruction for each unique set of labels, and outputs the results in a specified JSON file.

### Usage

### Running the Script

1. Prepare the input files:

* Input JSONL file: A file containing JSON lines with clustered tags.

2. Run the script using the following command:

```shell
python src/tag_based_cluster/tag_process.py <path_to_input_file> <path_to_intermediate_file> <path_to_output_file>
```

Replace `<path_to_input_file>`, `<path_to_intermediate_file>`, and `<path_to_output_file>` with the actual paths to your files.

### Output

The script will generate two JSON files:
1. An intermediate JSON file containing the longest instruction for each unique set of labels.
2. A final JSON file containing the mapping of labels to their corresponding instructions. Each entry in the output file corresponds to a label and its associated instructions.

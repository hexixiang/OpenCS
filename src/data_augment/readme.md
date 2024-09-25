# data_augment

## Overview

The `data_augment` module provides tools for augmenting data using OpenAI's GPT models. It includes functions to process data, generate better responses and instructions, and handle multiple API keys for parallel processing.

## `main.py`

The `main.py` script is designed to process data augmentation on a given dataset using OpenAI's GPT models. It includes functions to read data, call the OpenAI API, and process each data item to generate improved responses and instructions.

### Usage
1. **Set API Key and Base**: Set the API key and base URL for the OpenAI API.
```python
# an example
openai_api_key = [
    "sk-xxxx",
    "sk-xxxx",
    "sk-xxxx",
    "sk-xxxx",
    "sk-xxxx"
    ]

openai_api_base = "https://xxxx/v1"
```
2. **Prepare the Data**: Prepare a JSON file with the data you want to process.
3. **Set Paths**: You can now provide the paths for the input data file and the output data file as command-line arguments:

   ```bash
   python main.py --input_path <path_to_input_json> --output_path <path_to_output_json>
   ```
4. **Run the Script**: Execute the script to perform data augmentation and generate improved responses and instructions.

   ```bash
   python main.py --input_path data/input.json --output_path results/output.json
   ```

For more details on the command-line arguments, refer to the `parse_args` function in the script.

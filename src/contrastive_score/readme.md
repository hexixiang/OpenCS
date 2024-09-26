Here's a more detailed README file for your project:

```markdown
# Simple Double Processing

## Overview

This project processes JSON files, generating responses using a language model API based on predefined prompts. It saves the results to specified output files, facilitating easy data management and analysis.

## Features

- Reads prompts from text files.
- Processes multiple JSON files from a specified input directory.
- Randomly samples data from the input files.
- Generates different types of prompts (single, double, few-shot).
- Interacts with a language model API to generate responses.
- Saves results in structured JSON format.

## Requirements

- Python 3.x
- Libraries: 
  - `json`
  - `random`
  - `os`
  - `api` (custom module for API interactions)
  - `tqdm` (for progress tracking)

## Installation

1. **Clone the Repository**  
   Use git to clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/repo.git
   cd repo
   ```

2. **Install Required Libraries**  
   Make sure you have all the required libraries installed. You can install any missing libraries using pip:
   ```bash
   pip install tqdm
   ```

## Usage

### Step 1: Update the Script

- Open the script file (e.g., `your_script.py`).
- Modify the `IN_DIR` and `OUT_DIR` variables to point to your input and output directories, respectively:
   ```python
   IN_DIR = r"E:\path\to\input\data"
   OUT_DIR = r"E:\path\to\output\results"
   ```

- Replace the sensitive API information in the script with your actual credentials:
   ```python
   openai_api_key = "xxx"
   openai_api_base = "http://xxx/v1"
   ```

### Step 2: Prepare Input Files

- Place your JSON files in the specified input directory.
- Ensure that your prompt files are correctly formatted and located in the specified paths within the script.

### Step 3: Run the Script

Execute the script in your terminal:
```bash
python your_script.py
```

### Step 4: Access Output

The generated output files will be saved in the specified output directory. Each file will contain the API responses based on the prompts generated from your input data.

## How It Works

1. **Prompt Reading**  
   The script reads prompts from specified text files, which dictate the format of the API requests.

2. **Data Processing**  
   For each JSON file in the input directory, the script:
   - Loads the data.
   - Randomly samples a subset of entries.
   - Prepares prompts based on the sampled data.

3. **API Interaction**  
   The script interacts with a language model API, sending the generated prompts and receiving responses.

4. **Output Generation**  
   Responses are written to output files, structured in a JSON format for easy accessibility and further analysis.

## Example File Structure

```
/your_project_directory
│
├── your_script.py
├── prompts/
│   ├── simple_en.txt
│   ├── double.txt
│   ├── simple_few-shot.txt
│   └── double_few-shot.txt
└── data/
    ├── input_data1.json
    ├── input_data2.json
    └── ...
```

## Troubleshooting

- Ensure all file paths are correctly set and accessible.
- If you encounter API-related issues, verify your API key and endpoint.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes.
```

Feel free to customize any sections to better fit your project's specifics!
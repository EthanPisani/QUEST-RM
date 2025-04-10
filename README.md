# QUEST-RM Roleplay | Dataset Processing and Ranking System

This comprehensive system provides tools for processing roleplay datasets, exploring their contents, ranking roleplay interactions (both manually and automatically), and analyzing the results. The system consists of several interconnected components that work together to create a ranked dataset suitable for training roleplay language models.

## Table of Contents

1. [System Overview](#system-overview)
2. [Installation and Requirements](#installation-and-requirements)
3. [Component Descriptions](#component-descriptions)
   - [Dataset Preprocessor](#dataset-preprocessor)
   - [Dataset Explorer](#dataset-explorer)
   - [Human Ranking Web UI](#human-ranking-web-ui)
   - [Automatic Ranking System](#automatic-ranking-system)
   - [Dataset Analysis Tools](#dataset-analysis-tools)
   - [Model Performance Analysis](#model-performance-analysis)
4. [Workflow](#workflow)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Output Formats](#output-formats)

## System Overview

This system processes raw roleplay datasets from various sources, converts them into a standardized format, and provides both human and automated ranking capabilities. The ranked data can then be used for training or fine-tuning roleplay language models.

Key features:
- Multi-source dataset processing
- HTML cleaning and normalization
- Human ranking interface with Flask web UI
- Automated ranking with LLM evaluation
- Dataset analysis and visualization
- Parallel processing for efficiency

## Installation and Requirements

### Prerequisites

- Python 3.8+
- pip package manager
- CUDA-capable GPU (for LLM ranking)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/EthanPisani/QUEST-RM.git
cd roleplay-ranking-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages

Core dependencies:
- pandas
- flask
- tqdm
- requests
- scikit-learn
- matplotlib
- pyarrow (for parquet support)
- pydantic

## Component Descriptions

### Dataset Preprocessor

File: `dataset_preprocessed.py`

#### Purpose
Processes raw roleplay datasets from multiple sources into a standardized format suitable for ranking and analysis.

#### Key Features
- Handles multiple dataset formats (CSV, JSONL)
- Cleans HTML tags and special characters
- Groups messages by conversation thread
- Creates user-assistant message pairs
- Outputs to efficient Parquet format

#### Usage
```bash
python dataset_preprocessed.py
```

#### Configuration
Modify the `datasets` list at the top of the file to include your dataset paths and types:
```python
datasets = [
    {
        "path": '/path/to/dataset',
        "name": 'dataset_name',
        "type": 'csv'  # or 'jsonl'
    }
]
```

#### Output
- `./processed/combined_dataset.parquet`: Combined processed dataset

### Dataset Explorer

File: `explore_dataset.py`

#### Purpose
Provides tools for examining the processed dataset, sampling data, and preparing subsets for ranking.

#### Key Features
- Displays dataset statistics
- Creates manageable sample sizes
- Ensures no overlap between samples
- Interactive exploration

#### Usage
```bash
python explore_dataset.py
```

#### Configuration
Modify these variables at the top of the file:
```python
parquet_file = './processed/combined_dataset.parquet'  # Input file
output_file = './processed/sample_dataset.parquet'    # Output sample file
sample_size = 1000000                                # Number of samples
```

#### Output
- Sample datasets in Parquet format (e.g., `sample_dataset.parquet`)

### Human Ranking Web UI

File: `rank_roleplay_llm_async.py`

#### Purpose
Provides a web interface for humans to rank roleplay interactions across multiple quality dimensions.

#### Key Features
- Flask-based web interface
- Displays roleplay conversations
- Collects ratings on 6 quality dimensions
- Async LLM ranking for comparison
- Persistent storage of rankings

#### Usage
```bash
python rank_roleplay_llm_async.py
```

#### Configuration
Key configuration options:
```python
model = "deepseek-ai_DeepSeek-R1-Distill-Llama-8B-exl2"  # Model for async ranking
OPENAI_API_URL = "http://10.0.9.12:5008/v1/completions"   # API endpoint
parquet_file = "processed/sample_dataset.parquet"         # Dataset to rank
```

#### Accessing the UI
After starting the server, access the interface at:
```
http://localhost:9301
```

#### Output
- `rankings.parquet`: Contains all human rankings

### Automatic Ranking System

File: `auto_rank_batching.py`

#### Purpose
Automatically ranks roleplay interactions using an LLM to generate synthetic ranking data.

#### Key Features
- Batch processing of large datasets
- Multi-threaded for efficiency
- Resume capability
- Uses human rankings as examples
- Comprehensive evaluation criteria

#### Usage
```bash
python auto_rank_batching.py
```

#### Configuration
Key configuration options:
```python
model = "DeepSeek-R1-Distill-Qwen-14B-exl2"       # Model for ranking
OPENAI_API_URL = "http://127.0.0.1:5009/v1/completions"  # API endpoint
parquet_file = "processed/sample_dataset_2.parquet"      # Dataset to rank
rankings_parquet_file = "auto_rankings5.parquet"         # Output file
human_rankings_parquet_file = "rankings.parquet"         # Human examples
```

#### Output
- Auto-generated rankings in Parquet format (e.g., `auto_rankings5.parquet`)

### Dataset Analysis Tools

File: `dataset_analysis.py`

#### Purpose
Analyzes and visualizes the ranked datasets to understand quality distributions and relationships.

#### Key Features
- Statistical summaries
- Distribution visualizations
- Correlation analysis
- PCA for dimensionality reduction
- Text length analysis

#### Usage
```bash
python dataset_analysis.py
```

#### Configuration
Modify the file paths at the top of the file:
```python
file_paths = {
    "Dataset1": "../datasets/model1/model1_dataset1.parquet",
    "Dataset2": "../datasets/model2/model2_dataset2.parquet",
    # ...
}
```

#### Output
Various visualization files including:
- `*_boxplot.png`: Box plots of scores
- `*_side_by_side_histogram.png`: Score distributions
- `*_correlation_matrix.png`: Feature correlations
- `pca_numeric_scores.png`: PCA visualization
Here's the edited README with the new MODEL_ANALYSIS section and brief overviews of the Spearman analysis files:

### Model Performance Analysis

For analyzing model performance across benchmarks and attributes, see the detailed [MODEL_ANALYSIS.md](MODEL_ANALYSIS.md) documentation.

Key analysis scripts:

1. **spearman_multi_4.py**
   - Compares model rankings across multiple benchmarks
   - Calculates 4 correlation metrics:
     - Spearman Rank Correlation
     - Kendall Tau Correlation  
     - Mean Absolute Rank Difference (MAR)
     - Rank-Biased Overlap (RBO)
   - Generates comprehensive visualizations
   - Usage: 
   ```
   python spearman_multi_4.py --ranking_files bench1.txt bench2.csv model_results.csv
   ```

2. **spearman_multi_4_attributes.py**  
   - Analyzes correlations between different performance attributes within a single model
   - Processes CSV files with multiple numeric columns
   - Same 4 metrics as above but for intra-model analysis
   - Usage: 
   ```
   python spearman_multi_4_attributes.py --ranking_files model_details.csv
   ```

## Workflow

1. **Dataset Preparation**
   - Run `dataset_preprocessed.py` to process raw datasets
   - Use `explore_dataset.py` to create manageable samples

2. **Human Ranking (Optional but Recommended)**
   - Start `rank_roleplay_llm_async.py`
   - Access the web interface to provide human rankings
   - These serve as quality examples for the auto-ranker

3. **Automatic Ranking**
   - Configure `auto_rank_batching.py` with desired model
   - Run to generate synthetic rankings at scale

4. **Analysis**
   - Use `dataset_analysis.py` to examine results
   - Visualizations help understand data quality

## Configuration

### API Configuration

For the ranking components, you'll need to configure:

1. **API Endpoint**: Set the `OPENAI_API_URL` to your LLM server
2. **Model Selection**: Choose appropriate models in both ranking scripts
3. **API Key**: Set `OPENAI_API_KEY` if required

### Dataset Paths

All scripts need proper paths configured:
- Input dataset locations
- Output directories (ensure they exist)
- Ranking storage files

### Performance Tuning

For `auto_rank_batching.py`:
- Adjust `batch_size` based on your hardware
- Modify `max_workers` in `ThreadPoolExecutor`
- Consider smaller samples if memory is limited

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Ensure all packages are installed from requirements.txt
   - Some components may need additional CUDA libraries

2. **File Path Errors**
   - Verify all paths exist
   - Check permissions on directories

3. **API Connection Issues**
   - Test your LLM server independently
   - Verify API keys and URLs

4. **Memory Errors**
   - Reduce sample sizes
   - Use smaller batch sizes
   - Consider processing in chunks

### Logging

- Most scripts provide progress bars and console output
- Failed requests are logged in auto_rank_batching.py
- The web UI shows errors in the browser console

## Output Formats

### Processed Dataset Format

The processed dataset contains these columns:
- `dataset`: Source dataset name
- `title`: Conversation title
- `message`: JSON string of message pairs (user/assistant)

### Ranking Format

Ranking files contain:
- `roleplay_id`: Original message ID
- `dataset`: Source dataset
- `title`: Conversation title
- `message`: Full message JSON
- Six scoring columns (1.0-10.0 scale)

### Analysis Outputs

The analysis script generates:
- Statistical summaries (console)
- Visualization PNG files
- PCA coordinates for quality dimensions

## Conclusion

This system provides a complete pipeline for creating high-quality ranked roleplay datasets suitable for training advanced language models. The combination of human and automated ranking ensures both quality and scale, while the analysis tools help understand the resulting dataset characteristics.
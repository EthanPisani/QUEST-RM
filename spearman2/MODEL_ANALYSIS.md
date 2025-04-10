# Model Performance Analysis

This section describes how to analyze model performance using Spearman correlation metrics, comparing rankings across different benchmarks and model attributes.

## Model Comparison Analysis

File: `spearman_multi_4.py`

### Purpose
Compares model rankings across multiple benchmarks using several correlation metrics to understand how consistently models perform across different evaluation criteria.

### Key Features
- Processes multiple ranking files (TXT and CSV formats)
- Calculates four comparison metrics:
  - Spearman Rank Correlation
  - Kendall Tau Correlation
  - Mean Absolute Rank Difference (MAR)
  - Rank-Biased Overlap (RBO)
- Generates comprehensive visualizations
- Handles partial overlaps between rankings
- Provides detailed console output

### Usage
```bash
python spearman_multi_4.py --ranking_files MMLU_pro.txt ping_pong_v2.txt rpbench.txt lm_arena.txt model1_g.csv model2_g.csv model3_g.csv model4_g.csv
```

### Input Formats
1. **TXT files**: Simple text files with one model name per line
   ```
   model_a
   model_b
   model_c
   ```

2. **CSV files**: Structured files with model names and rankings
   ```
   Model,rank,score
   model_a,1,0.95
   model_b,2,0.92
   model_c,3,0.89
   ```

### Output
- Visual comparison plot (`ranking_comparison.png` by default)
- Console output with all correlation matrices
- Common model counts between each pair of rankings

### Configuration
Modify these aspects of the script:
- `plt.rcParams['figure.figsize']`: Adjust plot dimensions
- `p=0.9` in `rbo_score()`: Change RBO persistence parameter
- Heatmap color schemes (`cmap` parameters)

### Interpretation of Metrics
1. **Spearman Rank Correlation**:
   - Measures monotonic relationship between rankings
   - Range: -1 (perfect inverse) to 1 (perfect agreement)
   - Values near 0 indicate no correlation

2. **Kendall Tau Correlation**:
   - Measures ordinal association
   - More robust to small ranking changes than Spearman
   - Same -1 to 1 range

3. **Mean Absolute Rank Difference (MAR)**:
   - Average absolute difference in ranks
   - Lower values indicate better agreement
   - Scale depends on number of models

4. **Rank-Biased Overlap (RBO)**:
   - Measures similarity with emphasis on top ranks
   - Range: 0 (no overlap) to 1 (identical)
   - Weighted by persistence parameter (p=0.9)

## Model Attribute Analysis

File: `spearman_multi_4_attributes.py`

### Purpose
Analyzes correlations between different performance attributes within a single model's evaluation results.

### Key Features
- Processes CSV files with multiple numeric columns
- Automatically detects model name column
- Creates rankings for each numeric attribute
- Same four comparison metrics as above
- Special handling for attribute comparisons

### Usage
```bash
python spearman_multi_4_attributes.py --ranking_files model3_g.csv
```

### Input Format
CSV files with multiple numeric columns representing different attributes:
```
Model,Contextual_Alignment,Character_Consistency,Descriptive_Depth
model_a,8.2,7.5,9.1
model_b,7.8,8.3,8.7
model_c,6.5,7.1,7.9
```

### Output
- Visual comparison plot (`ranking_comparison_attributes.png` by default)
- Console output with all correlation matrices
- Shows how different quality attributes correlate within a model

### Key Differences from Model Comparison
- Designed for intra-model analysis rather than inter-model
- Automatically processes all numeric columns
- Simplified annotations (no count display)
- Assumes higher numeric values are better

## Workflow for Model Analysis

1. **Collect Benchmark Results**:
   - Run models through various benchmarks (MMLU, RPBench, etc.)
   - Save results in consistent format (TXT or CSV)

2. **Run Model Comparison**:
   ```bash
   python spearman_multi_4.py --ranking_files benchmark1.txt benchmark2.csv model_results.csv
   ```

3. **Run Attribute Analysis**:
   ```bash
   python spearman_multi_4_attributes.py --ranking_files model_details.csv
   ```

4. **Interpret Results**:
   - High correlations between benchmarks suggest consistent model performance
   - Low correlations may indicate benchmark-specific strengths
   - Attribute correlations show which qualities tend to co-occur

## Example Use Cases

1. **Model Selection**:
   - Identify models that perform consistently well across all benchmarks
   - Find specialists that excel in specific areas

2. **Benchmark Evaluation**:
   - Assess how similar different benchmarks are in their evaluations
   - Identify redundant benchmarks

3. **Quality Attribute Analysis**:
   - Understand relationships between different model qualities
   - Identify trade-offs between attributes

4. **Model Development**:
   - Track how model improvements affect different benchmarks
   - Verify balanced improvement across attributes

## Troubleshooting

### Common Issues

1. **File Format Errors**:
   - Ensure CSV files have proper headers
   - Verify TXT files have one model per line

2. **Missing Models**:
   - Some metrics will show NaN when comparing rankings with no common models

3. **Memory Errors**:
   - Reduce number of rankings being compared simultaneously
   - For very large model sets, consider comparing subsets

4. **Plot Rendering Issues**:
   - Adjust figure size if labels overlap
   - Modify font sizes if needed

### Output Interpretation

- **High Spearman/Kendall, Low MAR, High RBO**: Strong agreement
- **Negative correlations**: Inverse relationship (rare in model comparisons)
- **Near-zero correlations**: No relationship between rankings
- **N/A values**: No common models between those rankings
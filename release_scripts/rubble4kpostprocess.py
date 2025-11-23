import pandas as pd
import os
import sys

def create_psnr_table_from_csv(csv_path, output_md=None):
    """
    Read experiment_results.csv and create a table with model size, offload strategy, and metrics.
    Extracts: test_psnr, train_psnr, num_3dgs, max_gpu_memory_gb, pinned_cpu_memory_gb, total_time_s
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} does not exist")
        return
    
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Print available columns for debugging
    print(f"Available columns in CSV: {list(df.columns)}")
    print()
    
    # Parse experiment names to extract model_size and offload_strategy
    # Format: TIMESTAMP_TIMESTAMP_MODELSIZE_OFFLOADSTRATEGY
    # e.g., 20251111_231509_small_clm_offload
    def parse_experiment_name(exp_name):
        parts = exp_name.split('_')
        if len(parts) >= 3:
            # Try to match the offload pattern at the end
            exp_lower = exp_name.lower()
            if exp_lower.endswith('_clm_offload'):
                offload_strategy = 'clm_offload'
                remainder = exp_name[:-len('_clm_offload')]
            elif exp_lower.endswith('_naive_offload'):
                offload_strategy = 'naive_offload'
                remainder = exp_name[:-len('_naive_offload')]
            elif exp_lower.endswith('_no_offload'):
                offload_strategy = 'no_offload'
                remainder = exp_name[:-len('_no_offload')]
            else:
                # Fallback: assume last part is offload strategy
                offload_strategy = parts[-1]
                remainder = '_'.join(parts[:-1])
            
            # Remove timestamp prefix (first 2 parts: date_time)
            remainder_parts = remainder.split('_')
            if len(remainder_parts) >= 3:
                model_size = '_'.join(remainder_parts[2:])
            else:
                model_size = remainder
            
            # Create experiment identifier combining model_size and offload_strategy
            experiment_id = f"{model_size}_{offload_strategy}"
            
            return model_size, offload_strategy, experiment_id
        return None, None, None
    
    # Add model_size, offload_strategy, and experiment_id columns
    df[['model_size', 'offload_strategy', 'experiment_id']] = df['experiment'].apply(
        lambda x: pd.Series(parse_experiment_name(x))
    )
    
    # Select the columns we want to display
    columns_to_extract = ['experiment_id', 'test_psnr', 'train_psnr', 'num_3dgs', 
                          'max_gpu_memory_gb', 'pinned_cpu_memory_gb', 'total_time_s']
    
    # Filter to only available columns
    available_columns = [col for col in columns_to_extract if col in df.columns]
    result_df = df[available_columns].copy()
    
    # Add fake OOM rows for comparison
    oom_rows = [
        {'experiment_id': 'rubble4k_28M_no_offload'},
    ]
    
    # Create OOM dataframe with same columns
    oom_df = pd.DataFrame(oom_rows)
    for col in available_columns:
        if col not in oom_df.columns:
            oom_df[col] = 'OOM'
    
    # Ensure column order matches
    oom_df = oom_df[available_columns]
    
    # Append OOM rows to result
    result_df = pd.concat([result_df, oom_df], ignore_index=True)
    
    # Round numeric columns for cleaner display (only for non-OOM rows)
    numeric_columns = ['test_psnr', 'train_psnr', 'max_gpu_memory_gb', 'pinned_cpu_memory_gb', 'total_time_s']
    for col in numeric_columns:
        if col in result_df.columns:
            # Convert to numeric, coercing non-numeric (OOM) to NaN, then back to original with OOM
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').round(2)
            result_df[col] = result_df[col].fillna('OOM')
    
    # Round num_3dgs to integer if present (only for non-OOM rows)
    if 'num_3dgs' in result_df.columns:
        result_df['num_3dgs'] = pd.to_numeric(result_df['num_3dgs'], errors='coerce').round(0)
        result_df['num_3dgs'] = result_df['num_3dgs'].fillna('OOM')
        # Convert back to int where possible
        result_df['num_3dgs'] = result_df['num_3dgs'].apply(lambda x: int(x) if x != 'OOM' else x)
    
    # Rename columns for better display
    column_rename = {
        'experiment_id': 'Experiment',
        'test_psnr': 'Test PSNR',
        'train_psnr': 'Train PSNR',
        'num_3dgs': 'Num 3DGS',
        'max_gpu_memory_gb': 'Max GPU Memory (GB)',
        'pinned_cpu_memory_gb': 'Pinned CPU Memory (GB)',
        'total_time_s': 'Training Time (s)'
    }
    result_df = result_df.rename(columns=column_rename)
    
    # Generate markdown table
    markdown_table = result_df.to_markdown(index=False)
    
    # Save to file if output path specified
    if output_md is None:
        output_md = csv_path.replace('.csv', '_results_table.md')
    
    with open(output_md, 'w') as f:
        f.write("# Rubble 4K Experiment Results\n\n")
        f.write("## Performance Metrics by Model Size and Offload Strategy\n\n")
        f.write(markdown_table)
        f.write("\n")
    
    print(f"\nSuccessfully created results table and saved to {output_md}")
    print("\n" + "="*80)
    print("Rubble 4K Experiment Results")
    print("="*80)
    print(markdown_table)
    print("="*80)
    
    return result_df

if __name__ == "__main__":
    # read args from command line
    if len(sys.argv) < 2:
        print("Usage: python rubble4kpostprocess.py <csv_path> [output_md]")
        print("  csv_path: Path to experiment_results.csv file")
        print("  output_md: (optional) Path to output markdown file")
        print("\nExample:")
        print("  python rubble4kpostprocess.py experiment_results.csv")
        print("  python rubble4kpostprocess.py experiment_results.csv results_table.md")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_md = sys.argv[2] if len(sys.argv) > 2 else None
    
    create_psnr_table_from_csv(csv_path, output_md)

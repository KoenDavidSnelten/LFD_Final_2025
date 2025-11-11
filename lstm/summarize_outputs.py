#!/usr/bin/env python3
import re
from pathlib import Path
import argparse # 1. Import argparse

# --- Configuration ---
# Set a default path which will be overridden by the --path argument if provided
DEFAULT_BASE_DIR = Path('/scratch/s4568966/LFD_Final_2025/lstm/output/other')
# --- End Configuration ---


def parse_config(content):
    """Extracts hyperparameters from the Namespace(...) section."""
    config = {}
    ns_match = re.search(r'Namespace\((.*?)\)', content, re.DOTALL)
    if not ns_match:
        return None

    ns_text = ns_match.group(1)

    def extract(key, cast_type=str):
        m = re.search(rf'{key}=([^\s,)\]]+)', ns_text)
        if not m:
            return None
        val = m.group(1).strip().strip("'\"")
        try:
            return cast_type(val)
        except Exception:
            return val

    config['lr'] = extract('learning_rate', float)
    config['batch_size'] = extract('batch_size', int)
    config['max_length'] = extract('lstm_units', int) # or 'max_length' if applicable
    config['patience'] = extract('epochs', int)

    # Additional LSTM-related parameters
    config['dropout'] = extract('dropout', float)
    config['recurrent_dropout'] = extract('recurrent_dropout', float)
    config['lstm_layers'] = extract('lstm_layers', int)
    config['bidirectional'] = extract('bidirectional_layer', str)

    return config


def parse_metrics(content):
    """
    Parses '--- Evaluation on dev set ---' and '--- Evaluation on test set ---'
    blocks and extracts Accuracy and Macro F1 Score.
    """
    def extract_block_metrics(block_text):
        acc_match = re.search(r'Accuracy:\s*([0-9.]+)', block_text)
        macro_f1_match = re.search(r'Macro F1 Score:\s*([0-9.]+)', block_text)
        if not acc_match or not macro_f1_match:
            return None
        return {
            'accuracy': float(acc_match.group(1)),
            'f1_macro': float(macro_f1_match.group(1)),
        }

    dev_block = re.search(r'--- Evaluation on dev set ---([\s\S]*?)---', content)
    test_block = re.search(r'--- Evaluation on test set ---([\s\S]*?)---', content)

    dev_metrics = extract_block_metrics(dev_block.group(1)) if dev_block else None
    test_metrics = extract_block_metrics(test_block.group(1)) if test_block else None

    return dev_metrics, test_metrics


def main():
    # 2. Argument Parsing Setup
    parser = argparse.ArgumentParser(
        description="Parse results.txt files from hyperparameter search output directories."
    )
    parser.add_argument(
        '--path',
        type=str,
        default=str(DEFAULT_BASE_DIR), # Use the default path as the default argument value
        help="The base directory containing the run subdirectories (e.g., 'path/to/output/raw')."
    )
    args = parser.parse_args()

    # 3. Use the provided argument to set the final BASE_DIR
    BASE_DIR = Path(args.path)
    
    # 4. Main logic starts here
    all_results = []
    result_files = list(BASE_DIR.glob('*/results.txt'))

    if not result_files:
        print(f"Error: No 'results.txt' files found in {BASE_DIR}")
        return

    print(f'Found {len(result_files)} result files. Parsing...\n')

    for result_file in result_files:
        run_name = result_file.parent.name
        content = result_file.read_text()

        config = parse_config(content)
        dev_metrics, test_metrics = parse_metrics(content)

        if config and dev_metrics and test_metrics:
            all_results.append({
                'run': run_name,
                'config': config,
                'dev_metrics': dev_metrics,
                'test_metrics': test_metrics,
            })
        else:
            print(f'--- Skipping incomplete or failed run: {run_name} ---')

    sorted_results = sorted(
        all_results,
        key=lambda x: x['test_metrics'].get('f1_macro', 0),
        reverse=True,
    )

    print('\n--- Grid Search Results (Sorted by Test F1-Macro) ---')
    for run in sorted_results:
        cfg = run['config']
        dev = run['dev_metrics']
        test = run['test_metrics']

        print(f'\n=======================================================')
        print(f"RUN: {run['run']}")
        print(
            f"Config: LR={cfg['lr']}, BS={cfg['batch_size']}, LSTM Units={cfg['max_length']}, "
            f"Epochs={cfg['patience']}, Dropout={cfg['dropout']}, RecDrop={cfg['recurrent_dropout']}, "
            f"Layers={cfg['lstm_layers']}, Bi={cfg['bidirectional']}"
        )
        print('-------------------------------------------------------')
        print(f"  DEV  -> F1-Macro: {dev['f1_macro']:<6} | Accuracy: {dev['accuracy']:<6}")
        print(f"  TEST -> F1-Macro: {test['f1_macro']:<6} | Accuracy: {test['accuracy']:<6}")
        print(f'=======================================================')


if __name__ == '__main__':
    main()
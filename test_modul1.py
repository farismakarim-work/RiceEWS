"""
Test Script untuk MODUL 1 - Data Preprocessing
===============================================

Script ini mengtest semua functionality dari MODUL 1:
- Load pilot dataset dari data/raw/Pilot Dataset.xlsx
- Jalankan full preprocessing pipeline
- Generate output dan report

Cara menjalankan:
    python test_modul1.py

Requirements:
    - pandas
    - openpyxl
    - numpy
    - scipy
    - statsmodels
"""

import sys
from pathlib import Path
import logging

# Add src to path untuk import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from modules.preprocessing.data_preprocessor import run_full_preprocessing_pipeline


def main():
    """Main test function."""
    
    print("\n" + "="*70)
    print("MODUL 1 - DATA PREPROCESSING TEST")
    print("="*70)
    
    # Define paths
    input_file = Path(__file__).parent / "data" / "raw" / "Pilot Dataset.xlsx"
    output_file = Path(__file__).parent / "data" / "processed" / "preprocessed_pilot_data.csv"
    
    # Check if input file exists
    if not input_file.exists():
        print(f"\n✗ ERROR: Input file not found!")
        print(f"Expected: {input_file}")
        print(f"\nPlease ensure:")
        print(f"  1. You are on 'develop' branch")
        print(f"  2. data/raw/Pilot Dataset.xlsx exists")
        return False
    
    print(f"\n✓ Input file found: {input_file}")
    
    try:
        # Run preprocessing pipeline
        print("\nRunning preprocessing pipeline...")
        df_processed = run_full_preprocessing_pipeline(
            input_file=str(input_file),
            output_file=str(output_file),
            config={
                'missing_method': 'interpolate',
                'outlier_method': 'iqr',
                'outlier_threshold': 1.5,
                'detrend_method': 'linear',
                'differencing_order': 1,
                'standardize_method': 'zscore',
                'stationarity_test': 'adf'
            }
        )
        
        # Print success message
        print("\n" + "="*70)
        print("✓ TEST SUCCESSFUL!")
        print("="*70)
        
        # Print results
        print(f"\nOutput Results:")
        print(f"  Shape: {df_processed.shape}")
        print(f"  Columns: {df_processed.columns.tolist()}")
        print(f"  Output file: {output_file}")
        
        # Print data summary
        print(f"\nData Summary:")
        print(f"  Date range: {df_processed['date'].min()} to {df_processed['date'].max()}")
        print(f"  Markets: {sorted(df_processed['market_id'].unique().tolist())}")
        print(f"  Grades: {sorted(df_processed['grade'].unique().tolist())}")
        
        # Print sample data
        print(f"\nSample Data (first 10 rows):")
        print(df_processed.head(10).to_string())
        
        # Print statistics per grade
        print(f"\nStatistics per Grade:")
        for grade in sorted(df_processed['grade'].unique()):
            grade_data = df_processed[df_processed['grade'] == grade]
            print(f"\n  {grade}:")
            print(f"    Records: {len(grade_data):,}")
            print(f"    Price - Min: {grade_data['price'].min()}, Max: {grade_data['price'].max()}, Mean: {grade_data['price'].mean():.2f}")
            if 'price_diff' in df_processed.columns:
                valid_diff = grade_data['price_diff'].dropna()
                print(f"    Diff - Min: {valid_diff.min():.2f}, Max: {valid_diff.max():.2f}, Mean: {valid_diff.mean():.2f}")
        
        # Check output files
        print(f"\nOutput Files:")
        report_file = output_file.parent / f"{output_file.stem}_report.json"
        if output_file.exists():
            print(f"  ✓ {output_file} ({output_file.stat().st_size / 1024:.2f} KB)")
        else:
            print(f"  ✗ {output_file} NOT FOUND")
            
        if report_file.exists():
            print(f"  ✓ {report_file} ({report_file.stat().st_size / 1024:.2f} KB)")
        else:
            print(f"  ✗ {report_file} NOT FOUND")
        
        print(f"\n{'='*70}")
        print("Ready for MODUL 2: Granger Causality Testing")
        print(f"{'='*70}\n")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ FILE NOT FOUND ERROR: {e}")
        print(f"\nPlease check:")
        print(f"  1. Input file exists: data/raw/Pilot Dataset.xlsx")
        print(f"  2. You are on 'develop' branch")
        return False
        
    except ImportError as e:
        print(f"\n✗ IMPORT ERROR: {e}")
        print(f"\nPlease install required packages:")
        print(f"  pip install pandas openpyxl numpy scipy statsmodels scikit-learn")
        return False
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""
Test Script untuk MODUL 2 - Granger Causality Testing
=====================================================

Script ini mengtest functionality dari MODUL 2:
- Load preprocessed data dari MODUL 1
- Perform pairwise Granger causality tests
- Build causal matrices per grade
- Identify market leaders
- Generate output dalam MULTIPLE formats

Output Formats:
- JSON: Raw data structure
- CSV: Pairwise results per grade
- Excel: Summary + Pairwise + Matrices (multiple sheets)
- NPZ: NumPy binary format untuk matrices
- Markdown: Human-readable report

Cara menjalankan:
    python test_modul2.py

Requirements:
    - pandas
    - numpy
    - scipy
    - openpyxl (untuk Excel export)
    - Hasil dari MODUL 1 (preprocessed_pilot_data.csv)
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

from modules.causality_testing.granger_tester import run_full_granger_analysis


def main():
    """Main test function."""
    
    print("\n" + "="*70)
    print("MODUL 2 - GRANGER CAUSALITY TESTING (ALL OUTPUT FORMATS)")
    print("="*70)
    
    # Define paths
    input_file = Path(__file__).parent / "data" / "processed" / "preprocessed_pilot_data.csv"
    output_dir = Path(__file__).parent / "data" / "processed"
    
    # Check if input file exists
    if not input_file.exists():
        print(f"\n✗ ERROR: Input file not found!")
        print(f"Expected: {input_file}")
        print(f"\nPlease ensure:")
        print(f"  1. You ran MODUL 1 successfully")
        print(f"  2. Output file exists: preprocessed_pilot_data.csv")
        return False
    
    print(f"\n✓ Input file found: {input_file}")
    
    try:
        # Run Granger analysis
        print("\nRunning Granger causality analysis...")
        
        results = run_full_granger_analysis(
            input_file=str(input_file),
            output_dir=str(output_dir),
            config={
                'lag_order': 4,
                'price_col': 'price_diff',
                'significance_level': 0.05
            }
        )
        
        # Print success message
        print("\n" + "="*70)
        print("✓ TEST SUCCESSFUL!")
        print("="*70)
        
        # Print results summary
        print(f"\nGranger Causality Results Summary:")
        print(f"\nGrades analyzed: {list(results.keys())}\n")
        
        for grade, grade_data in results.items():
            print(f"{'Grade: ' + grade:^70}")
            print(f"{'-'*70}")
            
            # Market leaders
            leaders = grade_data['market_leaders']
            out_degrees = grade_data['out_degrees']
            in_degrees = grade_data['in_degrees']
            
            print(f"\nMarket Leaders (by influence - out-degree):")
            for i, market in enumerate(leaders, 1):
                print(f"  {i}. Market {market}: out-degree={out_degrees[market]}, in-degree={in_degrees[market]}")
            
            # Causal relationships
            print(f"\nCausal relationships matrix (M_i → M_j):")
            print(f"  Rows = target market (y), Cols = source market (x)")
            print(f"  1 = x Granger-causes y")
            
            # Count significant relationships
            significant_count = sum(1 for test in grade_data['pairwise_tests'].values() 
                                   if test.get('granger_causes', False))
            print(f"\nSignificant causal relationships: {significant_count}/30")
            
            # Show top causal relationships
            print(f"\nTop causal relationships (sorted by F-statistic):")
            sorted_tests = sorted(
                [(k, v) for k, v in grade_data['pairwise_tests'].items() 
                 if v.get('granger_causes', False)],
                key=lambda x: x[1].get('f_statistic', 0),
                reverse=True
            )
            
            if sorted_tests:
                for i, (relationship, test_result) in enumerate(sorted_tests[:10], 1):
                    f_stat = test_result.get('f_statistic', 0)
                    p_val = test_result.get('p_value', 1)
                    print(f"  {i}. {relationship}: F={f_stat:.4f}, p={p_val:.4f}")
            else:
                print("  No significant relationships found for this grade")
            
            print()
        
        # Check output files
        print(f"\nOutput Files Generated:")
        print(f"{'-'*70}")
        
        output_files = {
            'JSON': output_dir / "granger_results.json",
            'CSV (Pairwise)': list(output_dir.glob("granger_pairwise_*.csv")),
            'Excel': output_dir / "granger_results.xlsx",
            'NPZ': output_dir / "granger_matrices.npz",
            'Markdown': output_dir / "granger_results.md"
        }
        
        for format_name, file_path in output_files.items():
            if isinstance(file_path, list):
                for fp in file_path:
                    if fp.exists():
                        size_kb = fp.stat().st_size / 1024
                        print(f"  ✓ {format_name}: {fp.name} ({size_kb:.2f} KB)")
            else:
                if file_path.exists():
                    size_kb = file_path.stat().st_size / 1024
                    print(f"  ✓ {format_name}: {file_path.name} ({size_kb:.2f} KB)")
                else:
                    print(f"  ✗ {format_name}: NOT FOUND")
        
        print(f"\n{'-'*70}")
        print("Output Directory: " + str(output_dir))
        
        print(f"\n{'='*70}")
        print("OUTPUT FILES EXPLANATION:")
        print(f"{'='*70}")
        print("""
1. granger_results.json
   - Complete raw data structure
   - Suitable for programming/automation
   - Contains all test results, matrices, metrics
   
2. granger_pairwise_*.csv (one per grade)
   - Pairwise test results
   - Columns: Relationship, F_Statistic, P_Value, Granger_Causes, etc.
   - Easy to import in Excel/Sheets
   
3. granger_results.xlsx
   - Multiple sheets:
     * Summary: Market leaders & metrics per grade
     * Pairwise_{grade}: All 30 tests per grade
     * Matrix_{grade}: Causal adjacency matrices
   - Professional format for reports
   
4. granger_matrices.npz
   - NumPy binary format
   - Efficient storage for matrices
   - Use: np.load('file.npz') untuk load
   
5. granger_results.md
   - Human-readable Markdown report
   - Formatted tables with results
   - Easy to share/publish
        """)
        
        print(f"\n{'='*70}")
        print("Ready for MODUL 3: Network Inference & Leader Detection")
        print(f"{'='*70}\n")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ FILE NOT FOUND ERROR: {e}")
        print(f"\nPlease check:")
        print(f"  1. Preprocessed data exists: data/processed/preprocessed_pilot_data.csv")
        print(f"  2. You are in the correct directory")
        return False
        
    except ImportError as e:
        print(f"\n✗ IMPORT ERROR: {e}")
        print(f"\nPlease install required packages:")
        print(f"  pip install pandas numpy scipy openpyxl")
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

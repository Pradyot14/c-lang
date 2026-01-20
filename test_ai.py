#!/usr/bin/env python3
"""Quick test script to show AI results for all test cases."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_analyzer import AIDataFlowAnalyzer

def test_all():
    test_cases_dir = os.path.join(os.path.dirname(__file__), 'test_cases')
    include_dir = os.path.join(test_cases_dir, 'include')
    
    # Get all test directories
    test_dirs = []
    for d in sorted(os.listdir(test_cases_dir)):
        full_path = os.path.join(test_cases_dir, d)
        if os.path.isdir(full_path) and d.startswith('apl') and d != 'include':
            test_dirs.append((d, full_path))
    
    print("\n" + "="*60)
    print("  AI ANALYZER TEST RESULTS")
    print("="*60)
    print(f"\n{'Case':<12} {'FileNo Values':<30} {'Confidence':<12}")
    print("-"*60)
    
    for name, path in test_dirs:
        try:
            analyzer = AIDataFlowAnalyzer(include_paths=[include_dir], verbose=False)
            results = analyzer.analyze_project(path)
            
            if results:
                values = [str(r.resolved_value) for r in results if r.resolved_value is not None]
                confs = [f"{r.confidence:.2f}" for r in results if r.resolved_value is not None]
                values_str = ", ".join(values) if values else "None"
                confs_str = ", ".join(confs) if confs else "-"
            else:
                values_str = "No calls found"
                confs_str = "-"
            
            print(f"{name:<12} {values_str:<30} {confs_str:<12}")
            
        except Exception as e:
            print(f"{name:<12} ERROR: {str(e)[:40]}")
    
    print("-"*60)
    print()

if __name__ == "__main__":
    test_all()

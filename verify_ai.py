#!/usr/bin/env python3
"""Comprehensive AI Analyzer Test - ALL test cases"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_analyzer import AIDataFlowAnalyzer

# ALL expected values
EXPECTED = {
    # Original tests
    'apl001': [2001],
    'apl002': [2002],
    'apl003': [2003],
    'apl004': [2003],
    'apl005': [],
    'apl006': [2001],
    'apl008': [4001],
    'apl009': [5001, 5002, 5003],
    'apl010': [6001],
    'apl011': [7002],
    'apl012': [8001, 8002, 8050],
    'apl100': [2003],
    
    # New tests (switch, hex, struct, etc.)
    'apl_new': [9005],
    'apl_new2': [7777],
    
    # Conditional + arithmetic tests
    'apl_cond1': [3007],  # Nested if-else
    'apl_cond2': [4025],  # Chained ternary
    'apl_cond3': [5015],  # Loop accumulation
    'apl_cond4': [6030],  # Function return affects branch
    'apl_cond5': [7111],  # Bitwise flags
    'apl_cond6': [8235],  # Multi-variable conditionals
    'apl_cond7': [9003],  # State machine
    'apl_cond8': [1050],  # Variable reassignment
    
    # Hard tests
    'apl_hard1': [2222],  # Pointer modification
    'apl_hard2': [3333],  # Array-driven operations
    'apl_hard3': [4233],  # Recursive fibonacci
    'apl_hard4': [5055],  # 2D array lookup
    'apl_hard5': [6789],  # Nested struct
}

def test_all():
    test_cases_dir = os.path.join(os.path.dirname(__file__), 'test_cases')
    include_dir = os.path.join(test_cases_dir, 'include')
    
    print("\n" + "="*80)
    print("  COMPREHENSIVE AI ANALYZER TEST - ALL CASES")
    print("="*80)
    print(f"\n{'Case':<12} {'Expected':<24} {'AI Result':<24} {'Status':<8}")
    print("-"*80)
    
    passed = 0
    failed = 0
    failures = []
    
    for name in sorted(EXPECTED.keys()):
        path = os.path.join(test_cases_dir, name)
        if not os.path.isdir(path):
            continue
            
        expected = sorted(EXPECTED[name])
        
        try:
            analyzer = AIDataFlowAnalyzer(include_paths=[include_dir], verbose=False)
            results = analyzer.analyze_project(path)
            
            actual = sorted([r.resolved_value for r in results if r.resolved_value is not None])
            
            expected_str = str(expected) if expected else "[]"
            actual_str = str(actual) if actual else "[]"
            
            if set(expected) == set(actual):
                status = "✅ PASS"
                passed += 1
            else:
                status = "❌ FAIL"
                failed += 1
                failures.append((name, expected, actual))
            
            print(f"{name:<12} {expected_str:<24} {actual_str:<24} {status}")
            
        except Exception as e:
            print(f"{name:<12} {str(expected):<24} {'ERROR':<24} ❌")
            failed += 1
            failures.append((name, expected, f"ERROR: {e}"))
    
    print("-"*80)
    total = passed + failed
    pct = passed * 100 // total if total > 0 else 0
    
    print(f"\n📊 FINAL RESULTS: {passed}/{total} passed ({pct}% accuracy)")
    
    if failures:
        print("\n❌ FAILURES:")
        for name, exp, got in failures:
            print(f"   {name}: expected {exp}, got {got}")
    
    print("="*80 + "\n")
    
    return passed, failed

if __name__ == "__main__":
    test_all()

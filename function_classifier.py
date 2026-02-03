#!/usr/bin/env python3
"""
Function Read/Write Classifier
Analyzes C code and classifies functions as read/write based on name patterns.
Does NOT modify parser.py - works as a standalone tool.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Import parser without modifying it
from parser import CParser, analyze


# ==================== CLASSIFICATION PATTERNS ====================

READ_PATTERNS = [
    'read', 'get', 'fetch', 'load', 'recv', 'receive', 
    'query', 'find', 'search', 'lookup', 'select', 'peek',
    'check', 'is_', 'has_', 'can_', 'exist', 'valid',
    'open',  # open for reading
]

WRITE_PATTERNS = [
    'write', 'set', 'put', 'store', 'save', 'send',
    'update', 'modify', 'change', 'alter', 'edit',
    'insert', 'add', 'append', 'push', 'create', 'new',
    'delete', 'remove', 'clear', 'reset', 'free', 'close',
    'init', 'alloc', 'malloc',
]

MIXED_PATTERNS = [
    'process', 'handle', 'execute', 'run', 'do_',
    'sync', 'flush', 'commit', 'transfer', 'copy', 'move',
]


# ==================== CLASSIFIER ====================

def classify_function(func_name: str) -> Tuple[str, str]:
    """
    Classify a function as read/write/mixed/unknown based on name.
    Returns: (category, matched_pattern)
    """
    name_lower = func_name.lower()
    
    # Check read patterns
    for pattern in READ_PATTERNS:
        if pattern in name_lower:
            return ('READ', pattern)
    
    # Check write patterns
    for pattern in WRITE_PATTERNS:
        if pattern in name_lower:
            return ('WRITE', pattern)
    
    # Check mixed patterns
    for pattern in MIXED_PATTERNS:
        if pattern in name_lower:
            return ('MIXED', pattern)
    
    return ('UNKNOWN', '')


def classify_all_functions(functions: Set[str]) -> Dict[str, List[dict]]:
    """Classify all functions and group by category."""
    results = {
        'READ': [],
        'WRITE': [],
        'MIXED': [],
        'UNKNOWN': [],
    }
    
    for func in sorted(functions):
        category, pattern = classify_function(func)
        results[category].append({
            'name': func,
            'pattern': pattern
        })
    
    return results


# ==================== ANALYSIS ====================

def analyze_project(project_dir: str, include_paths: List[str] = None) -> Dict:
    """Analyze a C project and classify all functions."""
    
    # Use existing parser
    parser = CParser(include_paths=include_paths or [], verbose=False)
    parser.parse_directory(project_dir)
    
    # Get all functions found (all_functions is already a set)
    all_functions = set(parser.all_functions)
    
    # Also include called functions (even if not defined)
    for site in parser.call_sites:
        all_functions.add(site.callee)
    
    # Classify
    classified = classify_all_functions(all_functions)
    
    return {
        'project': project_dir,
        'total_functions': len(all_functions),
        'classified': classified,
        'summary': {
            'read': len(classified['READ']),
            'write': len(classified['WRITE']),
            'mixed': len(classified['MIXED']),
            'unknown': len(classified['UNKNOWN']),
        }
    }


# ==================== OUTPUT ====================

def print_report(result: Dict, show_unknown: bool = False):
    """Print classification report."""
    
    print(f"\n{'='*60}")
    print(f"📊 FUNCTION CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(f"Project: {result['project']}")
    print(f"Total functions: {result['total_functions']}")
    print(f"{'='*60}\n")
    
    # Summary
    s = result['summary']
    print(f"📈 SUMMARY:")
    print(f"   READ:    {s['read']:3d} functions")
    print(f"   WRITE:   {s['write']:3d} functions")
    print(f"   MIXED:   {s['mixed']:3d} functions")
    print(f"   UNKNOWN: {s['unknown']:3d} functions")
    print()
    
    # Details
    classified = result['classified']
    
    if classified['READ']:
        print(f"📖 READ FUNCTIONS ({len(classified['READ'])}):")
        for f in classified['READ']:
            print(f"   • {f['name']:<40} [matched: {f['pattern']}]")
        print()
    
    if classified['WRITE']:
        print(f"✏️  WRITE FUNCTIONS ({len(classified['WRITE'])}):")
        for f in classified['WRITE']:
            print(f"   • {f['name']:<40} [matched: {f['pattern']}]")
        print()
    
    if classified['MIXED']:
        print(f"🔄 MIXED FUNCTIONS ({len(classified['MIXED'])}):")
        for f in classified['MIXED']:
            print(f"   • {f['name']:<40} [matched: {f['pattern']}]")
        print()
    
    if show_unknown and classified['UNKNOWN']:
        print(f"❓ UNKNOWN FUNCTIONS ({len(classified['UNKNOWN'])}):")
        for f in classified['UNKNOWN']:
            print(f"   • {f['name']}")
        print()


def export_json(result: Dict, output_path: str):
    """Export results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"📁 Exported to: {output_path}")


# ==================== MAIN ====================

def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="Classify C functions as read/write")
    ap.add_argument("project_dir", nargs="?", help="Project directory with C files")
    ap.add_argument("-I", "--include", action="append", default=[], help="Include paths")
    ap.add_argument("-o", "--output", help="Export results to JSON file")
    ap.add_argument("-u", "--show-unknown", action="store_true", help="Show unknown functions")
    ap.add_argument("--all", action="store_true", help="Analyze all test cases")
    
    args = ap.parse_args()
    
    script_dir = Path(__file__).parent
    test_dir = script_dir.parent / "test_cases"
    include_dir = test_dir / "include"
    
    includes = args.include.copy()
    if include_dir.exists():
        includes.append(str(include_dir))
    
    if args.all:
        # Analyze all test cases
        import os
        cases = sorted([d for d in os.listdir(test_dir) 
                       if os.path.isdir(test_dir/d) and d.startswith("apl")])
        
        all_functions = set()
        for case in cases:
            parser = CParser(include_paths=includes, verbose=False)
            parser.parse_directory(str(test_dir / case))
            all_functions.update(parser.all_functions)
            for site in parser.call_sites:
                all_functions.add(site.callee)
        
        classified = classify_all_functions(all_functions)
        result = {
            'project': 'ALL TEST CASES',
            'total_functions': len(all_functions),
            'classified': classified,
            'summary': {
                'read': len(classified['READ']),
                'write': len(classified['WRITE']),
                'mixed': len(classified['MIXED']),
                'unknown': len(classified['UNKNOWN']),
            }
        }
        print_report(result, args.show_unknown)
        
        if args.output:
            export_json(result, args.output)
    
    elif args.project_dir:
        result = analyze_project(args.project_dir, includes)
        print_report(result, args.show_unknown)
        
        if args.output:
            export_json(result, args.output)
    
    else:
        ap.print_help()


if __name__ == "__main__":
    main()


#uv run python3 src/function_classifier.py --all
#uv run python3 src/function_classifier.py test_cases/apl100
# Export to JSON
#uv run python3 src/function_classifier.py test_cases/apl100 -o output.json
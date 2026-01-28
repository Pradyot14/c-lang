#!/usr/bin/env python3
"""
Diagnostic script to compare v17 parsing with HTML output.
Run this on your real data to find where they diverge.

Usage:
    python diagnose.py <source_dir> <include_dir> [entry_func] [target_func]
    
Example:
    python diagnose.py ./src ./include main mpf_mfs_open
"""

import sys
import os
import json

# Import v17
from agent_v17 import CParser, PathFinder

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    src_dir = sys.argv[1]
    inc_dir = sys.argv[2]
    entry = sys.argv[3] if len(sys.argv) > 3 else "main"
    target = sys.argv[4] if len(sys.argv) > 4 else "mpf_mfs_open"
    
    print("=" * 70)
    print("v17 DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Source: {src_dir}")
    print(f"Include: {inc_dir}")
    print(f"Entry: {entry}")
    print(f"Target: {target}")
    print()
    
    # Parse
    parser = CParser(verbose=False)
    
    print("[1] FILES LOADED")
    print("-" * 70)
    if os.path.isdir(src_dir):
        parser.load_directory(src_dir)
    else:
        parser.parse_file(src_dir)
    
    if os.path.isdir(inc_dir):
        parser.load_directory(inc_dir)
    else:
        parser.parse_file(inc_dir)
    
    for fname in sorted(parser.files.keys()):
        size = len(parser.files[fname])
        print(f"  {fname}: {size} bytes")
    print(f"  Total: {len(parser.files)} files")
    print()
    
    # Build graph
    parser.build_graph()
    
    print("[2] FUNCTIONS FOUND")
    print("-" * 70)
    for name in sorted(parser.functions.keys()):
        func = parser.functions[name]
        calls = list(func.calls.keys())[:5]
        more = f"... +{len(func.calls)-5}" if len(func.calls) > 5 else ""
        print(f"  {name} ({func.file}): calls {calls}{more}")
    print(f"  Total: {len(parser.functions)} functions")
    print()
    
    print("[3] MACROS WITH FUNCTION CALLS")
    print("-" * 70)
    for name, macro in sorted(parser.macros.items()):
        if macro.calls:
            print(f"  {name}: {macro.calls}")
    print()
    
    print("[4] FUNCTION POINTERS")
    print("-" * 70)
    for name, targets in sorted(parser.func_pointers.items()):
        print(f"  {name}: {list(targets)}")
    if not parser.func_pointers:
        print("  (none found)")
    print()
    
    print(f"[5] ENTRY FUNCTION: {entry}")
    print("-" * 70)
    if entry in parser.functions:
        func = parser.functions[entry]
        print(f"  File: {func.file}")
        print(f"  Calls: {list(func.calls.keys())}")
        print(f"  In call graph: {entry in parser.call_graph}")
    else:
        print(f"  *** NOT FOUND ***")
        print(f"  Available functions: {list(parser.functions.keys())[:20]}")
    print()
    
    print(f"[6] TARGET FUNCTION: {target}")
    print("-" * 70)
    if target in parser.functions:
        func = parser.functions[target]
        print(f"  File: {func.file}")
        print(f"  In call graph: {target in parser.call_graph}")
    else:
        print(f"  *** NOT FOUND in functions ***")
        # Check if it's called by anyone
        callers = []
        for fname, calls in parser.call_graph.items():
            if target in calls:
                callers.append(fname)
        if callers:
            print(f"  But called by: {callers}")
        else:
            print(f"  And not called by anyone!")
    print()
    
    print("[7] PATH FINDING")
    print("-" * 70)
    finder = PathFinder(parser)
    paths = finder.find_paths(entry, target)
    
    if paths:
        print(f"  Found {len(paths)} paths:")
        for i, path in enumerate(paths[:10], 1):
            path_str = " -> ".join(s.name for s in path)
            print(f"    {i}. {path_str}")
        if len(paths) > 10:
            print(f"    ... and {len(paths)-10} more")
    else:
        print("  *** NO PATHS FOUND ***")
        print()
        print("  Analyzing why...")
        
        # Try to find partial path
        visited = set()
        queue = [(entry, [entry])]
        max_depth = 10
        closest = []
        
        while queue and len(closest) < 5:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                continue
            if current in visited:
                continue
            visited.add(current)
            
            calls = parser.call_graph.get(current, {})
            for callee in calls:
                if callee == target:
                    closest.append(path + [callee])
                elif callee in parser.call_graph:
                    queue.append((callee, path + [callee]))
        
        if closest:
            print(f"  Partial paths found:")
            for p in closest:
                print(f"    {' -> '.join(p)}")
        else:
            print(f"  Can't reach {target} from {entry} within {max_depth} hops")
            print()
            # Show what entry can reach
            reachable = list(visited)[:20]
            print(f"  Reachable from {entry}: {reachable}")
    
    print()
    print("=" * 70)
    print("EXPORT DATA FOR HTML COMPARISON")
    print("=" * 70)
    
    # Export data that can be compared with HTML output
    export = {
        "files": list(parser.files.keys()),
        "functions": list(parser.functions.keys()),
        "call_graph": {k: list(v.keys()) for k, v in parser.call_graph.items()},
        "macros_with_calls": {k: v.calls for k, v in parser.macros.items() if v.calls},
        "func_pointers": {k: list(v) for k, v in parser.func_pointers.items()},
    }
    
    with open("diagnose_export.json", "w") as f:
        json.dump(export, f, indent=2)
    print("Exported to diagnose_export.json")
    print()
    print("Run the HTML tool on the same files and compare:")
    print("  1. Number of functions")
    print("  2. Call graph edges")
    print("  3. Which functions call the target")

if __name__ == "__main__":
    main()

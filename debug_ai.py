#!/usr/bin/env python3
"""
Debug script for troubleshooting AI analyzer issues on real data.
Run this to see detailed error information.

Usage:
    python3 debug_ai.py /path/to/your/project [include_path1] [include_path2]
"""
import os
import sys
import traceback
from pathlib import Path

# Load .env
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_analyzer import AIDataFlowAnalyzer

def debug_analyze(project_path: str, include_paths: list = None):
    """Run analysis with full debug output."""
    
    print("=" * 70)
    print("  AI ANALYZER DEBUG MODE")
    print("=" * 70)
    
    # Check API key
    api_key = os.environ.get('OPENAI_API_KEY', '')
    if not api_key:
        print("\n❌ ERROR: OPENAI_API_KEY not set in .env file!")
        return
    print(f"\n✅ API Key: Set ({len(api_key)} chars)")
    
    # Check project path
    if not os.path.isdir(project_path):
        print(f"\n❌ ERROR: Project path not found: {project_path}")
        return
    print(f"✅ Project: {project_path}")
    
    # Check include paths
    include_paths = include_paths or []
    for ip in include_paths:
        if os.path.isdir(ip):
            print(f"✅ Include: {ip}")
        else:
            print(f"⚠️ Include not found: {ip}")
    
    print("\n" + "-" * 70)
    print("LOADING PROJECT...")
    print("-" * 70)
    
    try:
        # Create analyzer with debug mode
        analyzer = AIDataFlowAnalyzer(
            include_paths=include_paths, 
            verbose=True, 
            debug=True
        )
        
        # Load project manually to see stats
        analyzer._load_all_code(project_path)
        
        print(f"\n📁 Files loaded: {len(analyzer.all_code)}")
        for fp in sorted(analyzer.all_code.keys()):
            size = len(analyzer.all_code[fp])
            print(f"   - {os.path.basename(fp)}: {size:,} chars")
        
        print(f"\n🔤 Macros found: {len(analyzer.all_macros)}")
        for name, (value, source) in sorted(analyzer.all_macros.items())[:20]:
            print(f"   - {name} = {value[:50]}{'...' if len(value) > 50 else ''} (from {source})")
        if len(analyzer.all_macros) > 20:
            print(f"   ... and {len(analyzer.all_macros) - 20} more")
        
        print(f"\n⚙️ Functions found: {len(analyzer.functions)}")
        for name in sorted(analyzer.functions.keys())[:15]:
            print(f"   - {name}")
        if len(analyzer.functions) > 15:
            print(f"   ... and {len(analyzer.functions) - 15} more")
        
        # Find calls
        calls = analyzer._find_all_calls()
        print(f"\n📞 mpf_mfs_open calls found: {len(calls)}")
        
        for i, call in enumerate(calls):
            print(f"\n{'=' * 70}")
            print(f"ANALYZING CALL {i+1}/{len(calls)}")
            print(f"{'=' * 70}")
            print(f"File: {os.path.basename(call['file'])}")
            print(f"Line: {call['line']}")
            print(f"Function: {call['func']}")
            print(f"Call: {call['full_call']}")
            print(f"3rd Arg (raw): {call['raw_arg']}")
            
            # Get relevant functions
            relevant_funcs = analyzer._get_relevant_functions(call)
            print(f"\nRelevant functions in call chain: {list(relevant_funcs.keys())}")
            
            # Get relevant macros
            relevant_macros = analyzer._get_relevant_macros(call, relevant_funcs)
            print(f"Relevant macros: {list(relevant_macros.keys())}")
            
            print("\n" + "-" * 40)
            print("Running AI analysis...")
            print("-" * 40)
            
            try:
                result = analyzer._analyze_call(call)
                
                print(f"\n📊 RESULT:")
                print(f"   Resolved Value: {result.resolved_value}")
                print(f"   Method: {result.analysis_method}")
                print(f"   Confidence: {result.confidence}")
                print(f"   Error: {result.error}")
                print(f"   Chain: {result.resolution_chain}")
                
                if result.reasoning_trace:
                    print(f"\n📝 AI REASONING TRACE:")
                    print("-" * 40)
                    # Print first 1500 chars of trace
                    trace = result.reasoning_trace[:1500]
                    print(trace)
                    if len(result.reasoning_trace) > 1500:
                        print(f"\n... (trace truncated, {len(result.reasoning_trace)} total chars)")
                
            except Exception as e:
                print(f"\n❌ ANALYSIS ERROR:")
                print(traceback.format_exc())
        
        print("\n" + "=" * 70)
        print("DEBUG COMPLETE")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR:")
        print(traceback.format_exc())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample:")
        print("  python3 debug_ai.py test_cases/apl006 test_cases/include")
        sys.exit(1)
    
    project = sys.argv[1]
    includes = sys.argv[2:] if len(sys.argv) > 2 else []
    
    debug_analyze(project, includes)

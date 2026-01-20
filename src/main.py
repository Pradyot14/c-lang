#!/usr/bin/env python3
"""
FileNo Extractor - Main Entry Point
Extracts file numbers from mpf_mfs_open() calls in C projects.

Usage:
    python main.py <project_directory>
    python main.py --all                      # Run on all test cases
    python main.py --all --method ai          # Use AI analysis
    python main.py --tree <case_name>         # Show data flow tree
    python main.py --method pattern           # Pattern-based (default)
    python main.py --method ai                # AI-based (Groq LLM)
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TEST_CASES_DIR, INCLUDE_DIR
from macro_extractor import MacroExtractor
from ast_parser import ASTParser


def get_analyzer(method: str = "pattern", include_paths: List[str] = None):
    """Get the appropriate analyzer based on method."""
    if include_paths is None:
        include_paths = [INCLUDE_DIR]
    
    if method == "ai":
        try:
            from ai_analyzer import AIDataFlowAnalyzer
            return AIDataFlowAnalyzer(include_paths=include_paths)
        except ImportError as e:
            print(f"⚠️  AI analyzer not available: {e}")
            print("   Falling back to pattern-based analyzer...")
            from data_flow_analyzer import DataFlowAnalyzer
            return DataFlowAnalyzer(include_paths=include_paths)
    else:
        from data_flow_analyzer import DataFlowAnalyzer
        return DataFlowAnalyzer(include_paths=include_paths)


def print_banner(method: str = "pattern"):
    """Print application banner."""
    method_str = "🤖 AI-Powered (Azure OpenAI)" if method == "ai" else "🔍 Pattern-Based"
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    FileNo Extractor v2.0                      ║
║         Extract file numbers from mpf_mfs_open() calls        ║
║                                                               ║
║  Method: {method_str:<43} ║
╚═══════════════════════════════════════════════════════════════╝
    """)


def analyze_single_project(project_dir: str, include_paths: List[str] = None, 
                           return_raw: bool = False, method: str = "pattern"):
    """Analyze a single project directory."""
    if include_paths is None:
        include_paths = [INCLUDE_DIR]
    
    # Add project directory's own include path
    include_paths = include_paths + [project_dir]
    
    analyzer = get_analyzer(method, include_paths)
    results = analyzer.analyze_project(project_dir)
    
    if return_raw:
        return results
    
    output = []
    for r in results:
        output.append({
            "file": os.path.basename(r.file_path),
            "file_path": r.file_path,
            "line": r.line_number,
            "function": r.containing_function,
            "raw_argument": r.raw_argument,
            "resolved_fileno": r.resolved_value,
            "confidence": r.confidence,
            "resolution_chain": r.resolution_chain,
            "method": getattr(r, 'analysis_method', method),
        })
    
    return output


def analyze_all_test_cases(return_raw: bool = False, method: str = "pattern") -> Dict:
    """Analyze all test cases."""
    all_results = {}
    
    test_cases = sorted([
        d for d in os.listdir(TEST_CASES_DIR)
        if os.path.isdir(os.path.join(TEST_CASES_DIR, d)) and d.startswith("apl")
    ])
    
    for case in test_cases:
        case_dir = os.path.join(TEST_CASES_DIR, case)
        print(f"\n{'='*60}")
        print(f"Analyzing: {case}")
        print('='*60)
        
        results = analyze_single_project(case_dir, include_paths=[INCLUDE_DIR], return_raw=return_raw, method=method)
        all_results[case] = results
        
        if results:
            for r in results:
                if return_raw:
                    print(f"  📄 File: {os.path.basename(r.file_path)}")
                    print(f"  📍 Line: {r.line_number}")
                    print(f"  🔍 Raw argument: {r.raw_argument}")
                    print(f"  ✅ Resolved fileno: {r.resolved_value}")
                    print(f"  📊 Confidence: {r.confidence:.2f}")
                    chain = r.resolution_chain[:3] if r.resolution_chain else []
                    print(f"  🔗 Chain: {' -> '.join(chain)}")
                else:
                    print(f"  📄 File: {r['file']}")
                    print(f"  📍 Line: {r['line']}")
                    print(f"  🔍 Raw argument: {r['raw_argument']}")
                    print(f"  ✅ Resolved fileno: {r['resolved_fileno']}")
                    print(f"  📊 Confidence: {r['confidence']:.2f}")
                    chain = r['resolution_chain'][:3] if r['resolution_chain'] else []
                    print(f"  🔗 Chain: {' -> '.join(chain)}")
        else:
            print("  ⚠️  No mpf_mfs_open() calls found")
    
    return all_results


def export_results(results: Dict[str, List[Dict]], output_file: str):
    """Export results to JSON file."""
    # Convert raw results to dicts if needed
    export_data = {}
    for case, case_results in results.items():
        if case_results and hasattr(case_results[0], 'resolved_value'):
            export_data[case] = [{
                "file": os.path.basename(r.file_path),
                "line": r.line_number,
                "raw_argument": r.raw_argument,
                "resolved_fileno": r.resolved_value,
                "confidence": r.confidence,
                "resolution_chain": r.resolution_chain,
                "method": getattr(r, 'analysis_method', 'unknown'),
            } for r in case_results]
        else:
            export_data[case] = case_results
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    print(f"\n📁 Results exported to: {output_file}")


def show_tree(case_name: str, output_format: str = "ascii", output_file: str = None, method: str = "pattern"):
    """Show data flow tree for a specific case."""
    from tree_visualizer import generate_tree_for_result, DataFlowTreeBuilder, TreeVisualizer
    
    # Find the case directory
    case_dir = os.path.join(TEST_CASES_DIR, case_name)
    if not os.path.isdir(case_dir):
        # Try as a full path
        if os.path.isdir(case_name):
            case_dir = case_name
            case_name = os.path.basename(case_name)
        else:
            print(f"❌ Error: Case '{case_name}' not found")
            print(f"\nAvailable test cases:")
            for d in sorted(os.listdir(TEST_CASES_DIR)):
                if os.path.isdir(os.path.join(TEST_CASES_DIR, d)) and d.startswith("apl"):
                    print(f"  • {d}")
            return
    
    print(f"\n🔍 Analyzing: {case_name}")
    print("="*60)
    
    # Analyze the case
    results = analyze_single_project(case_dir, return_raw=True, method=method)
    
    if not results:
        print("⚠️  No mpf_mfs_open() calls found in this case")
        return
    
    # For HTML format, auto-save to output directory if no output file specified
    if output_format == "html" and output_file is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{case_name}_tree")
    
    # Generate tree for each result
    saved_files = []
    for i, result in enumerate(results):
        print(f"\n📊 Data Flow Tree #{i+1}")
        print(f"   File: {os.path.basename(result.file_path)}:{result.line_number}")
        print(f"   Raw Argument: {result.raw_argument}")
        print(f"   Resolved: {result.resolved_value}")
        print("-"*60)
        
        tree_output = generate_tree_for_result(result, output_format)
        
        if output_file:
            # Determine file extension based on format
            ext_map = {
                "ascii": ".txt",
                "mermaid": ".md",
                "graphviz": ".dot",
                "html": ".html"
            }
            ext = ext_map.get(output_format, ".txt")
            
            # Create filename
            if len(results) > 1:
                file_name = f"{output_file}_{i+1}{ext}"
            else:
                file_name = f"{output_file}{ext}"
            
            # Create directory if needed
            os.makedirs(os.path.dirname(file_name) if os.path.dirname(file_name) else ".", exist_ok=True)
            
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(tree_output)
            print(f"✅ Tree saved to: {file_name}")
            saved_files.append(file_name)
        else:
            print(tree_output)
    
    # Open HTML files in browser
    if output_format == "html" and saved_files:
        import webbrowser
        for file_path in saved_files:
            abs_path = os.path.abspath(file_path)
            url = f"file://{abs_path}"
            print(f"🌐 Opening in browser: {url}")
            webbrowser.open(url)
    
    print("\n" + "="*60)


def show_all_trees(output_format: str = "ascii", output_dir: str = None, method: str = "pattern"):
    """Show data flow trees for all test cases."""
    from tree_visualizer import generate_tree_for_result
    
    test_cases = sorted([
        d for d in os.listdir(TEST_CASES_DIR)
        if os.path.isdir(os.path.join(TEST_CASES_DIR, d)) and d.startswith("apl")
    ])
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for case in test_cases:
        case_dir = os.path.join(TEST_CASES_DIR, case)
        
        print(f"\n{'='*60}")
        print(f"🌳 Tree for: {case}")
        print('='*60)
        
        results = analyze_single_project(case_dir, return_raw=True, method=method)
        
        if not results:
            print("  ⚠️  No mpf_mfs_open() calls found")
            continue
        
        for i, result in enumerate(results):
            tree_output = generate_tree_for_result(result, output_format)
            
            if output_dir:
                ext_map = {"ascii": ".txt", "mermaid": ".md", "graphviz": ".dot", "html": ".html"}
                ext = ext_map.get(output_format, ".txt")
                file_name = os.path.join(output_dir, f"{case}_tree{ext}")
                
                with open(file_name, 'w', encoding='utf-8') as f:
                    f.write(tree_output)
                print(f"  ✅ Saved: {file_name}")
            else:
                print(tree_output)


def compare_methods():
    """Compare pattern-based vs AI-based analysis on all test cases."""
    print("\n" + "="*70)
    print("  Comparing Analysis Methods: Pattern vs AI")
    print("="*70)
    
    test_cases = sorted([
        d for d in os.listdir(TEST_CASES_DIR)
        if os.path.isdir(os.path.join(TEST_CASES_DIR, d))
    ])
    
    print(f"\n{'Case':<15} {'Pattern Results':<25} {'AI Results':<25} {'Match':<10}")
    print("-"*75)
    
    matches = 0
    total = 0
    
    for case in test_cases:
        case_dir = os.path.join(TEST_CASES_DIR, case)
        
        # Pattern analysis
        pattern_results = analyze_single_project(case_dir, return_raw=True, method="pattern")
        pattern_vals = sorted([r.resolved_value for r in pattern_results if r.resolved_value])
        
        # AI analysis
        try:
            ai_results = analyze_single_project(case_dir, return_raw=True, method="ai")
            ai_vals = sorted([r.resolved_value for r in ai_results if r.resolved_value])
        except Exception as e:
            ai_vals = [f"Error: {str(e)[:20]}"]
        
        # Check if both methods agree
        is_match = pattern_vals == ai_vals
        if is_match:
            matches += 1
        total += 1
        
        match_status = "✅ Yes" if is_match else "❌ No"
        
        print(f"{case:<15} {str(pattern_vals):<25} {str(ai_vals):<25} {match_status:<10}")
    
    print("-"*75)
    print(f"\n📊 Agreement: {matches}/{total} cases ({100*matches/total:.1f}%)")
    print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract file numbers from mpf_mfs_open() calls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py /path/to/project              # Analyze with pattern method
  python main.py --all                          # Run on all test cases
  python main.py --all --method ai              # Use AI (Azure OpenAI) method
  python main.py --tree <case_name>             # Show tree for a case
  python main.py --tree <case_name> --format html  # Generate HTML tree
  python main.py --compare                      # Compare both methods

Analysis Methods:
  pattern   Pattern-based analysis (default) - Fast, offline, no API calls
  ai        AI-based analysis (Azure OpenAI GPT-4) - Uses LLM for complex cases

Tree Formats:
  ascii     Terminal-friendly tree (default)
  mermaid   Mermaid diagram (for markdown)
  graphviz  DOT format (for generating images)
  html      Interactive HTML visualization
        """
    )
    parser.add_argument(
        "project_dir",
        nargs="?",
        help="Path to project directory to analyze"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run on all test cases"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for results"
    )
    parser.add_argument(
        "--include", "-I",
        action="append",
        default=[],
        help="Additional include paths"
    )
    
    # Method selection
    parser.add_argument(
        "--method", "-m",
        choices=["pattern", "ai"],
        default="pattern",
        help="Analysis method: pattern (default) or ai (Azure OpenAI GPT-4)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare pattern-based vs AI-based analysis"
    )
    
    # Tree visualization options
    parser.add_argument(
        "--tree", "-t",
        metavar="CASE",
        help="Show data flow tree for a specific case"
    )
    parser.add_argument(
        "--tree-all",
        action="store_true",
        help="Show data flow trees for all test cases"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["ascii", "mermaid", "graphviz", "html"],
        default="ascii",
        help="Output format for tree visualization (default: ascii)"
    )
    
    # Data flow visualization
    parser.add_argument(
        "--visualize", "-v",
        action="store_true",
        help="Show detailed data flow visualization in terminal"
    )
    
    args = parser.parse_args()
    
    print_banner(args.method)
    
    # Compare mode
    if args.compare:
        compare_methods()
        return
    
    # Tree visualization mode
    if args.tree:
        show_tree(args.tree, args.format, args.output, args.method)
        return
    
    if args.tree_all:
        show_all_trees(args.format, args.output, args.method)
        return
    
    # Data flow visualization mode
    if args.visualize and (args.project_dir or args.all):
        # Import visualizer from parent directory
        visualize_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'visualize_flow.py')
        if os.path.exists(visualize_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("visualize_flow", visualize_path)
            visualize_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(visualize_module)
            
            include_paths = [INCLUDE_DIR] + args.include
            
            if args.all:
                test_cases = sorted([
                    d for d in os.listdir(TEST_CASES_DIR)
                    if os.path.isdir(os.path.join(TEST_CASES_DIR, d)) and d.startswith("apl")
                ])
                for case in test_cases:
                    case_dir = os.path.join(TEST_CASES_DIR, case)
                    visualize_module.analyze_with_visualization(case_dir, include_paths, detailed=True)
            else:
                visualize_module.analyze_with_visualization(args.project_dir, include_paths, detailed=True)
            return
        else:
            print("⚠️  Visualizer not found, falling back to standard output")
    
    # Analysis mode
    if args.all:
        results = analyze_all_test_cases(return_raw=True, method=args.method)
        
        if args.output:
            export_results(results, args.output)
        
    elif args.project_dir:
        if not os.path.isdir(args.project_dir):
            print(f"Error: {args.project_dir} is not a valid directory")
            sys.exit(1)
        
        include_paths = [INCLUDE_DIR] + args.include
        results = analyze_single_project(args.project_dir, include_paths, method=args.method)
        
        print(f"\nResults for: {args.project_dir}")
        print("-" * 40)
        
        if results:
            for r in results:
                print(f"  File: {r['file']}")
                print(f"  Line: {r['line']}")
                print(f"  Raw: {r['raw_argument']}")
                print(f"  Resolved: {r['resolved_fileno']}")
                print(f"  Confidence: {r['confidence']:.2f}")
                print(f"  Method: {r.get('method', args.method)}")
                print()
        else:
            print("  No mpf_mfs_open() calls found")
        
        if args.output:
            export_results({os.path.basename(args.project_dir): results}, args.output)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

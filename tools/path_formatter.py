#!/usr/bin/env python3
"""
Path Formatter - Extracts and formats path functions in execution order.
Supports multiple paths - generates separate files for each path.
Now with static array lookup support and DATA FLOW TRACKING!

v2: Added data flow analysis to include helper functions that compute
    values passed to the target function.
"""

import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple

try:
    from .parser import CParser, analyze
except ImportError:
    from parser import CParser, analyze


def get_function_body(parser: CParser, func_name: str, files: Dict[str, str]) -> Optional[str]:
    """Extract function body from source files."""
    if func_name not in parser.functions:
        return None
    
    func_info = parser.functions[func_name]
    file_path = func_info.file_path
    
    if file_path not in files:
        try:
            files[file_path] = Path(file_path).read_text(errors='ignore')
        except:
            return None
    
    content = files[file_path]
    lines = content.split('\n')
    start_line = func_info.line - 1
    
    brace_line = start_line
    while brace_line < len(lines) and '{' not in lines[brace_line]:
        brace_line += 1
    
    if brace_line >= len(lines):
        return None
    
    depth = 0
    end_line = brace_line
    for i in range(brace_line, len(lines)):
        depth += lines[i].count('{') - lines[i].count('}')
        if depth == 0:
            end_line = i
            break
    
    return '\n'.join(lines[start_line:end_line + 1])


def resolve_macro(parser: CParser, name: str, visited: Set[str] = None, depth: int = 0) -> Optional[int]:
    """Recursively resolve macro to integer value."""
    if visited is None:
        visited = set()
    
    if depth > 20 or name in visited:
        return None
    
    visited.add(name)
    
    if name not in parser.macros:
        return None
    
    val = parser.macros[name].value.strip()
    
    if re.match(r'^-?\d+$', val):
        return int(val)
    if re.match(r'^0[xX][0-9a-fA-F]+$', val):
        return int(val, 16)
    
    paren_match = re.match(r'^\(\s*(-?\d+)\s*\)$', val)
    if paren_match:
        return int(paren_match.group(1))
    
    paren_hex = re.match(r'^\(\s*(0[xX][0-9a-fA-F]+)\s*\)$', val)
    if paren_hex:
        return int(paren_hex.group(1), 16)
    
    if re.match(r'^[A-Z_][A-Z0-9_]*$', val):
        return resolve_macro(parser, val, visited, depth + 1)
    
    expr = val
    all_resolved = True
    
    for macro_name in parser.macros:
        if re.search(rf'\b{macro_name}\b', expr):
            resolved = resolve_macro(parser, macro_name, visited.copy(), depth + 1)
            if resolved is not None:
                expr = re.sub(rf'\b{macro_name}\b', str(resolved), expr)
            else:
                all_resolved = False
    
    if all_resolved:
        clean = re.sub(r'\(int\)|\(long\)|\(unsigned\)', '', expr)
        clean = re.sub(r'\bsizeof\s*\([^)]+\)', '4', clean)
        
        if re.match(r'^[\d\s\+\-\*\/\(\)\&\|\^\~\<\>\%]+$', clean):
            try:
                return int(eval(clean))
            except:
                pass
    
    return None


def find_macros_in_code(code: str, all_macros: Set[str]) -> Set[str]:
    """Find macro names used in code."""
    used = set()
    for macro in all_macros:
        if re.search(rf'\b{macro}\b', code):
            used.add(macro)
    return used


def find_array_accesses(code: str) -> List[Tuple[str, str]]:
    """
    Find array accesses in code like: array_name[INDEX] or array_name[MACRO]
    Returns list of (array_name, index_expression)
    """
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\[\s*([a-zA-Z0-9_]+)\s*\]'
    matches = re.findall(pattern, code)
    
    skip_names = {'sizeof', 'typeof', 'alignof'}
    return [(name, idx) for name, idx in matches if name not in skip_names]


def find_array_definition(array_name: str, files: Dict[str, str]) -> Optional[Dict]:
    """
    Find static/const array definition and extract its values.
    Returns dict with 'values' list and 'raw' definition string.
    """
    for file_path, content in files.items():
        patterns = [
            rf'(?:static\s+)?(?:const\s+)?(?:int|long|unsigned|short|uint32_t)\s+{array_name}\s*\[\s*\]\s*=\s*\{{([^}}]+)\}}',
            rf'(?:static\s+)?(?:const\s+)?(?:int|long|unsigned|short|uint32_t)\s+{array_name}\s*\[\s*\d*\s*\]\s*=\s*\{{([^}}]+)\}}',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
            if match:
                values_str = match.group(1)
                values = []
                for val in values_str.split(','):
                    val = val.strip()
                    val = re.sub(r'/\*.*?\*/', '', val).strip()
                    val = re.sub(r'//.*$', '', val, flags=re.MULTILINE).strip()
                    if not val:
                        continue
                    try:
                        if val.startswith('0x') or val.startswith('0X'):
                            values.append(int(val, 16))
                        elif val.lstrip('-').isdigit():
                            values.append(int(val))
                        else:
                            values.append(val)
                    except:
                        values.append(val)
                
                return {
                    'values': values,
                    'raw': match.group(0),
                    'file': file_path
                }
    
    return None


def resolve_array_access(
    array_name: str, 
    index_expr: str, 
    parser: CParser, 
    files: Dict[str, str]
) -> Optional[Dict]:
    """
    Resolve array[INDEX] to its actual value.
    Returns dict with resolved info or None if can't resolve.
    """
    index = None
    if index_expr.isdigit():
        index = int(index_expr)
    else:
        index = resolve_macro(parser, index_expr)
    
    if index is None:
        return None
    
    array_def = find_array_definition(array_name, files)
    if not array_def:
        return None
    
    values = array_def['values']
    if index < 0 or index >= len(values):
        return None
    
    value = values[index]
    
    if isinstance(value, str):
        resolved = resolve_macro(parser, value)
        if resolved is not None:
            value = resolved
    
    return {
        'array_name': array_name,
        'index_expr': index_expr,
        'index': index,
        'value': value,
        'all_values': values
    }


# ============================================================================
# NEW: Data Flow Analysis - Find helper functions that compute values
# ============================================================================

def find_target_call_args(code: str, target_function: str) -> Dict[str, List[str]]:
    """
    Find the arguments passed to the target function.
    Returns dict with 'raw_args' list and 'variables' (identifiers used).
    """
    # Pattern: target_function(arg1, arg2, arg3, ...)
    pattern = rf'{target_function}\s*\(\s*([^)]+)\s*\)'
    match = re.search(pattern, code)
    
    if not match:
        return {'raw_args': [], 'variables': set()}
    
    args_str = match.group(1)
    
    # Split by comma, but respect nested parentheses
    args = []
    depth = 0
    current = ""
    for ch in args_str:
        if ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        args.append(current.strip())
    
    # Extract variable names from args
    variables = set()
    for arg in args:
        # Find identifiers (not numbers, not strings, not &addr)
        identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', arg)
        for ident in identifiers:
            # Filter out keywords and common non-variables
            if ident not in {'NULL', 'sizeof', 'true', 'false', 'void', 'int', 'char', 'const'}:
                variables.add(ident)
    
    return {'raw_args': args, 'variables': variables}


def find_variable_assignments(code: str, variables: Set[str]) -> Dict[str, List[str]]:
    """
    Find how variables are assigned in the code.
    Returns dict: variable -> list of function calls that assign to it
    """
    assignments = {}
    
    for var in variables:
        # Pattern: var = function_call(...)
        # Captures: function_name
        pattern = rf'\b{var}\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, code)
        
        if matches:
            # Filter out common non-functions
            funcs = [m for m in matches if m not in {'sizeof', 'typeof', 'offsetof', 'int', 'long', 'char'}]
            if funcs:
                assignments[var] = funcs
    
    return assignments


def find_data_flow_functions(
    path_functions_code: str,
    target_function: str,
    parser: CParser,
    files: Dict[str, str],
    max_depth: int = 5
) -> List[str]:
    """
    Find helper functions that compute values passed to the target function.
    
    This performs a data flow analysis:
    1. Find variables used in target_function call
    2. Find functions that assign to those variables
    3. Recursively check those functions for more helpers
    
    Returns list of function names to include (in order).
    """
    helper_functions = []
    visited = set()
    
    def trace_helpers(code: str, depth: int):
        if depth > max_depth:
            return
        
        # Find target call arguments
        target_info = find_target_call_args(code, target_function)
        variables = target_info['variables']
        
        if not variables:
            return
        
        # Find function calls that assign to these variables
        assignments = find_variable_assignments(code, variables)
        
        for var, funcs in assignments.items():
            for func_name in funcs:
                if func_name in visited:
                    continue
                if func_name == target_function:
                    continue
                
                visited.add(func_name)
                
                # Check if this function exists in the parser
                if func_name in parser.functions:
                    helper_functions.append(func_name)
                    
                    # Get this function's body and recursively check
                    helper_body = get_function_body(parser, func_name, files)
                    if helper_body:
                        # Look for more helpers in this function
                        # Check if this function calls other functions whose return value it uses
                        trace_nested_helpers(helper_body, depth + 1)
    
    def trace_nested_helpers(code: str, depth: int):
        """Find functions called within a helper that compute return values."""
        if depth > max_depth:
            return
        
        # Find patterns like: result = helper_func(...) or return helper_func(...)
        # Pattern 1: variable = func(...)
        pattern1 = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)'
        # Pattern 2: return func(...)
        pattern2 = r'\breturn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        
        for match in re.finditer(pattern1, code):
            var_name = match.group(1)
            func_name = match.group(2)
            
            # Check if var_name is used in return or further computation
            if func_name in visited:
                continue
            if func_name in {'sizeof', 'typeof', 'offsetof', 'int', 'long', 'malloc', 'calloc', 'free'}:
                continue
            if func_name == target_function:
                continue
            
            # Check if function exists
            if func_name in parser.functions:
                visited.add(func_name)
                helper_functions.append(func_name)
                
                # Recurse
                helper_body = get_function_body(parser, func_name, files)
                if helper_body:
                    trace_nested_helpers(helper_body, depth + 1)
        
        for match in re.finditer(pattern2, code):
            func_name = match.group(1)
            if func_name in visited:
                continue
            if func_name in {'sizeof', 'typeof', 'offsetof', 'int', 'long', 'malloc', 'calloc', 'free'}:
                continue
            if func_name == target_function:
                continue
            
            if func_name in parser.functions:
                visited.add(func_name)
                helper_functions.append(func_name)
                
                helper_body = get_function_body(parser, func_name, files)
                if helper_body:
                    trace_nested_helpers(helper_body, depth + 1)
    
    # Start tracing from the path functions
    trace_helpers(path_functions_code, 0)
    
    return helper_functions


# ============================================================================
# Modified format_single_path with data flow tracking
# ============================================================================

def format_single_path(
    parser: CParser,
    path: List[str],
    files: Dict[str, str],
    target_function: str,
    path_index: int = 1,
    total_paths: int = 1
) -> str:
    """Format a single path to string, including helper functions for data flow."""
    
    # Collect function bodies for path functions
    func_bodies = []
    for func_name in path[:-1]:
        body = get_function_body(parser, func_name, files)
        if body:
            func_bodies.append(body)
    
    path_code = '\n'.join(func_bodies)
    
    # ========== NEW: Find data flow helper functions ==========
    helper_funcs = find_data_flow_functions(path_code, target_function, parser, files)
    
    # Get helper function bodies
    helper_bodies = {}
    for func_name in helper_funcs:
        body = get_function_body(parser, func_name, files)
        if body:
            helper_bodies[func_name] = body
    
    # Combine all code for macro/array analysis
    all_code = path_code + '\n' + '\n'.join(helper_bodies.values())
    
    # Find macros used
    all_macro_names = set(parser.macros.keys())
    used_macros = find_macros_in_code(all_code, all_macro_names)
    
    # Get macro dependencies
    def get_macro_deps(macro_name: str, visited: Set[str]) -> Set[str]:
        if macro_name in visited:
            return set()
        visited.add(macro_name)
        deps = set()
        if macro_name in parser.macros:
            val = parser.macros[macro_name].value
            for other in all_macro_names:
                if re.search(rf'\b{other}\b', val):
                    deps.add(other)
                    deps.update(get_macro_deps(other, visited))
        return deps
    
    all_used = set(used_macros)
    for m in used_macros:
        all_used.update(get_macro_deps(m, set()))
    
    # Resolve macros
    resolved_macros = []
    unresolved_macros = []
    
    for macro_name in sorted(all_used):
        if macro_name.startswith('_'):
            continue
        
        resolved = resolve_macro(parser, macro_name)
        raw = parser.macros[macro_name].value.strip()
        
        if resolved is not None:
            resolved_macros.append((macro_name, resolved, raw))
        else:
            unresolved_macros.append((macro_name, raw))
    
    # Find and resolve array accesses
    array_accesses = find_array_accesses(all_code)
    resolved_arrays = []
    
    for array_name, index_expr in array_accesses:
        result = resolve_array_access(array_name, index_expr, parser, files)
        if result:
            existing = [a for a in resolved_arrays if a['array_name'] == array_name and a['index_expr'] == index_expr]
            if not existing:
                resolved_arrays.append(result)
    
    # Build output
    output = []
    
    # Header
    if total_paths > 1:
        output.append(f"// {'═' * 55}")
        output.append(f"// PATH {path_index} of {total_paths}")
        output.append(f"// {'═' * 55}")
    
    output.append(f"// PATH: {' → '.join(path)}")
    output.append(f"// Target: {target_function}()")
    
    # Show helper functions discovered
    if helper_funcs:
        output.append(f"// Data flow helpers: {', '.join(helper_funcs)}")
    
    output.append("")
    
    # Macros
    if resolved_macros or unresolved_macros:
        output.append("// ========== MACROS (pre-resolved) ==========")
        output.append("")
        
        for name, value, raw in resolved_macros:
            if str(value) == raw or f"({value})" == raw:
                output.append(f"#define {name} {value}")
            else:
                output.append(f"#define {name} {value}  // was: {raw}")
        
        if unresolved_macros:
            output.append("")
            output.append("// Unresolved (non-numeric):")
            for name, raw in unresolved_macros:
                output.append(f"// #define {name} {raw}")
        
        output.append("")
    
    # Array definitions
    if resolved_arrays:
        output.append("// ========== ARRAYS (pre-resolved) ==========")
        output.append("")
        
        for arr in resolved_arrays:
            array_name = arr['array_name']
            values = arr['all_values']
            index = arr['index']
            value = arr['value']
            index_expr = arr['index_expr']
            
            values_str = ', '.join(str(v) for v in values)
            output.append(f"// {array_name}[] = {{ {values_str} }}")
            output.append(f"// {array_name}[{index_expr}] = {array_name}[{index}] = {value}")
            output.append(f"#define {array_name.upper()}_{index_expr} {value}  // {array_name}[{index_expr}]")
            output.append("")
        
        output.append("")
    
    # Functions - Path functions first, then helpers
    output.append("// ========== PATH FUNCTIONS (in execution order) ==========")
    output.append("")
    
    func_index = 1
    for func_name in path[:-1]:
        body = get_function_body(parser, func_name, files)
        if body:
            modified_body = body
            for arr in resolved_arrays:
                array_name = arr['array_name']
                index_expr = arr['index_expr']
                value = arr['value']
                
                pattern = rf'\b{array_name}\s*\[\s*{index_expr}\s*\]'
                replacement = f'{value} /* {array_name}[{index_expr}] */'
                modified_body = re.sub(pattern, replacement, modified_body)
            
            output.append(f"// [{func_index}] {func_name}()")
            output.append(modified_body)
            output.append("")
        else:
            output.append(f"// [{func_index}] {func_name}() - body not found")
            output.append("")
        func_index += 1
    
    # Helper functions (data flow)
    if helper_bodies:
        output.append("// ========== HELPER FUNCTIONS (data flow) ==========")
        output.append("")
        
        for func_name, body in helper_bodies.items():
            modified_body = body
            for arr in resolved_arrays:
                array_name = arr['array_name']
                index_expr = arr['index_expr']
                value = arr['value']
                
                pattern = rf'\b{array_name}\s*\[\s*{index_expr}\s*\]'
                replacement = f'{value} /* {array_name}[{index_expr}] */'
                modified_body = re.sub(pattern, replacement, modified_body)
            
            output.append(f"// [H] {func_name}() - computes value for target")
            output.append(modified_body)
            output.append("")
    
    output.append(f"// [{func_index}] {path[-1]}() ← TARGET (body not needed)")
    
    return '\n'.join(output)


def format_path_to_file(
    project_dir: str,
    target_function: str,
    include_paths: List[str] = None,
    output_file: str = None
) -> str:
    """
    Format path functions and save to file(s).
    
    If multiple paths exist, creates:
      - _path_trace_1.c
      - _path_trace_2.c
      - etc.
    
    If single path, creates:
      - _path_trace.c
    
    Returns the content of the first path.
    """
    # Parse project
    result = analyze(project_dir, target_function, include_paths or [], verbose=False)
    parser = result.get('parser')
    paths = result.get('paths', [])
    
    if not paths:
        return "// No paths found"
    
    # Load files
    files: Dict[str, str] = {}
    project = Path(project_dir)
    for f in project.glob('**/*.c'):
        files[str(f)] = f.read_text(errors='ignore')
    for f in project.glob('**/*.h'):
        files[str(f)] = f.read_text(errors='ignore')
    for inc in (include_paths or []):
        for f in Path(inc).glob('**/*.h'):
            files[str(f)] = f.read_text(errors='ignore')
    
    total_paths = len(paths)
    first_content = None
    
    for idx, path_info in enumerate(paths, 1):
        path = list(path_info.path)
        
        content = format_single_path(
            parser, path, files, target_function,
            path_index=idx, total_paths=total_paths
        )
        
        if idx == 1:
            first_content = content
        
        # Determine output filename
        if total_paths == 1:
            if output_file:
                out_path = output_file
            else:
                out_path = str(Path(project_dir) / '_path_trace.c')
        else:
            out_path = str(Path(project_dir) / f'_path_trace_{idx}.c')
        
        Path(out_path).write_text(content)
        print(f"✅ Saved: {out_path}")
    
    if total_paths > 1:
        print(f"\n📊 Total: {total_paths} paths found")
    
    return first_content


def get_all_path_files(project_dir: str) -> List[str]:
    """Get all path trace files in a project directory."""
    project = Path(project_dir)
    
    # Check for multi-path files first
    multi_files = sorted(project.glob('_path_trace_*.c'))
    if multi_files:
        return [str(f) for f in multi_files]
    
    # Single file
    single_file = project / '_path_trace.c'
    if single_file.exists():
        return [str(single_file)]
    
    return []


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python path_formatter.py <project_dir> [target_function]")
        sys.exit(1)
    
    project_dir = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else "mpf_mfs_open"
    
    test_dir = Path(__file__).parent.parent / 'test_cases'
    includes = [str(test_dir / 'include')] if (test_dir / 'include').exists() else []
    
    content = format_path_to_file(project_dir, target, includes)
    print("\n" + "=" * 60)
    print(content)

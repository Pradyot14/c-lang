#!/usr/bin/env python3
"""
Path Formatter - Extracts and formats path functions in execution order.
Supports multiple paths - generates separate files for each path.

v4: Added function-like macro detection for path transitions
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
    
    if not val:
        return None
    
    # Strip outer parentheses
    while val.startswith('(') and val.endswith(')'):
        inner = val[1:-1].strip()
        if inner.count('(') == inner.count(')'):
            val = inner
        else:
            break
    
    # Plain decimal
    if re.match(r'^-?\d+$', val):
        return int(val)
    
    # Hex
    if re.match(r'^0[xX][0-9a-fA-F]+$', val):
        return int(val, 16)
    
    # Octal
    if re.match(r'^0[0-7]+$', val) and len(val) > 1:
        try:
            return int(val, 8)
        except:
            pass
    
    # Single macro reference
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', val):
        return resolve_macro(parser, val, visited, depth + 1)
    
    # Expression with macros
    expr = val
    all_resolved = True
    
    for macro_name in sorted(parser.macros.keys(), key=len, reverse=True):
        if re.search(rf'\b{re.escape(macro_name)}\b', expr):
            resolved = resolve_macro(parser, macro_name, visited.copy(), depth + 1)
            if resolved is not None:
                expr = re.sub(rf'\b{re.escape(macro_name)}\b', str(resolved), expr)
            else:
                all_resolved = False
    
    if all_resolved:
        clean = expr
        clean = re.sub(r'\(int\)|\(long\)|\(unsigned\)|\(uint32_t\)|\(int32_t\)', '', clean)
        clean = re.sub(r'\bsizeof\s*\([^)]+\)', '4', clean)
        clean = re.sub(r'[uUlL]+$', '', clean)
        clean = re.sub(r'(\d+)[uUlL]+', r'\1', clean)
        
        if re.match(r'^[\d\s\+\-\*\/\(\)\&\|\^\~\<\>\%]+$', clean):
            try:
                result = eval(clean)
                return int(result)
            except:
                pass
    
    return None


def find_macros_in_code(code: str, all_macros: Set[str]) -> Set[str]:
    """Find macro names actually used in code."""
    used = set()
    for macro in all_macros:
        if re.search(rf'\b{re.escape(macro)}\b', code):
            used.add(macro)
    return used


def get_macro_dependencies(parser: CParser, macro_name: str, all_macros: Set[str], visited: Set[str] = None) -> Set[str]:
    """Get all macros that a macro depends on."""
    if visited is None:
        visited = set()
    
    if macro_name in visited:
        return set()
    
    visited.add(macro_name)
    deps = set()
    
    if macro_name in parser.macros:
        val = parser.macros[macro_name].value
        for other in all_macros:
            if re.search(rf'\b{re.escape(other)}\b', val):
                deps.add(other)
                deps.update(get_macro_dependencies(parser, other, all_macros, visited.copy()))
    
    return deps


# ============================================================================
# NEW: Function-like macro detection
# ============================================================================

def find_function_like_macros(files: Dict[str, str], path_functions: List[str], helper_functions: List[str] = None) -> List[Dict]:
    """
    Find function-like macros that are used for PATH TRANSITIONS only.
    
    For path: main → FuncA → FuncB → target
    We check: does main() call a macro that expands to FuncA()?
              does FuncA() call a macro that expands to FuncB()?
    
    Also checks if any path function calls a helper function via macro.
    Only include macros that explain transitions.
    """
    if helper_functions is None:
        helper_functions = []
    
    function_macros = []
    seen_macros = set()
    
    # Build map: actual_function -> list of macros that expand to it
    func_to_macros = {}
    macro_pattern = r'#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s+(.+?)(?:\n|$)'
    
    for file_path, content in files.items():
        for match in re.finditer(macro_pattern, content, re.MULTILINE):
            macro_name = match.group(1)
            macro_args = match.group(2).strip()
            macro_expansion = match.group(3).strip()
            
            # Find which function this macro calls
            func_call_match = re.search(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(', macro_expansion)
            if func_call_match:
                target_func = func_call_match.group(1)
                if target_func not in func_to_macros:
                    func_to_macros[target_func] = []
                func_to_macros[target_func].append({
                    'macro_name': macro_name,
                    'macro_args': macro_args,
                    'expansion': macro_expansion,
                    'target_func': target_func,
                    'file': file_path,
                    'full_definition': f"#define {macro_name}({macro_args}) {macro_expansion}"
                })
    
    def get_function_body_simple(func_name: str) -> Optional[str]:
        """Get function body from files."""
        for content in files.values():
            pattern = rf'\b{re.escape(func_name)}\s*\([^)]*\)\s*\{{'
            if re.search(pattern, content):
                start = content.find(f'{func_name}')
                if start != -1:
                    brace_start = content.find('{', start)
                    if brace_start != -1:
                        depth = 1
                        pos = brace_start + 1
                        while pos < len(content) and depth > 0:
                            if content[pos] == '{':
                                depth += 1
                            elif content[pos] == '}':
                                depth -= 1
                            pos += 1
                        return content[brace_start:pos]
        return None
    
    # All functions that could be callees (path functions + helpers, except target)
    all_callees = set(path_functions[1:]) | set(helper_functions)
    
    # All functions that could be callers (path functions except target, + helpers)
    all_callers = list(path_functions[:-1]) + helper_functions
    
    # Check each caller for macro-based calls to any callee
    for caller in all_callers:
        caller_body = get_function_body_simple(caller)
        if not caller_body:
            continue
        
        # Check if caller uses any macro to call any callee
        for callee in all_callees:
            if callee in func_to_macros:
                for macro_info in func_to_macros[callee]:
                    macro_name = macro_info['macro_name']
                    if macro_name in seen_macros:
                        continue
                    if re.search(rf'\b{re.escape(macro_name)}\s*\(', caller_body):
                        seen_macros.add(macro_name)
                        function_macros.append(macro_info)
    
    return function_macros


def find_function_declarations(files: Dict[str, str], func_names: List[str]) -> Dict[str, str]:
    """
    Find function declarations/prototypes from header files.
    
    Returns dict: func_name -> declaration string
    """
    declarations = {}
    
    for func_name in func_names:
        # Pattern for function declaration (not definition - no opening brace)
        # Matches various styles:
        #   void *FuncName(int, char, ...);
        #   int FuncName (int a, char b);
        #   static inline int FuncName(void);
        patterns = [
            # Standard declaration with types
            rf'^[^#]*?\b(\w+[\s\*]+{re.escape(func_name)}\s*\([^)]*\))\s*;',
            # With return type on same line
            rf'^\s*(?:static\s+)?(?:inline\s+)?(?:extern\s+)?(\w[\w\s\*]*\b{re.escape(func_name)}\s*\([^)]*\))\s*;',
        ]
        
        for file_path, content in files.items():
            # Only look in header files
            if not file_path.endswith('.h'):
                continue
            
            for pattern in patterns:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    decl = match.group(1).strip()
                    if func_name not in declarations:
                        declarations[func_name] = decl + ";"
                    break
    
    return declarations


# ============================================================================
# Array handling
# ============================================================================

def find_array_accesses(code: str) -> List[Tuple[str, str]]:
    """Find array accesses like: array_name[INDEX]"""
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\[\s*([a-zA-Z0-9_]+)\s*\]'
    matches = re.findall(pattern, code)
    skip_names = {'sizeof', 'typeof', 'alignof'}
    return [(name, idx) for name, idx in matches if name not in skip_names]


def find_array_definition(array_name: str, files: Dict[str, str]) -> Optional[Dict]:
    """Find static/const array definition and extract values."""
    for file_path, content in files.items():
        patterns = [
            rf'(?:static\s+)?(?:const\s+)?(?:int|long|unsigned|short|uint32_t|int32_t)\s+{re.escape(array_name)}\s*\[\s*\]\s*=\s*\{{([^}}]+)\}}',
            rf'(?:static\s+)?(?:const\s+)?(?:int|long|unsigned|short|uint32_t|int32_t)\s+{re.escape(array_name)}\s*\[\s*\d*\s*\]\s*=\s*\{{([^}}]+)\}}',
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
                
                return {'values': values, 'raw': match.group(0), 'file': file_path}
    
    return None


def resolve_array_access(array_name: str, index_expr: str, parser: CParser, files: Dict[str, str]) -> Optional[Dict]:
    """Resolve array[INDEX] to actual value."""
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
# Data Flow Analysis
# ============================================================================

def find_target_call_args(code: str, target_function: str) -> Dict:
    """Find arguments passed to target function."""
    pattern = rf'{re.escape(target_function)}\s*\(\s*([^)]+)\s*\)'
    match = re.search(pattern, code)
    
    if not match:
        return {'raw_args': [], 'variables': set()}
    
    args_str = match.group(1)
    
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
    
    variables = set()
    for arg in args:
        identifiers = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', arg)
        for ident in identifiers:
            if ident not in {'NULL', 'sizeof', 'true', 'false', 'void', 'int', 'char', 'const'}:
                variables.add(ident)
    
    return {'raw_args': args, 'variables': variables}


def find_variable_assignments(code: str, variables: Set[str]) -> Dict[str, List[str]]:
    """Find function calls that assign to variables."""
    assignments = {}
    
    for var in variables:
        pattern = rf'\b{re.escape(var)}\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, code)
        
        if matches:
            funcs = [m for m in matches if m not in {'sizeof', 'typeof', 'offsetof', 'int', 'long', 'malloc', 'calloc', 'free'}]
            if funcs:
                assignments[var] = funcs
    
    return assignments


def build_macro_to_func_map(files: Dict[str, str]) -> Dict[str, str]:
    """
    Build a map from function-like macro names to the actual functions they call.
    
    Example: RbtMfsGetDefine -> RbtMfsGetDefineFunc
    """
    macro_to_func = {}
    macro_pattern = r'#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\('
    
    for content in files.values():
        for match in re.finditer(macro_pattern, content, re.MULTILINE):
            macro_name = match.group(1)
            func_name = match.group(2)
            macro_to_func[macro_name] = func_name
    
    return macro_to_func


def find_data_flow_functions(
    path_functions_code: str,
    target_function: str,
    parser: CParser,
    files: Dict[str, str],
    max_depth: int = 5
) -> List[str]:
    """Find helper functions that compute values passed to target function."""
    helper_functions = []
    visited = set()
    
    # Build macro->function mapping
    macro_to_func = build_macro_to_func_map(files)
    
    def resolve_to_function(name: str) -> Optional[str]:
        """Resolve a name to actual function (handles macro indirection)."""
        if name in parser.functions:
            return name
        # Check if it's a macro that maps to a function
        if name in macro_to_func:
            actual_func = macro_to_func[name]
            if actual_func in parser.functions:
                return actual_func
        return None
    
    def trace_helpers(code: str, depth: int):
        if depth > max_depth:
            return
        
        target_info = find_target_call_args(code, target_function)
        variables = target_info['variables']
        
        if not variables:
            return
        
        assignments = find_variable_assignments(code, variables)
        
        for var, funcs in assignments.items():
            for func_name in funcs:
                if func_name in visited or func_name == target_function:
                    continue
                
                visited.add(func_name)
                
                # Resolve macro to actual function if needed
                actual_func = resolve_to_function(func_name)
                if actual_func:
                    if actual_func not in visited:
                        visited.add(actual_func)
                    helper_functions.append(actual_func)
                    helper_body = get_function_body(parser, actual_func, files)
                    if helper_body:
                        trace_nested_helpers(helper_body, depth + 1)
    
    def trace_nested_helpers(code: str, depth: int):
        if depth > max_depth:
            return
        
        pattern1 = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)'
        pattern2 = r'\breturn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        
        for match in re.finditer(pattern1, code):
            func_name = match.group(2)
            if func_name in visited:
                continue
            if func_name in {'sizeof', 'typeof', 'offsetof', 'int', 'long', 'malloc', 'calloc', 'free'}:
                continue
            if func_name == target_function:
                continue
            
            # Resolve macro to actual function if needed
            actual_func = resolve_to_function(func_name)
            if actual_func:
                if actual_func not in visited:
                    visited.add(actual_func)
                    helper_functions.append(actual_func)
                    helper_body = get_function_body(parser, actual_func, files)
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
            
            # Resolve macro to actual function if needed
            actual_func = resolve_to_function(func_name)
            if actual_func:
                if actual_func not in visited:
                    visited.add(actual_func)
                    helper_functions.append(actual_func)
                    helper_body = get_function_body(parser, actual_func, files)
                    if helper_body:
                        trace_nested_helpers(helper_body, depth + 1)
    
    trace_helpers(path_functions_code, 0)
    return helper_functions


# ============================================================================
# Main formatting function
# ============================================================================

def format_single_path(
    parser: CParser,
    path: List[str],
    files: Dict[str, str],
    target_function: str,
    path_index: int = 1,
    total_paths: int = 1
) -> str:
    """Format a single path to string."""
    
    # Collect path function bodies
    func_bodies = {}
    for func_name in path[:-1]:
        body = get_function_body(parser, func_name, files)
        if body:
            func_bodies[func_name] = body
    
    path_code = '\n'.join(func_bodies.values())
    
    # Find data flow helper functions
    helper_funcs = find_data_flow_functions(path_code, target_function, parser, files)
    
    # Get helper function bodies
    helper_bodies = {}
    for func_name in helper_funcs:
        body = get_function_body(parser, func_name, files)
        if body:
            helper_bodies[func_name] = body
    
    # Combine all code for analysis
    all_code = path_code + '\n' + '\n'.join(helper_bodies.values())
    
    # ========== Find function-like macros for path transitions ==========
    func_macros = find_function_like_macros(files, list(path), helper_funcs)
    
    # Find function declarations for functions called via macros
    macro_target_funcs = list(set(fm['target_func'] for fm in func_macros))
    func_declarations = find_function_declarations(files, macro_target_funcs)
    
    # ========== MACRO FILTERING ==========
    all_macro_names = set(parser.macros.keys())
    
    # Find macros directly used in the code
    used_macros = find_macros_in_code(all_code, all_macro_names)
    
    # Add macro dependencies
    all_used = set(used_macros)
    for m in used_macros:
        all_used.update(get_macro_dependencies(parser, m, all_macro_names))
    
    # Resolve and categorize macros
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
            if raw and not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', raw):
                if re.search(r'\d|0x', raw, re.IGNORECASE):
                    unresolved_macros.append((macro_name, raw))
    
    # Find and resolve array accesses
    array_accesses = find_array_accesses(all_code)
    resolved_arrays = []
    seen_arrays = set()
    
    for array_name, index_expr in array_accesses:
        key = (array_name, index_expr)
        if key in seen_arrays:
            continue
        seen_arrays.add(key)
        
        result = resolve_array_access(array_name, index_expr, parser, files)
        if result:
            resolved_arrays.append(result)
    
    # ========== BUILD OUTPUT ==========
    output = []
    
    # Header
    if total_paths > 1:
        output.append(f"// {'═' * 55}")
        output.append(f"// PATH {path_index} of {total_paths}")
        output.append(f"// {'═' * 55}")
    
    output.append(f"// PATH: {' → '.join(path)}")
    output.append(f"// Target: {target_function}()")
    
    if helper_funcs:
        output.append(f"// Data flow helpers: {', '.join(helper_funcs)}")
    
    output.append("")
    
    # ========== NEW: Function-like macros section ==========
    if func_macros:
        output.append("// ========== FUNCTION MACROS (path transitions) ==========")
        output.append("// These macros wrap function calls in the path:")
        output.append("")
        
        seen_macros = set()
        for fm in func_macros:
            if fm['macro_name'] in seen_macros:
                continue
            seen_macros.add(fm['macro_name'])
            
            output.append(f"// {fm['macro_name']}(...) expands to {fm['target_func']}(...)")
            output.append(fm['full_definition'])
            
            # Add function declaration if available
            if fm['target_func'] in func_declarations:
                output.append(func_declarations[fm['target_func']])
            
            output.append("")
    
    # Numeric macros section
    if resolved_macros:
        output.append("// ========== MACROS (pre-resolved) ==========")
        output.append("")
        
        for name, value, raw in resolved_macros:
            raw_clean = raw.strip('() ')
            if str(value) == raw_clean or str(value) == raw:
                output.append(f"#define {name} {value}")
            else:
                output.append(f"#define {name} {value}  // was: {raw}")
        
        output.append("")
    
    if unresolved_macros:
        output.append("// Unresolved (complex expressions):")
        for name, raw in unresolved_macros:
            output.append(f"// #define {name} {raw}")
        output.append("")
    
    # Arrays section
    if resolved_arrays:
        output.append("// ========== ARRAYS (pre-resolved) ==========")
        output.append("")
        
        for arr in resolved_arrays:
            array_name = arr['array_name']
            values = arr['all_values']
            index = arr['index']
            value = arr['value']
            index_expr = arr['index_expr']
            
            values_str = ', '.join(str(v) for v in values[:10])
            if len(values) > 10:
                values_str += f", ... ({len(values)} total)"
            
            output.append(f"// {array_name}[] = {{ {values_str} }}")
            output.append(f"// {array_name}[{index_expr}] = {array_name}[{index}] = {value}")
            output.append(f"#define {array_name.upper()}_{index_expr} {value}")
            output.append("")
    
    # Path functions
    output.append("// ========== PATH FUNCTIONS (in execution order) ==========")
    output.append("")
    
    func_index = 1
    for func_name in path[:-1]:
        if func_name in func_bodies:
            body = func_bodies[func_name]
            
            # Replace resolved array accesses
            for arr in resolved_arrays:
                pattern = rf'\b{re.escape(arr["array_name"])}\s*\[\s*{re.escape(arr["index_expr"])}\s*\]'
                replacement = f'{arr["value"]} /* {arr["array_name"]}[{arr["index_expr"]}] */'
                body = re.sub(pattern, replacement, body)
            
            output.append(f"// [{func_index}] {func_name}()")
            output.append(body)
            output.append("")
        else:
            output.append(f"// [{func_index}] {func_name}() - body not found")
            output.append("")
        func_index += 1
    
    # Helper functions
    if helper_bodies:
        output.append("// ========== HELPER FUNCTIONS (data flow) ==========")
        output.append("")
        
        for func_name, body in helper_bodies.items():
            for arr in resolved_arrays:
                pattern = rf'\b{re.escape(arr["array_name"])}\s*\[\s*{re.escape(arr["index_expr"])}\s*\]'
                replacement = f'{arr["value"]} /* {arr["array_name"]}[{arr["index_expr"]}] */'
                body = re.sub(pattern, replacement, body)
            
            output.append(f"// [H] {func_name}() - computes value for target")
            output.append(body)
            output.append("")
    
    output.append(f"// [{func_index}] {path[-1]}() ← TARGET (body not needed)")
    
    return '\n'.join(output)


def format_path_to_file(
    project_dir: str,
    target_function: str,
    include_paths: List[str] = None,
    output_file: str = None
) -> str:
    """Format path functions and save to file(s)."""
    
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
        
        if total_paths == 1:
            out_path = output_file or str(Path(project_dir) / '_path_trace.c')
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
    
    multi_files = sorted(project.glob('_path_trace_*.c'))
    if multi_files:
        return [str(f) for f in multi_files]
    
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

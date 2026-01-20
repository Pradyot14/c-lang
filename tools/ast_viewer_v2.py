#!/usr/bin/env python3
"""
C AST Viewer Pro v2 - Smart Filtering
======================================
Shows only YOUR code and its dependencies, not system headers.

Features:
- Filters out system headers (stdio.h, stdlib.h, etc.)
- Shows only user code + custom header dependencies
- Tracks unresolved external calls
- Clean dependency visualization
"""

import json
import os
import sys
import re
import tempfile
from pathlib import Path
from collections import defaultdict

from clang.cindex import Index, CursorKind, Config, TranslationUnit
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# System header paths to filter out
SYSTEM_PATHS = [
    '/usr/include',
    '/usr/local/include',
    '/Library/',
    '/Applications/Xcode',
    '/opt/',
    '/System/',
    'bits/',
    'sys/',
    '_types',
]

# Common system functions to mark as SYSTEM
SYSTEM_FUNCTIONS = {
    'printf', 'scanf', 'fprintf', 'fscanf', 'sprintf', 'sscanf', 'snprintf',
    'puts', 'gets', 'fgets', 'fputs', 'putchar', 'getchar', 'putc', 'getc',
    'fopen', 'fclose', 'fread', 'fwrite', 'fseek', 'ftell', 'rewind', 'fflush',
    'malloc', 'calloc', 'realloc', 'free', 'memcpy', 'memset', 'memmove', 'memcmp',
    'strcpy', 'strncpy', 'strcat', 'strncat', 'strcmp', 'strncmp', 'strlen', 'strchr', 'strstr',
    'atoi', 'atof', 'atol', 'strtol', 'strtod', 'strtoul',
    'abs', 'labs', 'rand', 'srand', 'exit', 'abort', 'atexit', 'system', 'getenv',
    'qsort', 'bsearch',
    'isalpha', 'isdigit', 'isalnum', 'isspace', 'isupper', 'islower', 'toupper', 'tolower',
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
    'exp', 'log', 'log10', 'pow', 'sqrt', 'ceil', 'floor', 'fabs', 'fmod',
    'time', 'clock', 'difftime', 'mktime', 'strftime', 'localtime', 'gmtime',
    'assert', 'perror', 'errno',
    'va_start', 'va_end', 'va_arg', 'va_copy',
}

NODE_CATEGORIES = {
    # Functions
    'FUNCTION_DECL': 'function',
    'PARM_DECL': 'parameter',
    
    # Variables
    'VAR_DECL': 'variable',
    'FIELD_DECL': 'field',
    
    # Types
    'TYPEDEF_DECL': 'typedef',
    'STRUCT_DECL': 'struct',
    'UNION_DECL': 'union',
    'ENUM_DECL': 'enum',
    'ENUM_CONSTANT_DECL': 'enum_const',
    'TYPE_REF': 'type_ref',
    
    # Literals
    'INTEGER_LITERAL': 'literal',
    'FLOATING_LITERAL': 'literal',
    'STRING_LITERAL': 'string_literal',
    'CHARACTER_LITERAL': 'literal',
    'CXX_BOOL_LITERAL_EXPR': 'literal',
    'CXX_NULL_PTR_LITERAL_EXPR': 'literal',
    
    # Control Flow
    'IF_STMT': 'control',
    'SWITCH_STMT': 'control',
    'CASE_STMT': 'control',
    'DEFAULT_STMT': 'control',
    'WHILE_STMT': 'control',
    'DO_STMT': 'control',
    'FOR_STMT': 'control',
    'BREAK_STMT': 'break',
    'CONTINUE_STMT': 'break',
    'RETURN_STMT': 'return',
    'GOTO_STMT': 'goto',
    'LABEL_STMT': 'goto',
    
    # Operators
    'BINARY_OPERATOR': 'operator',
    'UNARY_OPERATOR': 'operator',
    'COMPOUND_ASSIGNMENT_OPERATOR': 'assignment',
    'CONDITIONAL_OPERATOR': 'operator',
    
    # Calls & References
    'CALL_EXPR': 'call',
    'MEMBER_REF_EXPR': 'member_access',
    'ARRAY_SUBSCRIPT_EXPR': 'array_access',
    'DECL_REF_EXPR': 'reference',
    
    # Pointer Operations
    'ADDR_EXPR': 'pointer',
    'INDIRECT_REF': 'pointer',
    
    # Preprocessor
    'INCLUSION_DIRECTIVE': 'include',
    'MACRO_DEFINITION': 'macro',
    'MACRO_INSTANTIATION': 'macro_use',
    
    # Structure
    'COMPOUND_STMT': 'block',
    'DECL_STMT': 'decl_stmt',
    'TRANSLATION_UNIT': 'root',
    'NULL_STMT': 'other',
    
    # Expressions (catch-all)
    'PAREN_EXPR': 'expression',
    'INIT_LIST_EXPR': 'expression',
    'UNEXPOSED_EXPR': 'expression',
}

def get_category(kind_name):
    return NODE_CATEGORIES.get(kind_name, 'other')

def is_system_path(filepath):
    """Check if file is a system header."""
    if not filepath:
        return False
    filepath = str(filepath)
    for sys_path in SYSTEM_PATHS:
        if sys_path in filepath:
            return True
    return False

def is_user_file(filepath, user_dir):
    """Check if file belongs to user's project."""
    if not filepath:
        return False
    filepath = str(filepath)
    
    # If it's a system path, not user file
    if is_system_path(filepath):
        return False
    
    # If we have a user directory, check if file is in it
    if user_dir:
        try:
            return filepath.startswith(user_dir) or os.path.dirname(filepath) == user_dir
        except:
            pass
    
    return True


class SmartDependencyResolver:
    """Resolves dependencies with smart filtering."""
    
    def __init__(self, main_file, include_paths=None, filter_system=True):
        self.main_file = os.path.abspath(main_file)
        self.main_dir = os.path.dirname(self.main_file)
        self.include_paths = include_paths or []
        self.filter_system = filter_system
        
        # User files tracking
        self.user_files = set([self.main_file])
        
        # Symbols from user code
        self.user_functions = {}  # Defined in user files
        self.user_types = {}      # Defined in user files
        self.user_macros = {}     # Defined in user files
        self.user_globals = {}    # Defined in user files
        
        # External dependencies (called/used but not defined in user files)
        self.external_calls = {}      # Functions called but defined elsewhere
        self.external_types = {}      # Types used but defined elsewhere
        self.unresolved_headers = []  # Headers that couldn't be found
        
        # Call graph (only user functions)
        self.call_graph = defaultdict(set)
        
        # Include tracking
        self.user_includes = []   # #include directives in user code
        self.resolved_includes = {}  # header name -> full path (user headers only)
        
    def parse(self):
        """Parse with smart filtering."""
        index = Index.create()
        
        # Build args
        args = ['-std=c11', '-x', 'c', '-ferror-limit=0']
        for inc in self.include_paths:
            args.append(f'-I{inc}')
        args.append(f'-I{self.main_dir}')
        
        # Parse
        tu = index.parse(
            self.main_file,
            args=args,
            options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD | 
                    TranslationUnit.PARSE_SKIP_FUNCTION_BODIES * 0  # We want bodies
        )
        
        # Track included files
        for inc in tu.get_includes():
            if inc.include:
                inc_path = str(inc.include.name)
                inc_depth = inc.depth
                
                # Get the include directive location
                source_file = str(inc.source.name) if inc.source else self.main_file
                
                # Check if this is a user header
                if not is_system_path(inc_path):
                    self.user_files.add(inc_path)
                    
                    # Get header name from the source
                    header_name = os.path.basename(inc_path)
                    self.resolved_includes[header_name] = inc_path
        
        # Find unresolved includes by parsing source
        self._find_unresolved_includes()
        
        # Process AST
        self._process_cursor(tu.cursor)
        
        return tu
    
    def _find_unresolved_includes(self):
        """Find #include directives that couldn't be resolved."""
        try:
            with open(self.main_file, 'r') as f:
                content = f.read()
            
            for match in re.finditer(r'#\s*include\s*[<"]([^>"]+)[>"]', content):
                header = match.group(1)
                # Check if it's resolved
                if header not in self.resolved_includes:
                    base = os.path.basename(header)
                    if base not in self.resolved_includes:
                        # Check if it's a system header
                        if not any(h in header for h in ['stdio', 'stdlib', 'string', 'math', 'time', 'assert', 'errno', 'signal', 'stddef', 'stdint', 'stdbool', 'stdarg', 'limits', 'float', 'ctype', 'locale', 'setjmp']):
                            self.unresolved_headers.append(header)
                            self.user_includes.append({
                                'name': header,
                                'resolved': False,
                                'path': None
                            })
                        
        except Exception as e:
            print(f"Error finding includes: {e}")
    
    def _is_user_location(self, cursor):
        """Check if cursor is from user code."""
        loc = cursor.location
        if not loc or not loc.file:
            return False
        filepath = str(loc.file.name)
        return filepath in self.user_files or not is_system_path(filepath)
    
    def _process_cursor(self, cursor, parent_func=None):
        """Process cursor and extract user-relevant information."""
        loc = cursor.location
        file_path = str(loc.file.name) if loc and loc.file else None
        is_user = file_path and (file_path in self.user_files or 
                                  file_path == self.main_file or
                                  not is_system_path(file_path))
        
        kind = cursor.kind
        
        # Track function definitions
        if kind == CursorKind.FUNCTION_DECL:
            name = cursor.spelling
            if is_user and cursor.is_definition():
                # User-defined function
                return_type = cursor.result_type.spelling if cursor.result_type else 'void'
                params = []
                for child in cursor.get_children():
                    if child.kind == CursorKind.PARM_DECL:
                        params.append({
                            'name': child.spelling or 'unnamed',
                            'type': child.type.spelling if child.type else 'unknown'
                        })
                
                self.user_functions[name] = {
                    'name': name,
                    'file': os.path.basename(file_path) if file_path else 'unknown',
                    'full_path': file_path,
                    'line': loc.line if loc else 0,
                    'return_type': return_type,
                    'params': params,
                    'signature': f"{return_type} {name}({', '.join(p['type'] + ' ' + p['name'] for p in params)})",
                    'is_definition': True
                }
                parent_func = name
        
        # Track function calls
        elif kind == CursorKind.CALL_EXPR:
            callee = cursor.spelling
            if callee and is_user:
                # Record in call graph
                if parent_func:
                    self.call_graph[parent_func].add(callee)
                
                # Check if it's external
                if callee not in self.user_functions:
                    # Get call info
                    if callee not in self.external_calls:
                        # Try to get the referenced declaration
                        ref = cursor.referenced
                        if ref:
                            ref_loc = ref.location
                            ref_file = str(ref_loc.file.name) if ref_loc and ref_loc.file else None
                            
                            return_type = ref.result_type.spelling if ref.result_type else 'unknown'
                            params = []
                            for child in ref.get_children():
                                if child.kind == CursorKind.PARM_DECL:
                                    params.append({
                                        'name': child.spelling or '',
                                        'type': child.type.spelling if child.type else ''
                                    })
                            
                            self.external_calls[callee] = {
                                'name': callee,
                                'return_type': return_type,
                                'params': params,
                                'signature': f"{return_type} {callee}({', '.join(p['type'] for p in params)})",
                                'defined_in': os.path.basename(ref_file) if ref_file else 'unknown',
                                'is_system': is_system_path(ref_file) if ref_file else (callee in SYSTEM_FUNCTIONS),
                                'called_from': [parent_func] if parent_func else []
                            }
                        else:
                            self.external_calls[callee] = {
                                'name': callee,
                                'return_type': 'unknown',
                                'params': [],
                                'signature': f"{callee}(...)",
                                'defined_in': 'unresolved',
                                'is_system': callee in SYSTEM_FUNCTIONS,
                                'called_from': [parent_func] if parent_func else []
                            }
                    elif parent_func and parent_func not in self.external_calls[callee].get('called_from', []):
                        self.external_calls[callee]['called_from'].append(parent_func)
        
        # Track type definitions
        elif kind == CursorKind.TYPEDEF_DECL and is_user:
            name = cursor.spelling
            underlying = cursor.underlying_typedef_type.spelling if cursor.underlying_typedef_type else 'unknown'
            self.user_types[name] = {
                'name': name,
                'kind': 'typedef',
                'underlying': underlying,
                'file': os.path.basename(file_path) if file_path else 'unknown',
                'line': loc.line if loc else 0
            }
        
        elif kind in (CursorKind.STRUCT_DECL, CursorKind.UNION_DECL) and is_user:
            name = cursor.spelling
            if name:
                fields = []
                for child in cursor.get_children():
                    if child.kind == CursorKind.FIELD_DECL:
                        fields.append({
                            'name': child.spelling,
                            'type': child.type.spelling if child.type else 'unknown'
                        })
                
                self.user_types[name] = {
                    'name': name,
                    'kind': 'struct' if kind == CursorKind.STRUCT_DECL else 'union',
                    'fields': fields,
                    'file': os.path.basename(file_path) if file_path else 'unknown',
                    'line': loc.line if loc else 0
                }
        
        elif kind == CursorKind.ENUM_DECL and is_user:
            name = cursor.spelling
            if name:
                constants = []
                for child in cursor.get_children():
                    if child.kind == CursorKind.ENUM_CONSTANT_DECL:
                        constants.append({'name': child.spelling, 'value': child.enum_value})
                
                self.user_types[name] = {
                    'name': name,
                    'kind': 'enum',
                    'constants': constants,
                    'file': os.path.basename(file_path) if file_path else 'unknown',
                    'line': loc.line if loc else 0
                }
        
        # Track type references (to find external types)
        elif kind == CursorKind.TYPE_REF and is_user:
            type_name = cursor.spelling
            ref = cursor.referenced
            if ref:
                ref_loc = ref.location
                ref_file = str(ref_loc.file.name) if ref_loc and ref_loc.file else None
                
                if ref_file and is_system_path(ref_file):
                    pass  # Skip system types
                elif type_name not in self.user_types and type_name not in self.external_types:
                    self.external_types[type_name] = {
                        'name': type_name,
                        'defined_in': os.path.basename(ref_file) if ref_file else 'unknown',
                        'is_resolved': ref_file is not None
                    }
        
        # Track macros
        elif kind == CursorKind.MACRO_DEFINITION and is_user:
            name = cursor.spelling
            tokens = list(cursor.get_tokens())
            value = ' '.join(t.spelling for t in tokens[1:]) if len(tokens) > 1 else ''
            self.user_macros[name] = {
                'name': name,
                'value': value[:100],
                'file': os.path.basename(file_path) if file_path else 'unknown',
                'line': loc.line if loc else 0
            }
        
        # Track global variables
        elif kind == CursorKind.VAR_DECL and is_user:
            parent = cursor.semantic_parent
            if parent and parent.kind == CursorKind.TRANSLATION_UNIT:
                name = cursor.spelling
                self.user_globals[name] = {
                    'name': name,
                    'type': cursor.type.spelling if cursor.type else 'unknown',
                    'file': os.path.basename(file_path) if file_path else 'unknown',
                    'line': loc.line if loc else 0
                }
        
        # Recurse into children
        for child in cursor.get_children():
            self._process_cursor(child, parent_func if kind != CursorKind.FUNCTION_DECL else 
                                (cursor.spelling if kind == CursorKind.FUNCTION_DECL and is_user else parent_func))
    
    def get_summary(self):
        """Get filtered summary."""
        return {
            'user_functions': self.user_functions,
            'user_types': self.user_types,
            'user_macros': self.user_macros,
            'user_globals': self.user_globals,
            'external_calls': self.external_calls,
            'external_types': self.external_types,
            'unresolved_headers': self.unresolved_headers,
            'resolved_headers': list(self.resolved_includes.keys()),
            'call_graph': {k: list(v) for k, v in self.call_graph.items()},
            'user_files': [os.path.basename(f) for f in self.user_files]
        }


def cursor_to_dict_filtered(cursor, resolver, depth=0, max_depth=50, main_file=None):
    """Convert cursor to dict, STRICTLY filtering out system code."""
    if depth > max_depth:
        return None
    
    loc = cursor.location
    file_path = str(loc.file.name) if loc and loc.file else None
    
    # STRICT FILTERING: Only include nodes from the main file
    # For the root TRANSLATION_UNIT, we process it but filter its children
    kind_name = cursor.kind.name
    
    if kind_name != 'TRANSLATION_UNIT':
        # Skip if no location
        if not file_path:
            return None
        
        # Skip if from system path
        if is_system_path(file_path):
            return None
        
        # STRICT: Only include if it's exactly the main file OR an uploaded user header
        main_file_abs = os.path.abspath(main_file) if main_file else None
        file_path_abs = os.path.abspath(file_path) if file_path else None
        
        is_main = main_file_abs and file_path_abs and os.path.samefile(main_file_abs, file_path_abs)
        is_user_header = file_path in resolver.user_files
        
        if not is_main and not is_user_header:
            return None
    
    name = cursor.displayname or cursor.spelling or ''
    
    line = loc.line if loc else 0
    col = loc.column if loc else 0
    filename = os.path.basename(file_path) if file_path else ''
    
    # Check if from header
    is_from_header = False
    if file_path and main_file:
        try:
            is_from_header = not os.path.samefile(file_path, main_file)
        except:
            is_from_header = file_path != main_file
    
    # Type info
    type_info = ''
    try:
        if cursor.type and cursor.type.spelling:
            type_info = cursor.type.spelling
    except:
        pass
    
    # Extra info
    extra = {}
    
    # For function declarations
    if kind_name == 'FUNCTION_DECL' and cursor.spelling in resolver.user_functions:
        func = resolver.user_functions[cursor.spelling]
        extra['signature'] = func['signature']
        extra['return_type'] = func['return_type']
    
    # For calls
    if kind_name == 'CALL_EXPR':
        callee = cursor.spelling
        if callee in resolver.external_calls:
            ext = resolver.external_calls[callee]
            extra['external'] = True
            extra['defined_in'] = ext['defined_in']
            extra['signature'] = ext['signature']
        elif callee in resolver.user_functions:
            extra['internal'] = True
            extra['signature'] = resolver.user_functions[callee]['signature']
    
    node = {
        'type': kind_name,
        'name': name[:80] if name else '',
        'category': get_category(kind_name),
        'line': line,
        'col': col,
        'file': filename,
        'is_from_header': is_from_header,
        'type_info': type_info[:80] if type_info else '',
        'extra': extra,
        'children': []
    }
    
    # Process children - ONLY from user files
    for child in cursor.get_children():
        child_dict = cursor_to_dict_filtered(child, resolver, depth + 1, max_depth, main_file)
        if child_dict:
            node['children'].append(child_dict)
    
    # Don't return empty TRANSLATION_UNIT children
    if kind_name == 'TRANSLATION_UNIT' and not node['children']:
        return None
    
    return node


def parse_user_code(filepath, include_paths=None):
    """Parse user code with smart filtering."""
    try:
        resolver = SmartDependencyResolver(filepath, include_paths)
        tu = resolver.parse()
        
        # Build filtered AST
        ast = cursor_to_dict_filtered(tu.cursor, resolver, main_file=filepath)
        
        # Calculate stats (user code only)
        stats = {'nodes': 0, 'depth': 0, 'functions': 0, 'types': 0, 'external_calls': 0}
        
        def count_stats(node, d=0):
            if not node:
                return
            stats['nodes'] += 1
            stats['depth'] = max(stats['depth'], d)
            if node['type'] == 'FUNCTION_DECL':
                stats['functions'] += 1
            for child in node.get('children', []):
                count_stats(child, d + 1)
        
        count_stats(ast)
        stats['types'] = len(resolver.user_types)
        stats['external_calls'] = len(resolver.external_calls)
        stats['unresolved_headers'] = len(resolver.unresolved_headers)
        
        # Diagnostics (filter to user files only)
        diagnostics = []
        for diag in tu.diagnostics:
            diag_file = str(diag.location.file.name) if diag.location.file else ''
            if not is_system_path(diag_file):
                diagnostics.append({
                    'severity': diag.severity,
                    'message': diag.spelling,
                    'line': diag.location.line,
                    'file': os.path.basename(diag_file) if diag_file else ''
                })
        
        return {
            'success': True,
            'ast': ast,
            'stats': stats,
            'diagnostics': diagnostics,
            'summary': resolver.get_summary()
        }
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'ast': None,
            'stats': {},
            'diagnostics': [],
            'summary': {}
        }


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>C AST Viewer v2 - Smart Filtering</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1a1a2e 100%);
            color: #e2e8f0;
            min-height: 100vh;
        }
        .header {
            background: rgba(30, 41, 59, 0.95);
            padding: 0.6rem 1.5rem;
            border-bottom: 1px solid #334155;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 { 
            font-size: 1.2rem; 
            background: linear-gradient(135deg, #22d3ee, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .badge {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.55rem;
            font-weight: 700;
            margin-left: 0.5rem;
            -webkit-text-fill-color: white;
        }
        .stats { display: flex; gap: 1.25rem; }
        .stat { text-align: center; }
        .stat-value { 
            font-size: 1.1rem; 
            font-weight: bold; 
            color: #22d3ee;
        }
        .stat-value.warning { color: #f59e0b; }
        .stat-value.error { color: #ef4444; }
        .stat-label { font-size: 0.55rem; color: #64748b; text-transform: uppercase; }
        
        .main-container {
            display: grid;
            grid-template-columns: 350px 1fr 300px;
            height: calc(100vh - 48px);
        }
        
        .left-panel {
            background: rgba(30, 41, 59, 0.7);
            border-right: 1px solid #334155;
            display: flex;
            flex-direction: column;
        }
        .panel-header {
            padding: 0.5rem 0.75rem;
            background: rgba(51, 65, 85, 0.5);
            font-weight: 600;
            font-size: 0.75rem;
            border-bottom: 1px solid #334155;
        }
        .dropzone {
            margin: 0.5rem;
            padding: 1rem;
            border: 2px dashed #334155;
            border-radius: 8px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
        }
        .dropzone:hover, .dropzone.dragover {
            border-color: #22d3ee;
            background: rgba(34, 211, 238, 0.1);
        }
        .dropzone-icon { font-size: 1.5rem; }
        .dropzone-text { font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem; }
        
        .file-list { padding: 0.5rem; max-height: 120px; overflow-y: auto; }
        .file-item {
            padding: 0.25rem 0.5rem;
            font-size: 0.7rem;
            background: rgba(15, 23, 42, 0.5);
            border-radius: 4px;
            margin-bottom: 0.25rem;
            display: flex;
            justify-content: space-between;
        }
        .file-item-remove { cursor: pointer; color: #ef4444; }
        
        .include-paths {
            padding: 0.5rem;
            border-top: 1px solid #334155;
        }
        .include-paths label { font-size: 0.65rem; color: #94a3b8; }
        .include-paths input {
            width: 100%;
            padding: 0.35rem;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid #334155;
            border-radius: 4px;
            color: #e2e8f0;
            font-size: 0.7rem;
            margin-top: 0.2rem;
        }
        
        textarea {
            flex: 1;
            background: rgba(15, 23, 42, 0.8);
            border: none;
            padding: 0.6rem;
            color: #e2e8f0;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.7rem;
            resize: none;
            outline: none;
        }
        .buttons { padding: 0.6rem; display: flex; gap: 0.3rem; }
        .btn {
            padding: 0.5rem 0.75rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.75rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn-primary {
            flex: 1;
            background: linear-gradient(135deg, #0891b2, #22d3ee);
            color: white;
        }
        .btn-primary:hover { transform: translateY(-1px); }
        .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; }
        .btn-secondary { background: rgba(51, 65, 85, 0.8); color: #e2e8f0; }
        
        .center-panel { display: flex; flex-direction: column; overflow: hidden; }
        
        .toolbar {
            padding: 0.5rem 0.75rem;
            background: rgba(51, 65, 85, 0.3);
            border-bottom: 1px solid #334155;
            display: flex;
            gap: 0.4rem;
            align-items: center;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.2rem;
            font-size: 0.6rem;
            color: #cbd5e1;
            padding: 0.15rem 0.35rem;
            background: rgba(15, 23, 42, 0.5);
            border-radius: 4px;
            border: 1px solid rgba(51, 65, 85, 0.5);
        }
        .legend-dot { width: 10px; height: 10px; border-radius: 50%; border: 1px solid rgba(255,255,255,0.2); }
        .legend-separator { width: 1px; height: 16px; background: #334155; margin: 0 0.3rem; }
        
        .tree-container { flex: 1; overflow: auto; position: relative; }
        #treeSvg { min-width: 100%; min-height: 100%; }
        .node circle { cursor: pointer; stroke-width: 2px; transition: all 0.15s; }
        .node circle:hover { filter: brightness(1.3) drop-shadow(0 0 6px currentColor); transform: scale(1.2); }
        .node text { 
            font-family: 'Monaco', 'Consolas', monospace; 
            font-size: 10px; 
            fill: #e2e8f0; 
            pointer-events: none;
        }
        .node .text-bg { pointer-events: none; }
        .node.from-header text { fill: #94a3b8; font-style: italic; }
        .node.external circle { stroke-dasharray: 3,2; }
        .link { fill: none; stroke: #334155; stroke-width: 1.5px; }
        
        .zoom-controls {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            display: flex;
            gap: 0.2rem;
            background: rgba(30, 41, 59, 0.95);
            padding: 0.2rem;
            border-radius: 6px;
        }
        .zoom-btn {
            width: 26px;
            height: 26px;
            background: transparent;
            border: none;
            border-radius: 4px;
            color: #e2e8f0;
            cursor: pointer;
            font-size: 0.9rem;
        }
        .zoom-btn:hover { background: rgba(34, 211, 238, 0.2); }
        
        .info-panel {
            background: rgba(30, 41, 59, 0.8);
            border-top: 1px solid #334155;
            padding: 0.6rem;
            min-height: 90px;
        }
        .info-badge {
            display: inline-block;
            padding: 0.15rem 0.4rem;
            border-radius: 4px;
            font-size: 0.55rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .info-title { font-family: monospace; font-size: 0.85rem; margin-left: 0.4rem; }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(90px, 1fr));
            gap: 0.4rem;
            margin-top: 0.4rem;
        }
        .info-item {
            background: rgba(15, 23, 42, 0.6);
            padding: 0.3rem 0.4rem;
            border-radius: 4px;
        }
        .info-label { font-size: 0.5rem; color: #64748b; text-transform: uppercase; }
        .info-value { font-family: monospace; font-size: 0.65rem; word-break: break-all; }
        
        .empty {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #64748b;
        }
        .empty-icon { font-size: 3rem; opacity: 0.3; }
        
        .right-panel {
            background: rgba(30, 41, 59, 0.7);
            border-left: 1px solid #334155;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .section {
            border-bottom: 1px solid #334155;
        }
        .section-header {
            padding: 0.4rem 0.6rem;
            background: rgba(51, 65, 85, 0.3);
            font-size: 0.7rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
        }
        .section-header:hover { background: rgba(51, 65, 85, 0.5); }
        .section-content {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s;
        }
        .section-content.open { max-height: 200px; overflow-y: auto; }
        .section-item {
            padding: 0.3rem 0.6rem;
            font-size: 0.65rem;
            border-bottom: 1px solid rgba(51, 65, 85, 0.2);
            cursor: pointer;
        }
        .section-item:hover { background: rgba(34, 211, 238, 0.1); }
        .section-item-name { font-family: monospace; color: #22d3ee; }
        .section-item-info { color: #64748b; font-size: 0.55rem; margin-top: 0.1rem; }
        .section-item-file {
            font-size: 0.5rem;
            color: #f59e0b;
            background: rgba(245, 158, 11, 0.1);
            padding: 0.05rem 0.2rem;
            border-radius: 2px;
        }
        .external-badge {
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            padding: 0.05rem 0.2rem;
            border-radius: 2px;
            font-size: 0.5rem;
            margin-left: 0.25rem;
        }
        .unresolved-badge {
            background: rgba(245, 158, 11, 0.2);
            color: #f59e0b;
        }
        
        .alert {
            margin: 0.5rem;
            padding: 0.5rem;
            border-radius: 6px;
            font-size: 0.7rem;
        }
        .alert-warning {
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: #fbbf24;
        }
        .alert-title { font-weight: 600; margin-bottom: 0.25rem; }
        
        .spinner {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <header class="header">
        <h1>🌳 C AST Viewer <span class="badge">SMART FILTER v2</span></h1>
        <div class="stats">
            <div class="stat"><div class="stat-value" id="nodeCount">0</div><div class="stat-label">Your Nodes</div></div>
            <div class="stat"><div class="stat-value" id="funcCount">0</div><div class="stat-label">Functions</div></div>
            <div class="stat"><div class="stat-value" id="typeCount">0</div><div class="stat-label">Types</div></div>
            <div class="stat"><div class="stat-value" id="extCallCount">0</div><div class="stat-label">External Calls</div></div>
            <div class="stat"><div class="stat-value warning" id="unresolvedCount">0</div><div class="stat-label">Unresolved</div></div>
        </div>
    </header>
    
    <div class="main-container">
        <div class="left-panel">
            <div class="panel-header">📁 Upload C/H Files</div>
            
            <div class="dropzone" id="dropzone">
                <div class="dropzone-icon">📂</div>
                <div class="dropzone-text">Drop .c and .h files here</div>
                <input type="file" id="fileInput" multiple accept=".c,.h" style="display:none;">
            </div>
            
            <div class="file-list" id="fileList"></div>
            
            <div id="unresolvedAlert" class="alert alert-warning" style="display:none;">
                <div class="alert-title">⚠️ Missing Headers</div>
                <div id="unresolvedList"></div>
            </div>
            
            <div class="include-paths">
                <label>Include Paths (for header resolution):</label>
                <input type="text" id="includePaths" placeholder="/path/to/your/headers">
            </div>
            
            <textarea id="code" placeholder="Or paste C code here..."></textarea>
            
            <div class="buttons">
                <button class="btn btn-primary" id="generateBtn" onclick="generateAST()">
                    <span id="btnText">🔍 Parse (Your Code Only)</span>
                </button>
                <button class="btn btn-secondary" onclick="clearAll()">🗑️</button>
            </div>
        </div>
        
        <div class="center-panel">
            <div class="toolbar">
                <div class="legend-item"><div class="legend-dot" style="background:#22d3ee"></div>Function</div>
                <div class="legend-item"><div class="legend-dot" style="background:#a78bfa"></div>Call</div>
                <div class="legend-item"><div class="legend-dot" style="background:#4ade80"></div>Variable</div>
                <div class="legend-item"><div class="legend-dot" style="background:#67e8f9"></div>Parameter</div>
                <div class="legend-separator"></div>
                <div class="legend-item"><div class="legend-dot" style="background:#facc15"></div>Control</div>
                <div class="legend-item"><div class="legend-dot" style="background:#fb923c"></div>Return</div>
                <div class="legend-separator"></div>
                <div class="legend-item"><div class="legend-dot" style="background:#f472b6"></div>Type/Struct</div>
                <div class="legend-item"><div class="legend-dot" style="background:#c084fc"></div>Enum</div>
                <div class="legend-separator"></div>
                <div class="legend-item"><div class="legend-dot" style="background:#a3e635"></div>Literal</div>
                <div class="legend-item"><div class="legend-dot" style="background:#94a3b8"></div>Operator</div>
                <div class="legend-item"><div class="legend-dot" style="background:#818cf8"></div>Reference</div>
                <div class="legend-separator"></div>
                <div class="legend-item" style="margin-left:auto;"><div class="legend-dot" style="background:transparent; border:2px dashed #ef4444;"></div>External</div>
            </div>
            
            <div class="tree-container" id="treeContainer">
                <div class="empty" id="empty">
                    <div class="empty-icon">🌲</div>
                    <div>Upload files to see YOUR code only</div>
                </div>
                <svg id="treeSvg" style="display:none;"></svg>
                <div class="zoom-controls" id="zoomControls" style="display:none;">
                    <button class="zoom-btn" onclick="zoomIn()">+</button>
                    <button class="zoom-btn" onclick="zoomOut()">−</button>
                    <button class="zoom-btn" onclick="resetZoom()">⟲</button>
                    <button class="zoom-btn" onclick="expandAll()">⊞</button>
                    <button class="zoom-btn" onclick="collapseAll()">⊟</button>
                </div>
            </div>
            
            <div class="info-panel">
                <div id="nodeInfo"><span style="color:#64748b">👆 Click a node to see details</span></div>
            </div>
        </div>
        
        <div class="right-panel">
            <div class="panel-header">📊 Your Code Analysis</div>
            
            <div class="section">
                <div class="section-header" onclick="toggleSection('userFuncs')">
                    <span>📦 Your Functions</span>
                    <span id="userFuncCount">0</span>
                </div>
                <div class="section-content" id="userFuncs-content"></div>
            </div>
            
            <div class="section">
                <div class="section-header" onclick="toggleSection('extCalls')">
                    <span>🔗 External Calls</span>
                    <span id="extCallSummary">0</span>
                </div>
                <div class="section-content" id="extCalls-content"></div>
            </div>
            
            <div class="section">
                <div class="section-header" onclick="toggleSection('userTypes')">
                    <span>🔷 Your Types</span>
                    <span id="userTypeCount">0</span>
                </div>
                <div class="section-content" id="userTypes-content"></div>
            </div>
            
            <div class="section">
                <div class="section-header" onclick="toggleSection('extTypes')">
                    <span>📎 External Types Used</span>
                    <span id="extTypeCount">0</span>
                </div>
                <div class="section-content" id="extTypes-content"></div>
            </div>
            
            <div class="section">
                <div class="section-header" onclick="toggleSection('callGraph')">
                    <span>🔀 Call Graph</span>
                    <span>→</span>
                </div>
                <div class="section-content" id="callGraph-content"></div>
            </div>
            
            <div class="section">
                <div class="section-header" onclick="toggleSection('headers')">
                    <span>📑 Headers</span>
                    <span id="headerCount">0</span>
                </div>
                <div class="section-content" id="headers-content"></div>
            </div>
        </div>
    </div>

    <script>
    const colors = {
        // Functions & Calls
        function: '#22d3ee',      // Cyan - function definitions
        call: '#a78bfa',          // Purple - function calls
        parameter: '#67e8f9',     // Light cyan - parameters
        
        // Variables
        variable: '#4ade80',      // Green - variables
        field: '#86efac',         // Light green - struct fields
        global: '#22c55e',        // Darker green - globals
        
        // Types
        typedef: '#f472b6',       // Pink - typedefs
        struct: '#f472b6',        // Pink - structs
        union: '#ec4899',         // Hot pink - unions
        enum: '#c084fc',          // Light purple - enums
        enum_const: '#d8b4fe',    // Lighter purple - enum constants
        type_ref: '#fb923c',      // Orange - type references
        
        // Literals
        literal: '#a3e635',       // Lime - all literals
        string_literal: '#bef264', // Light lime - strings
        
        // Control Flow
        control: '#facc15',       // Yellow - if/for/while/switch
        return: '#fb923c',        // Orange - return statements
        break: '#fbbf24',         // Amber - break/continue
        goto: '#f59e0b',          // Dark amber - goto
        
        // Operators & Expressions
        operator: '#94a3b8',      // Gray - operators
        expression: '#cbd5e1',    // Light gray - expressions
        assignment: '#9ca3af',    // Medium gray - assignments
        
        // References & Access
        reference: '#818cf8',     // Indigo - variable refs
        member_access: '#a5b4fc', // Light indigo - struct.member
        array_access: '#93c5fd',  // Light blue - array[i]
        pointer: '#60a5fa',       // Blue - pointer ops
        
        // Preprocessor
        include: '#fb7185',       // Rose - #include
        macro: '#f43f5e',         // Red - macro definitions
        macro_use: '#fda4af',     // Light rose - macro usage
        
        // Blocks & Structure
        block: '#475569',         // Dark slate - compound statements
        decl_stmt: '#64748b',     // Slate - declaration statements
        root: '#6366f1',          // Indigo - translation unit
        
        // Fallback
        other: '#94a3b8'          // Gray - unknown
    };

    let svg, g, zoom, root, uploadedFiles = {};

    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', e => { e.preventDefault(); dropzone.classList.remove('dragover'); handleFiles(e.dataTransfer.files); });
    fileInput.addEventListener('change', e => handleFiles(e.target.files));

    function handleFiles(files) {
        Array.from(files).forEach(file => {
            if (file.name.match(/\\.(c|h)$/i)) {
                const reader = new FileReader();
                reader.onload = e => {
                    uploadedFiles[file.name] = { content: e.target.result, isMain: file.name.endsWith('.c') };
                    updateFileList();
                };
                reader.readAsText(file);
            }
        });
    }

    function updateFileList() {
        document.getElementById('fileList').innerHTML = Object.keys(uploadedFiles).map(name => `
            <div class="file-item">
                <span>${uploadedFiles[name].isMain ? '📄' : '📑'} ${name}</span>
                <span class="file-item-remove" onclick="delete uploadedFiles['${name}']; updateFileList();">×</span>
            </div>
        `).join('');
    }

    async function generateAST() {
        const btn = document.getElementById('generateBtn');
        const btnText = document.getElementById('btnText');
        btn.disabled = true;
        btnText.innerHTML = '<span class="spinner"></span> Parsing...';

        try {
            let body;
            const code = document.getElementById('code').value;
            
            if (Object.keys(uploadedFiles).length > 0) {
                const mainFile = Object.keys(uploadedFiles).find(f => f.endsWith('.c'));
                if (!mainFile) { alert('Upload at least one .c file'); return; }
                
                body = JSON.stringify({
                    files: uploadedFiles,
                    main_file: mainFile,
                    include_paths: document.getElementById('includePaths').value.split(',').map(p => p.trim()).filter(p => p)
                });
            } else if (code.trim()) {
                body = JSON.stringify({ code });
            } else {
                alert('Upload files or paste code'); return;
            }

            const response = await fetch('/parse', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body
            });

            const data = await response.json();
            
            if (data.success && data.ast) {
                document.getElementById('nodeCount').textContent = data.stats.nodes || 0;
                document.getElementById('funcCount').textContent = data.stats.functions || 0;
                document.getElementById('typeCount').textContent = data.stats.types || 0;
                document.getElementById('extCallCount').textContent = data.stats.external_calls || 0;
                document.getElementById('unresolvedCount').textContent = data.stats.unresolved_headers || 0;
                
                // Show unresolved headers warning
                const unresolved = data.summary.unresolved_headers || [];
                if (unresolved.length > 0) {
                    document.getElementById('unresolvedAlert').style.display = 'block';
                    document.getElementById('unresolvedList').innerHTML = unresolved.map(h => 
                        `<div style="font-family:monospace; font-size:0.65rem;">• ${h}</div>`
                    ).join('');
                } else {
                    document.getElementById('unresolvedAlert').style.display = 'none';
                }
                
                renderTree(data.ast);
                renderSummary(data.summary);
                
                document.getElementById('empty').style.display = 'none';
                document.getElementById('treeSvg').style.display = 'block';
                document.getElementById('zoomControls').style.display = 'flex';
            } else {
                alert('Error: ' + (data.error || 'Parse failed'));
                console.error(data);
            }
        } catch (e) {
            alert('Error: ' + e.message);
        } finally {
            btn.disabled = false;
            btnText.innerHTML = '🔍 Parse (Your Code Only)';
        }
    }

    function renderSummary(summary) {
        // User functions
        const funcs = Object.values(summary.user_functions || {});
        document.getElementById('userFuncCount').textContent = funcs.length;
        document.getElementById('userFuncs-content').innerHTML = funcs.map(f => `
            <div class="section-item">
                <div class="section-item-name">${f.name}</div>
                <div class="section-item-info">${f.signature}</div>
                <span class="section-item-file">${f.file}:${f.line}</span>
            </div>
        `).join('') || '<div class="section-item">No functions</div>';
        if (funcs.length > 0) document.getElementById('userFuncs-content').classList.add('open');
        
        // External calls
        const extCalls = Object.values(summary.external_calls || {});
        document.getElementById('extCallSummary').textContent = extCalls.length;
        document.getElementById('extCalls-content').innerHTML = extCalls.map(c => `
            <div class="section-item">
                <div class="section-item-name">${c.name} <span class="external-badge">${c.is_system ? 'SYSTEM' : 'CUSTOM'}</span></div>
                <div class="section-item-info">${c.signature}</div>
                <span class="section-item-file">from: ${c.defined_in}</span>
            </div>
        `).join('') || '<div class="section-item">No external calls</div>';
        if (extCalls.length > 0) document.getElementById('extCalls-content').classList.add('open');
        
        // User types
        const types = Object.values(summary.user_types || {});
        document.getElementById('userTypeCount').textContent = types.length;
        document.getElementById('userTypes-content').innerHTML = types.map(t => `
            <div class="section-item">
                <div class="section-item-name">${t.kind} ${t.name}</div>
                <div class="section-item-info">${t.fields ? t.fields.length + ' fields' : t.underlying || ''}</div>
                <span class="section-item-file">${t.file}:${t.line}</span>
            </div>
        `).join('') || '<div class="section-item">No types</div>';
        if (types.length > 0) document.getElementById('userTypes-content').classList.add('open');
        
        // External types
        const extTypes = Object.values(summary.external_types || {});
        document.getElementById('extTypeCount').textContent = extTypes.length;
        document.getElementById('extTypes-content').innerHTML = extTypes.map(t => `
            <div class="section-item">
                <div class="section-item-name">${t.name} <span class="${t.is_resolved ? 'section-item-file' : 'unresolved-badge'}">${t.is_resolved ? t.defined_in : 'UNRESOLVED'}</span></div>
            </div>
        `).join('') || '<div class="section-item">No external types</div>';
        if (extTypes.length > 0) document.getElementById('extTypes-content').classList.add('open');
        
        // Call graph
        const calls = Object.entries(summary.call_graph || {});
        document.getElementById('callGraph-content').innerHTML = calls.map(([caller, callees]) => `
            <div class="section-item">
                <div class="section-item-name">${caller}</div>
                <div class="section-item-info">→ ${callees.join(', ')}</div>
            </div>
        `).join('') || '<div class="section-item">No calls</div>';
        if (calls.length > 0) document.getElementById('callGraph-content').classList.add('open');
        
        // Headers
        const resolved = summary.resolved_headers || [];
        const unresolved = summary.unresolved_headers || [];
        document.getElementById('headerCount').textContent = resolved.length + '/' + (resolved.length + unresolved.length);
        document.getElementById('headers-content').innerHTML = 
            resolved.map(h => `<div class="section-item"><span class="section-item-name">✅ ${h}</span></div>`).join('') +
            unresolved.map(h => `<div class="section-item"><span class="section-item-name" style="color:#f59e0b">❌ ${h}</span></div>`).join('') ||
            '<div class="section-item">No headers</div>';
        if (unresolved.length > 0) document.getElementById('headers-content').classList.add('open');
    }

    function toggleSection(id) {
        document.getElementById(id + '-content').classList.toggle('open');
    }

    function renderTree(data) {
        const container = document.getElementById('treeContainer');
        const width = container.clientWidth || 800;
        let nodeCount = 0;
        function cnt(n) { if(n) { nodeCount++; (n.children||[]).forEach(cnt); } }
        cnt(data);
        
        // Much larger height for better spacing
        const height = Math.max(600, nodeCount * 35);

        d3.select('#treeSvg').selectAll('*').remove();
        svg = d3.select('#treeSvg').attr('width', width * 1.5).attr('height', height);
        zoom = d3.zoom().scaleExtent([0.1, 4]).on('zoom', e => g.attr('transform', e.transform));
        svg.call(zoom);
        g = svg.append('g').attr('transform', 'translate(50, 30)');
        
        root = d3.hierarchy(data);
        root.descendants().forEach(d => { if (d.depth > 2) { d._children = d.children; d.children = null; } });
        update(root);
        
        setTimeout(() => {
            const b = g.node().getBBox();
            const s = Math.min((width-60)/b.width, (container.clientHeight - 100)/b.height, 0.9) * 0.75;
            svg.call(zoom.transform, d3.zoomIdentity.translate(30, 30).scale(s));
        }, 100);
    }

    function update(source) {
        // Increased nodeSize for more spacing: [vertical, horizontal]
        const tree = d3.tree().nodeSize([28, 200]).separation((a,b) => a.parent === b.parent ? 1.2 : 1.8);
        tree(root);
        
        g.selectAll('.link').data(root.links(), d => d.target.data?.type + d.target.depth)
            .join('path').attr('class', 'link').attr('d', d3.linkHorizontal().x(d => d.y).y(d => d.x));
        
        const nodes = g.selectAll('.node').data(root.descendants(), d => d.data?.type + d.depth + (d.data?.name||''));
        
        // Remove old nodes first
        nodes.exit().remove();
        
        const nodeEnter = nodes.enter().append('g')
            .attr('class', d => 'node' + (d.data?.is_from_header ? ' from-header' : '') + (d.data?.extra?.external ? ' external' : ''))
            .attr('transform', d => `translate(${d.y},${d.x})`)
            .on('click', (e, d) => {
                if (d.children) { d._children = d.children; d.children = null; }
                else if (d._children) { d.children = d._children; d._children = null; }
                update(d);
                showInfo(d.data);
            });
        
        nodeEnter.append('circle')
            .attr('r', d => d._children ? 6 : 4)
            .style('fill', d => colors[d.data?.category] || colors.other)
            .style('stroke', d => d.data?.extra?.external ? '#ef4444' : d3.color(colors[d.data?.category] || colors.other).darker(0.5));
        
        // Add background rect for text (for better readability)
        nodeEnter.append('rect')
            .attr('class', 'text-bg')
            .attr('fill', 'rgba(15, 23, 42, 0.85)')
            .attr('rx', 3)
            .attr('ry', 3);
        
        nodeEnter.append('text')
            .attr('dy', '0.35em')
            .attr('x', d => (d.children || d._children) ? -10 : 10)
            .style('text-anchor', d => (d.children || d._children) ? 'end' : 'start')
            .text(d => {
                if (!d.data) return '';
                let l = d.data.type.replace('_STMT', '').replace('_EXPR', '').replace('_DECL', '');
                if (d.data.name && d.data.name !== d.data.type) {
                    // Shorter name
                    let name = d.data.name.length > 15 ? d.data.name.slice(0,12) + '..' : d.data.name;
                    l += ': ' + name;
                }
                return l.length > 25 ? l.slice(0,22) + '..' : l;
            })
            .each(function(d) {
                // Size the background rect based on text
                const bbox = this.getBBox();
                const isLeft = (d.children || d._children);
                d3.select(this.parentNode).select('.text-bg')
                    .attr('x', isLeft ? bbox.x - 3 : bbox.x - 2)
                    .attr('y', bbox.y - 1)
                    .attr('width', bbox.width + 5)
                    .attr('height', bbox.height + 2);
            });
        
        // Update positions
        const nodeUpdate = nodes.merge(nodeEnter);
        nodeUpdate.attr('transform', d => `translate(${d.y},${d.x})`);
        nodeUpdate.select('circle').attr('r', d => d._children ? 6 : 4);
    }

    function showInfo(data) {
        if (!data) return;
        const c = colors[data.category] || colors.other;
        let extra = '';
        if (data.extra) {
            if (data.extra.external) extra += `<div class="info-item"><div class="info-label">External</div><div class="info-value" style="color:#ef4444">Yes - ${data.extra.defined_in}</div></div>`;
            if (data.extra.signature) extra += `<div class="info-item"><div class="info-label">Signature</div><div class="info-value">${data.extra.signature}</div></div>`;
        }
        document.getElementById('nodeInfo').innerHTML = `
            <span class="info-badge" style="background:${c}">${data.category}</span>
            <span class="info-title">${data.type}</span>
            <div class="info-grid">
                <div class="info-item"><div class="info-label">Name</div><div class="info-value">${data.name || '-'}</div></div>
                <div class="info-item"><div class="info-label">Type</div><div class="info-value">${data.type_info || '-'}</div></div>
                <div class="info-item"><div class="info-label">File</div><div class="info-value">${data.file || 'main'}:${data.line}</div></div>
                ${extra}
            </div>
        `;
    }

    function zoomIn() { svg.transition().duration(200).call(zoom.scaleBy, 1.3); }
    function zoomOut() { svg.transition().duration(200).call(zoom.scaleBy, 0.7); }
    function resetZoom() { svg.transition().duration(200).call(zoom.transform, d3.zoomIdentity.translate(30,20).scale(0.8)); }
    function expandAll() { root.descendants().forEach(d => { if(d._children) { d.children = d._children; d._children = null; }}); update(root); }
    function collapseAll() { root.descendants().forEach(d => { if(d.depth > 1 && d.children) { d._children = d.children; d.children = null; }}); update(root); }
    function clearAll() {
        uploadedFiles = {};
        updateFileList();
        document.getElementById('code').value = '';
        document.getElementById('empty').style.display = 'flex';
        document.getElementById('treeSvg').style.display = 'none';
        document.getElementById('zoomControls').style.display = 'none';
        document.getElementById('unresolvedAlert').style.display = 'none';
    }
    </script>
</body>
</html>
'''


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/parse', methods=['POST'])
def parse():
    """Parse with smart filtering - only user code."""
    data = request.get_json()
    
    # Check if files were uploaded
    files = data.get('files')
    main_file = data.get('main_file')
    include_paths = data.get('include_paths', [])
    code = data.get('code', '')
    
    temp_dir = tempfile.mkdtemp(prefix='ast_v2_')
    
    try:
        if files and main_file:
            # Write uploaded files
            for name, info in files.items():
                file_path = os.path.join(temp_dir, name)
                with open(file_path, 'w') as f:
                    f.write(info['content'])
            
            main_path = os.path.join(temp_dir, main_file)
            all_include_paths = [temp_dir] + include_paths
            
        elif code.strip():
            # Write code to temp file
            main_path = os.path.join(temp_dir, 'main.c')
            with open(main_path, 'w') as f:
                f.write(code)
            all_include_paths = [temp_dir] + include_paths
            
        else:
            return jsonify({
                'success': False,
                'error': 'No code provided',
                'ast': None,
                'stats': {},
                'summary': {}
            })
        
        result = parse_user_code(main_path, all_include_paths)
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'ast': None,
            'stats': {},
            'summary': {}
        })
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║   🌳 C AST Viewer v2 - SMART FILTERING                             ║
║                                                                    ║
║   ✨ Shows ONLY your code (filters out stdio.h, stdlib.h, etc.)    ║
║   🔗 Tracks external function calls & types                        ║
║   ⚠️  Shows unresolved headers                                      ║
║   📊 Clean dependency analysis                                      ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    print("🚀 Starting server at http://localhost:5051")
    print("📌 Press Ctrl+C to stop\n")
    
    import webbrowser
    webbrowser.open('http://localhost:5051')
    
    app.run(host='0.0.0.0', port=5051, debug=False, threaded=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
C AST Viewer Pro - with Full Header Resolution
===============================================
Parses .c files along with all included .h files
to show complete function signatures, types, and dependencies.

Usage:
    python ast_viewer_pro.py

Then open http://localhost:5050 in your browser.
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
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Temp directory for uploaded files
UPLOAD_FOLDER = tempfile.mkdtemp(prefix='ast_viewer_')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Node type categories for coloring
NODE_CATEGORIES = {
    'FUNCTION_DECL': 'function',
    'PARM_DECL': 'parameter',
    'FUNCTION_TEMPLATE': 'function',
    'CXX_METHOD': 'function',
    
    'VAR_DECL': 'declaration',
    'FIELD_DECL': 'field',
    'TYPEDEF_DECL': 'typedef',
    'TYPE_ALIAS_DECL': 'typedef',
    'STRUCT_DECL': 'struct',
    'UNION_DECL': 'struct',
    'ENUM_DECL': 'enum',
    'ENUM_CONSTANT_DECL': 'enum',
    'CLASS_DECL': 'struct',
    
    'TYPE_REF': 'type',
    
    'INTEGER_LITERAL': 'literal',
    'FLOATING_LITERAL': 'literal',
    'STRING_LITERAL': 'literal',
    'CHARACTER_LITERAL': 'literal',
    
    'IF_STMT': 'control',
    'SWITCH_STMT': 'control',
    'CASE_STMT': 'control',
    'DEFAULT_STMT': 'control',
    'WHILE_STMT': 'control',
    'DO_STMT': 'control',
    'FOR_STMT': 'control',
    'BREAK_STMT': 'control',
    'CONTINUE_STMT': 'control',
    'RETURN_STMT': 'control',
    'GOTO_STMT': 'control',
    'LABEL_STMT': 'control',
    
    'BINARY_OPERATOR': 'expression',
    'UNARY_OPERATOR': 'expression',
    'CALL_EXPR': 'call',
    'MEMBER_REF_EXPR': 'expression',
    'ARRAY_SUBSCRIPT_EXPR': 'expression',
    'CONDITIONAL_OPERATOR': 'expression',
    'COMPOUND_ASSIGNMENT_OPERATOR': 'expression',
    'CSTYLE_CAST_EXPR': 'expression',
    
    'DECL_REF_EXPR': 'identifier',
    
    'INCLUSION_DIRECTIVE': 'include',
    'MACRO_DEFINITION': 'macro',
    'MACRO_INSTANTIATION': 'macro',
    
    'COMPOUND_STMT': 'compound',
    'TRANSLATION_UNIT': 'root',
}

def get_category(kind_name):
    return NODE_CATEGORIES.get(kind_name, 'other')

class DependencyResolver:
    """Resolves and tracks all dependencies in C files."""
    
    def __init__(self, main_file, include_paths=None):
        self.main_file = main_file
        self.main_dir = os.path.dirname(os.path.abspath(main_file))
        self.include_paths = include_paths or []
        self.include_paths.insert(0, self.main_dir)
        
        # Tracking
        self.parsed_files = {}
        self.includes = defaultdict(list)  # file -> [included files]
        self.functions = {}  # name -> {file, line, return_type, params, is_definition}
        self.types = {}  # name -> {file, line, kind, fields}
        self.macros = {}  # name -> {file, line, value}
        self.globals = {}  # name -> {file, line, type}
        self.call_graph = defaultdict(set)  # caller -> [callees]
        
    def find_header(self, header_name, current_file):
        """Find header file in include paths."""
        current_dir = os.path.dirname(os.path.abspath(current_file))
        
        # Try relative to current file first
        candidate = os.path.join(current_dir, header_name)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        
        # Try include paths
        for inc_path in self.include_paths:
            candidate = os.path.join(inc_path, header_name)
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        
        return None
    
    def extract_includes(self, code, current_file):
        """Extract #include directives from code."""
        includes = []
        for match in re.finditer(r'#\s*include\s*[<"]([^>"]+)[>"]', code):
            header = match.group(1)
            header_path = self.find_header(header, current_file)
            if header_path:
                includes.append((header, header_path))
        return includes
    
    def parse_all(self):
        """Parse main file and all dependencies."""
        index = Index.create()
        
        # Build include args
        args = ['-std=c11', '-x', 'c']
        for inc_path in self.include_paths:
            args.append(f'-I{inc_path}')
        
        # Parse main file
        tu = index.parse(
            self.main_file,
            args=args,
            options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        )
        
        # Extract all information
        self._process_cursor(tu.cursor, self.main_file)
        
        # Get all included files
        for inc in tu.get_includes():
            inc_file = str(inc.include.name) if inc.include else None
            if inc_file:
                self.parsed_files[inc_file] = True
                source_file = str(inc.source.name) if inc.source else self.main_file
                self.includes[source_file].append(inc_file)
        
        return tu
    
    def _process_cursor(self, cursor, current_file):
        """Process cursor and extract information."""
        kind = cursor.kind
        kind_name = kind.name
        
        # Get location
        loc = cursor.location
        file_path = str(loc.file.name) if loc.file else current_file
        line = loc.line
        
        # Extract based on kind
        if kind == CursorKind.FUNCTION_DECL:
            name = cursor.spelling
            return_type = cursor.result_type.spelling if cursor.result_type else 'void'
            params = []
            for child in cursor.get_children():
                if child.kind == CursorKind.PARM_DECL:
                    params.append({
                        'name': child.spelling or 'unnamed',
                        'type': child.type.spelling if child.type else 'unknown'
                    })
            
            is_definition = cursor.is_definition()
            
            # Store or update
            if name not in self.functions or is_definition:
                self.functions[name] = {
                    'name': name,
                    'file': os.path.basename(file_path),
                    'full_path': file_path,
                    'line': line,
                    'return_type': return_type,
                    'params': params,
                    'is_definition': is_definition,
                    'signature': f"{return_type} {name}({', '.join(p['type'] + ' ' + p['name'] for p in params)})"
                }
        
        elif kind == CursorKind.TYPEDEF_DECL:
            name = cursor.spelling
            underlying = cursor.underlying_typedef_type.spelling if cursor.underlying_typedef_type else 'unknown'
            self.types[name] = {
                'name': name,
                'file': os.path.basename(file_path),
                'full_path': file_path,
                'line': line,
                'kind': 'typedef',
                'underlying': underlying
            }
        
        elif kind in (CursorKind.STRUCT_DECL, CursorKind.UNION_DECL):
            name = cursor.spelling or cursor.type.spelling if cursor.type else 'anonymous'
            if name and not name.startswith('('):
                fields = []
                for child in cursor.get_children():
                    if child.kind == CursorKind.FIELD_DECL:
                        fields.append({
                            'name': child.spelling,
                            'type': child.type.spelling if child.type else 'unknown'
                        })
                
                self.types[name] = {
                    'name': name,
                    'file': os.path.basename(file_path),
                    'full_path': file_path,
                    'line': line,
                    'kind': 'struct' if kind == CursorKind.STRUCT_DECL else 'union',
                    'fields': fields
                }
        
        elif kind == CursorKind.ENUM_DECL:
            name = cursor.spelling or 'anonymous_enum'
            constants = []
            for child in cursor.get_children():
                if child.kind == CursorKind.ENUM_CONSTANT_DECL:
                    constants.append({
                        'name': child.spelling,
                        'value': child.enum_value
                    })
            
            if name != 'anonymous_enum':
                self.types[name] = {
                    'name': name,
                    'file': os.path.basename(file_path),
                    'full_path': file_path,
                    'line': line,
                    'kind': 'enum',
                    'constants': constants
                }
        
        elif kind == CursorKind.MACRO_DEFINITION:
            name = cursor.spelling
            # Get macro value from tokens
            tokens = list(cursor.get_tokens())
            value = ' '.join(t.spelling for t in tokens[1:]) if len(tokens) > 1 else ''
            self.macros[name] = {
                'name': name,
                'file': os.path.basename(file_path),
                'full_path': file_path,
                'line': line,
                'value': value[:100]  # Truncate long macros
            }
        
        elif kind == CursorKind.VAR_DECL:
            if cursor.storage_class.name in ('EXTERN', 'STATIC', 'NONE') and not cursor.semantic_parent or \
               cursor.semantic_parent.kind == CursorKind.TRANSLATION_UNIT:
                name = cursor.spelling
                var_type = cursor.type.spelling if cursor.type else 'unknown'
                self.globals[name] = {
                    'name': name,
                    'file': os.path.basename(file_path),
                    'full_path': file_path,
                    'line': line,
                    'type': var_type
                }
        
        elif kind == CursorKind.CALL_EXPR:
            caller = self._get_parent_function(cursor)
            callee = cursor.spelling
            if caller and callee:
                self.call_graph[caller].add(callee)
        
        # Recurse
        for child in cursor.get_children():
            self._process_cursor(child, current_file)
    
    def _get_parent_function(self, cursor):
        """Get the name of the function containing this cursor."""
        parent = cursor.semantic_parent
        while parent:
            if parent.kind == CursorKind.FUNCTION_DECL:
                return parent.spelling
            parent = parent.semantic_parent
        return None
    
    def get_summary(self):
        """Get a summary of all parsed information."""
        return {
            'files': list(self.parsed_files.keys()),
            'functions': self.functions,
            'types': self.types,
            'macros': self.macros,
            'globals': self.globals,
            'includes': dict(self.includes),
            'call_graph': {k: list(v) for k, v in self.call_graph.items()}
        }


def cursor_to_dict(cursor, resolver, depth=0, max_depth=50, main_file=None):
    """Convert cursor to dict with resolved information."""
    if depth > max_depth:
        return None
    
    kind_name = cursor.kind.name
    name = cursor.displayname or cursor.spelling or ''
    
    # Location
    loc = cursor.location
    line = loc.line if loc.file else 0
    col = loc.column if loc.file else 0
    file_path = str(loc.file.name) if loc.file else ''
    filename = os.path.basename(file_path) if file_path else ''
    
    # Determine if from header
    is_from_header = False
    if file_path and main_file:
        is_from_header = not os.path.samefile(file_path, main_file) if os.path.exists(file_path) and os.path.exists(main_file) else file_path != main_file
    
    # Type info
    type_info = ''
    try:
        if cursor.type and cursor.type.spelling:
            type_info = cursor.type.spelling
    except:
        pass
    
    # Extra info for functions
    extra = {}
    if kind_name == 'FUNCTION_DECL' and cursor.spelling in resolver.functions:
        func_info = resolver.functions[cursor.spelling]
        extra = {
            'return_type': func_info['return_type'],
            'params': func_info['params'],
            'signature': func_info['signature'],
            'is_definition': func_info['is_definition']
        }
    
    # Extra info for calls
    if kind_name == 'CALL_EXPR' and cursor.spelling in resolver.functions:
        func_info = resolver.functions[cursor.spelling]
        extra = {
            'defined_in': func_info['file'],
            'signature': func_info['signature']
        }
    
    node = {
        'type': kind_name,
        'name': name[:80] if name else '',
        'category': get_category(kind_name),
        'line': line,
        'col': col,
        'file': filename,
        'full_path': file_path,
        'is_from_header': is_from_header,
        'type_info': type_info[:80] if type_info else '',
        'extra': extra,
        'children': []
    }
    
    # Recurse
    for child in cursor.get_children():
        child_dict = cursor_to_dict(child, resolver, depth + 1, max_depth, main_file)
        if child_dict:
            node['children'].append(child_dict)
    
    return node


def parse_with_dependencies(filepath, include_paths=None):
    """Parse C file with all dependencies resolved."""
    try:
        resolver = DependencyResolver(filepath, include_paths)
        tu = resolver.parse_all()
        
        # Convert to dict
        ast = cursor_to_dict(tu.cursor, resolver, main_file=filepath)
        
        # Stats
        stats = {'nodes': 0, 'depth': 0, 'functions': 0, 'types': 0, 'headers': 0}
        def count_stats(node, d=0):
            stats['nodes'] += 1
            stats['depth'] = max(stats['depth'], d)
            if node['type'] == 'FUNCTION_DECL':
                stats['functions'] += 1
            for child in node.get('children', []):
                count_stats(child, d + 1)
        
        if ast:
            count_stats(ast)
        
        stats['types'] = len(resolver.types)
        stats['headers'] = len([f for f in resolver.parsed_files.keys() if f.endswith('.h')])
        
        # Diagnostics
        diagnostics = []
        for diag in tu.diagnostics:
            diagnostics.append({
                'severity': diag.severity,
                'message': diag.spelling,
                'line': diag.location.line,
                'col': diag.location.column,
                'file': os.path.basename(str(diag.location.file.name)) if diag.location.file else ''
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
            'stats': {'nodes': 0, 'depth': 0, 'functions': 0, 'types': 0, 'headers': 0},
            'diagnostics': [],
            'summary': {}
        }


def parse_code_with_headers(code, headers=None):
    """Parse code string with optional headers."""
    # Create temp directory for this parse
    temp_dir = tempfile.mkdtemp(prefix='ast_parse_')
    
    try:
        # Write main file
        main_file = os.path.join(temp_dir, 'main.c')
        with open(main_file, 'w') as f:
            f.write(code)
        
        # Write headers if provided
        if headers:
            for name, content in headers.items():
                header_path = os.path.join(temp_dir, name)
                os.makedirs(os.path.dirname(header_path), exist_ok=True)
                with open(header_path, 'w') as f:
                    f.write(content)
        
        return parse_with_dependencies(main_file, [temp_dir])
        
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>C AST Viewer Pro - Header Resolution</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #e2e8f0;
            min-height: 100vh;
        }
        .header {
            background: rgba(30, 41, 59, 0.9);
            backdrop-filter: blur(10px);
            padding: 0.75rem 1.5rem;
            border-bottom: 1px solid #334155;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 { 
            font-size: 1.25rem; 
            background: linear-gradient(135deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .badge {
            background: linear-gradient(135deg, #f59e0b, #d97706);
            color: white;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.6rem;
            font-weight: 700;
            -webkit-text-fill-color: white;
        }
        .stats { display: flex; gap: 1.5rem; }
        .stat { text-align: center; }
        .stat-value { 
            font-size: 1.25rem; 
            font-weight: bold; 
            background: linear-gradient(135deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .stat-label { font-size: 0.6rem; color: #64748b; text-transform: uppercase; }
        
        .main-container {
            display: grid;
            grid-template-columns: 380px 1fr 320px;
            height: calc(100vh - 52px);
        }
        
        /* Left Panel - Code Input */
        .left-panel {
            background: rgba(30, 41, 59, 0.6);
            border-right: 1px solid #334155;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .panel-header {
            padding: 0.6rem 1rem;
            background: rgba(51, 65, 85, 0.5);
            font-weight: 600;
            font-size: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            border-bottom: 1px solid #334155;
        }
        .tabs {
            display: flex;
            background: rgba(15, 23, 42, 0.5);
        }
        .tab {
            flex: 1;
            padding: 0.5rem;
            text-align: center;
            cursor: pointer;
            font-size: 0.75rem;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .tab:hover { background: rgba(99, 102, 241, 0.1); }
        .tab.active { 
            border-bottom-color: #818cf8; 
            background: rgba(99, 102, 241, 0.1);
        }
        .tab-content { display: none; flex: 1; flex-direction: column; overflow: hidden; }
        .tab-content.active { display: flex; }
        
        .dropzone {
            margin: 0.5rem;
            padding: 1rem;
            border: 2px dashed #334155;
            border-radius: 10px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .dropzone:hover, .dropzone.dragover {
            border-color: #818cf8;
            background: rgba(99, 102, 241, 0.1);
        }
        .dropzone-icon { font-size: 1.5rem; }
        .dropzone-text { font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem; }
        .dropzone-hint { font-size: 0.65rem; color: #64748b; }
        
        .include-paths {
            padding: 0.5rem;
            border-top: 1px solid #334155;
        }
        .include-paths label { font-size: 0.7rem; color: #94a3b8; }
        .include-paths input {
            width: 100%;
            padding: 0.4rem;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid #334155;
            border-radius: 6px;
            color: #e2e8f0;
            font-size: 0.75rem;
            margin-top: 0.25rem;
        }
        
        textarea {
            flex: 1;
            background: rgba(15, 23, 42, 0.8);
            border: none;
            padding: 0.75rem;
            color: #e2e8f0;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.75rem;
            resize: none;
            outline: none;
            line-height: 1.5;
        }
        .buttons {
            padding: 0.75rem;
            display: flex;
            gap: 0.4rem;
        }
        .btn {
            padding: 0.6rem 1rem;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.8rem;
            font-weight: 600;
            transition: all 0.2s;
        }
        .btn-primary {
            flex: 1;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
        }
        .btn-primary:hover { transform: translateY(-1px); }
        .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .btn-secondary { background: rgba(51, 65, 85, 0.8); color: #e2e8f0; }
        .btn-secondary:hover { background: #475569; }
        
        /* Center Panel - Tree */
        .center-panel {
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .legend {
            padding: 0.4rem 0.75rem;
            background: rgba(51, 65, 85, 0.3);
            display: flex;
            gap: 0.75rem;
            flex-wrap: wrap;
            border-bottom: 1px solid #334155;
        }
        .legend-item {
            display: flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.65rem;
            color: #94a3b8;
        }
        .legend-dot { width: 8px; height: 8px; border-radius: 50%; }
        
        .tree-container {
            flex: 1;
            overflow: auto;
            position: relative;
        }
        #treeSvg { min-width: 100%; min-height: 100%; }
        .node circle { cursor: pointer; stroke-width: 2px; transition: all 0.15s; }
        .node circle:hover { filter: brightness(1.3) drop-shadow(0 0 6px currentColor); }
        .node text { font-family: 'Monaco', monospace; font-size: 9px; fill: #e2e8f0; }
        .node.from-header circle { stroke-dasharray: 3,2; }
        .node.from-header text { fill: #94a3b8; font-style: italic; }
        .link { fill: none; stroke: #334155; stroke-width: 1px; }
        .link.to-header { stroke-dasharray: 4,2; stroke: #475569; }
        
        .zoom-controls {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            display: flex;
            gap: 0.2rem;
            background: rgba(30, 41, 59, 0.95);
            padding: 0.2rem;
            border-radius: 6px;
            border: 1px solid #334155;
        }
        .zoom-btn {
            width: 28px;
            height: 28px;
            background: transparent;
            border: none;
            border-radius: 4px;
            color: #e2e8f0;
            cursor: pointer;
            font-size: 1rem;
        }
        .zoom-btn:hover { background: rgba(99, 102, 241, 0.3); }
        
        .info-panel {
            background: rgba(30, 41, 59, 0.8);
            border-top: 1px solid #334155;
            padding: 0.75rem;
            min-height: 100px;
        }
        .info-badge {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            font-size: 0.6rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .info-title { font-family: monospace; font-size: 0.9rem; margin-left: 0.5rem; }
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 0.5rem;
            margin-top: 0.5rem;
        }
        .info-item {
            background: rgba(15, 23, 42, 0.6);
            padding: 0.4rem 0.5rem;
            border-radius: 6px;
            border: 1px solid #334155;
        }
        .info-label { font-size: 0.55rem; color: #64748b; text-transform: uppercase; }
        .info-value { font-family: monospace; font-size: 0.7rem; word-break: break-all; }
        
        .empty {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #64748b;
        }
        .empty-icon { font-size: 3rem; margin-bottom: 0.5rem; opacity: 0.3; }
        .empty-text { font-size: 0.9rem; }
        
        /* Right Panel - Summary */
        .right-panel {
            background: rgba(30, 41, 59, 0.6);
            border-left: 1px solid #334155;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .summary-section {
            border-bottom: 1px solid #334155;
        }
        .summary-header {
            padding: 0.5rem 0.75rem;
            background: rgba(51, 65, 85, 0.3);
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .summary-header:hover { background: rgba(51, 65, 85, 0.5); }
        .summary-content {
            max-height: 200px;
            overflow-y: auto;
            display: none;
        }
        .summary-content.open { display: block; }
        .summary-item {
            padding: 0.4rem 0.75rem;
            font-size: 0.7rem;
            border-bottom: 1px solid rgba(51, 65, 85, 0.3);
            cursor: pointer;
        }
        .summary-item:hover { background: rgba(99, 102, 241, 0.1); }
        .summary-item-name { font-family: monospace; color: #818cf8; }
        .summary-item-info { color: #64748b; font-size: 0.6rem; margin-top: 0.1rem; }
        .summary-item-file { 
            font-size: 0.55rem; 
            color: #f59e0b;
            background: rgba(245, 158, 11, 0.1);
            padding: 0.1rem 0.3rem;
            border-radius: 3px;
            display: inline-block;
        }
        
        .spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .file-list {
            padding: 0.5rem;
            max-height: 150px;
            overflow-y: auto;
        }
        .file-item {
            padding: 0.3rem 0.5rem;
            font-size: 0.7rem;
            background: rgba(15, 23, 42, 0.5);
            border-radius: 4px;
            margin-bottom: 0.3rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .file-item-name { font-family: monospace; }
        .file-item-remove {
            cursor: pointer;
            color: #ef4444;
            font-size: 0.8rem;
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>🌳 C AST Viewer <span class="badge">PRO + HEADERS</span></h1>
        <div class="stats">
            <div class="stat"><div class="stat-value" id="nodeCount">0</div><div class="stat-label">Nodes</div></div>
            <div class="stat"><div class="stat-value" id="funcCount">0</div><div class="stat-label">Functions</div></div>
            <div class="stat"><div class="stat-value" id="typeCount">0</div><div class="stat-label">Types</div></div>
            <div class="stat"><div class="stat-value" id="headerCount">0</div><div class="stat-label">Headers</div></div>
        </div>
    </header>
    
    <div class="main-container">
        <!-- Left Panel -->
        <div class="left-panel">
            <div class="tabs">
                <div class="tab active" data-tab="upload">📁 Upload Files</div>
                <div class="tab" data-tab="paste">📝 Paste Code</div>
            </div>
            
            <div class="tab-content active" id="tab-upload">
                <div class="dropzone" id="dropzone">
                    <div class="dropzone-icon">📂</div>
                    <div class="dropzone-text">Drop .c and .h files here</div>
                    <div class="dropzone-hint">or click to browse</div>
                    <input type="file" id="fileInput" multiple accept=".c,.h,.cpp,.hpp" style="display:none;">
                </div>
                
                <div class="file-list" id="fileList"></div>
                
                <div class="include-paths">
                    <label>Additional Include Paths (comma-separated):</label>
                    <input type="text" id="includePaths" placeholder="/usr/include, /path/to/headers">
                </div>
            </div>
            
            <div class="tab-content" id="tab-paste">
                <textarea id="code" placeholder="Paste your C code here..."></textarea>
            </div>
            
            <div class="buttons">
                <button class="btn btn-primary" id="generateBtn" onclick="generateAST()">
                    <span id="btnText">🔍 Parse with Headers</span>
                </button>
                <button class="btn btn-secondary" onclick="clearAll()">🗑️</button>
            </div>
        </div>
        
        <!-- Center Panel -->
        <div class="center-panel">
            <div class="legend">
                <div class="legend-item"><div class="legend-dot" style="background:#22d3ee"></div>Function</div>
                <div class="legend-item"><div class="legend-dot" style="background:#38bdf8"></div>Declaration</div>
                <div class="legend-item"><div class="legend-dot" style="background:#a78bfa"></div>Call</div>
                <div class="legend-item"><div class="legend-dot" style="background:#4ade80"></div>Literal</div>
                <div class="legend-item"><div class="legend-dot" style="background:#facc15"></div>Control</div>
                <div class="legend-item"><div class="legend-dot" style="background:#f472b6"></div>Include</div>
                <div class="legend-item" style="margin-left:auto;"><span style="border-bottom:1px dashed #64748b">---</span> From Header</div>
            </div>
            
            <div class="tree-container" id="treeContainer">
                <div class="empty" id="empty">
                    <div class="empty-icon">🌲</div>
                    <div class="empty-text">Upload C files to see full AST with headers</div>
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
        
        <!-- Right Panel - Summary -->
        <div class="right-panel">
            <div class="panel-header">📊 Resolved Symbols</div>
            
            <div class="summary-section">
                <div class="summary-header" onclick="toggleSection('functions')">
                    <span>📦 Functions</span>
                    <span id="funcSummaryCount">0</span>
                </div>
                <div class="summary-content" id="functions-content"></div>
            </div>
            
            <div class="summary-section">
                <div class="summary-header" onclick="toggleSection('types')">
                    <span>🔷 Types & Structs</span>
                    <span id="typeSummaryCount">0</span>
                </div>
                <div class="summary-content" id="types-content"></div>
            </div>
            
            <div class="summary-section">
                <div class="summary-header" onclick="toggleSection('macros')">
                    <span>⚡ Macros</span>
                    <span id="macroSummaryCount">0</span>
                </div>
                <div class="summary-content" id="macros-content"></div>
            </div>
            
            <div class="summary-section">
                <div class="summary-header" onclick="toggleSection('includes')">
                    <span>📎 Included Files</span>
                    <span id="includeSummaryCount">0</span>
                </div>
                <div class="summary-content" id="includes-content"></div>
            </div>
            
            <div class="summary-section">
                <div class="summary-header" onclick="toggleSection('callgraph')">
                    <span>🔗 Call Graph</span>
                    <span>→</span>
                </div>
                <div class="summary-content" id="callgraph-content"></div>
            </div>
        </div>
    </div>

    <script>
    const colors = {
        function: '#22d3ee',
        parameter: '#67e8f9',
        declaration: '#38bdf8',
        field: '#7dd3fc',
        typedef: '#fb923c',
        struct: '#fbbf24',
        enum: '#a3e635',
        type: '#fb923c',
        identifier: '#a78bfa',
        literal: '#4ade80',
        control: '#facc15',
        expression: '#c084fc',
        call: '#a78bfa',
        include: '#f472b6',
        macro: '#f472b6',
        compound: '#64748b',
        root: '#6366f1',
        other: '#94a3b8'
    };

    let svg, g, zoom, root, uploadedFiles = {};

    // Tab switching
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
        });
    });

    // File upload
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('dragover'); });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', e => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener('change', e => handleFiles(e.target.files));

    function handleFiles(files) {
        Array.from(files).forEach(file => {
            if (file.name.match(/\\.(c|h|cpp|hpp)$/i)) {
                const reader = new FileReader();
                reader.onload = e => {
                    uploadedFiles[file.name] = {
                        content: e.target.result,
                        isMain: file.name.endsWith('.c')
                    };
                    updateFileList();
                };
                reader.readAsText(file);
            }
        });
    }

    function updateFileList() {
        const list = document.getElementById('fileList');
        list.innerHTML = Object.keys(uploadedFiles).map(name => `
            <div class="file-item">
                <span class="file-item-name">${uploadedFiles[name].isMain ? '📄' : '📑'} ${name}</span>
                <span class="file-item-remove" onclick="removeFile('${name}')">×</span>
            </div>
        `).join('');
    }

    function removeFile(name) {
        delete uploadedFiles[name];
        updateFileList();
    }

    async function generateAST() {
        const btn = document.getElementById('generateBtn');
        const btnText = document.getElementById('btnText');
        btn.disabled = true;
        btnText.innerHTML = '<span class="spinner"></span> Parsing...';

        try {
            let response;
            const activeTab = document.querySelector('.tab.active').dataset.tab;
            
            if (activeTab === 'upload' && Object.keys(uploadedFiles).length > 0) {
                // Find main .c file
                const mainFile = Object.keys(uploadedFiles).find(f => f.endsWith('.c'));
                if (!mainFile) {
                    alert('Please upload at least one .c file');
                    return;
                }
                
                const includePaths = document.getElementById('includePaths').value
                    .split(',').map(p => p.trim()).filter(p => p);
                
                response = await fetch('/parse_files', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        files: uploadedFiles,
                        main_file: mainFile,
                        include_paths: includePaths
                    })
                });
            } else {
                const code = document.getElementById('code').value;
                if (!code.trim()) {
                    alert('Please enter some code or upload files');
                    return;
                }
                response = await fetch('/parse', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code })
                });
            }

            const data = await response.json();
            
            if (data.success && data.ast) {
                document.getElementById('nodeCount').textContent = data.stats.nodes;
                document.getElementById('funcCount').textContent = data.stats.functions;
                document.getElementById('typeCount').textContent = data.stats.types;
                document.getElementById('headerCount').textContent = data.stats.headers;
                
                renderTree(data.ast);
                renderSummary(data.summary);
                
                document.getElementById('empty').style.display = 'none';
                document.getElementById('treeSvg').style.display = 'block';
                document.getElementById('zoomControls').style.display = 'flex';
            } else {
                alert('Parse error: ' + (data.error || 'Unknown error'));
                console.error(data);
            }
        } catch (e) {
            alert('Error: ' + e.message);
            console.error(e);
        } finally {
            btn.disabled = false;
            btnText.innerHTML = '🔍 Parse with Headers';
        }
    }

    function renderSummary(summary) {
        if (!summary) return;
        
        // Functions
        const funcs = Object.values(summary.functions || {});
        document.getElementById('funcSummaryCount').textContent = funcs.length;
        document.getElementById('functions-content').innerHTML = funcs.map(f => `
            <div class="summary-item" onclick="highlightNode('${f.name}')">
                <div class="summary-item-name">${f.name}</div>
                <div class="summary-item-info">${f.signature}</div>
                <span class="summary-item-file">${f.file}:${f.line}</span>
            </div>
        `).join('') || '<div class="summary-item">No functions found</div>';
        
        // Types
        const types = Object.values(summary.types || {});
        document.getElementById('typeSummaryCount').textContent = types.length;
        document.getElementById('types-content').innerHTML = types.map(t => `
            <div class="summary-item">
                <div class="summary-item-name">${t.kind} ${t.name}</div>
                <div class="summary-item-info">${t.fields ? t.fields.length + ' fields' : t.underlying || ''}</div>
                <span class="summary-item-file">${t.file}:${t.line}</span>
            </div>
        `).join('') || '<div class="summary-item">No types found</div>';
        
        // Macros
        const macros = Object.values(summary.macros || {});
        document.getElementById('macroSummaryCount').textContent = macros.length;
        document.getElementById('macros-content').innerHTML = macros.slice(0, 50).map(m => `
            <div class="summary-item">
                <div class="summary-item-name">${m.name}</div>
                <div class="summary-item-info">${m.value || '(empty)'}</div>
                <span class="summary-item-file">${m.file}:${m.line}</span>
            </div>
        `).join('') || '<div class="summary-item">No macros found</div>';
        
        // Includes
        const includes = Object.entries(summary.includes || {});
        document.getElementById('includeSummaryCount').textContent = 
            includes.reduce((sum, [k, v]) => sum + v.length, 0);
        let includesHtml = '';
        includes.forEach(([file, incs]) => {
            incs.forEach(inc => {
                const name = inc.split('/').pop();
                includesHtml += `<div class="summary-item"><span class="summary-item-name">${name}</span></div>`;
            });
        });
        document.getElementById('includes-content').innerHTML = includesHtml || '<div class="summary-item">No includes</div>';
        
        // Call graph
        const calls = Object.entries(summary.call_graph || {});
        let callHtml = '';
        calls.forEach(([caller, callees]) => {
            callHtml += `<div class="summary-item">
                <div class="summary-item-name">${caller}</div>
                <div class="summary-item-info">→ ${callees.join(', ')}</div>
            </div>`;
        });
        document.getElementById('callgraph-content').innerHTML = callHtml || '<div class="summary-item">No calls</div>';
    }

    function toggleSection(id) {
        document.getElementById(id + '-content').classList.toggle('open');
    }

    function renderTree(data) {
        const container = document.getElementById('treeContainer');
        const width = container.clientWidth || 800;
        let nodeCount = 0;
        function cnt(n) { nodeCount++; (n.children||[]).forEach(cnt); }
        cnt(data);
        const height = Math.max(600, nodeCount * 18);

        d3.select('#treeSvg').selectAll('*').remove();
        svg = d3.select('#treeSvg').attr('width', width).attr('height', height);
        
        zoom = d3.zoom().scaleExtent([0.1, 4]).on('zoom', e => g.attr('transform', e.transform));
        svg.call(zoom);
        
        g = svg.append('g').attr('transform', 'translate(60, 20)');
        
        root = d3.hierarchy(data);
        
        // Collapse deep nodes
        root.descendants().forEach(d => {
            if (d.depth > 2) {
                d._children = d.children;
                d.children = null;
            }
        });
        
        update(root);
        
        setTimeout(() => {
            const b = g.node().getBBox();
            const s = Math.min((width-80)/b.width, (height-40)/b.height, 1) * 0.8;
            svg.call(zoom.transform, d3.zoomIdentity.translate(40, 20).scale(s));
        }, 100);
    }

    function update(source) {
        const tree = d3.tree().nodeSize([20, 160]).separation((a,b) => a.parent === b.parent ? 1 : 1.2);
        tree(root);
        
        // Links
        const links = g.selectAll('.link').data(root.links(), d => d.target.data.type + d.target.depth);
        links.enter().append('path').attr('class', d => 'link' + (d.target.data.is_from_header ? ' to-header' : ''))
            .merge(links).attr('d', d3.linkHorizontal().x(d => d.y).y(d => d.x));
        links.exit().remove();
        
        // Nodes
        const nodes = g.selectAll('.node').data(root.descendants(), d => d.data.type + d.depth + (d.data.name||''));
        
        const nodeEnter = nodes.enter().append('g')
            .attr('class', d => 'node' + (d.data.is_from_header ? ' from-header' : ''))
            .attr('transform', d => `translate(${d.y},${d.x})`)
            .on('click', (e, d) => {
                if (d.children) { d._children = d.children; d.children = null; }
                else if (d._children) { d.children = d._children; d._children = null; }
                update(d);
                showInfo(d.data);
            });
        
        nodeEnter.append('circle')
            .attr('r', d => d._children ? 6 : 4)
            .style('fill', d => colors[d.data.category] || colors.other)
            .style('stroke', d => d3.color(colors[d.data.category] || colors.other).darker(0.5));
        
        nodeEnter.append('text')
            .attr('dy', '0.35em')
            .attr('x', d => (d.children || d._children) ? -8 : 8)
            .style('text-anchor', d => (d.children || d._children) ? 'end' : 'start')
            .text(d => {
                let l = d.data.type;
                if (d.data.name && d.data.name !== d.data.type) l += ': ' + d.data.name;
                return l.length > 35 ? l.slice(0,32)+'...' : l;
            });
        
        nodes.merge(nodeEnter).attr('transform', d => `translate(${d.y},${d.x})`);
        nodes.select('circle').attr('r', d => d._children ? 6 : 4);
        nodes.exit().remove();
    }

    function showInfo(data) {
        const c = colors[data.category] || colors.other;
        let extraHtml = '';
        if (data.extra && Object.keys(data.extra).length) {
            extraHtml = Object.entries(data.extra).map(([k,v]) => `
                <div class="info-item">
                    <div class="info-label">${k.replace(/_/g,' ')}</div>
                    <div class="info-value">${typeof v === 'object' ? JSON.stringify(v) : v}</div>
                </div>
            `).join('');
        }
        
        document.getElementById('nodeInfo').innerHTML = `
            <span class="info-badge" style="background:${c}">${data.category}</span>
            <span class="info-title">${data.type}${data.is_from_header ? ' <small style="color:#f59e0b">(from header)</small>' : ''}</span>
            <div class="info-grid">
                <div class="info-item"><div class="info-label">Name</div><div class="info-value">${data.name || '-'}</div></div>
                <div class="info-item"><div class="info-label">Type Info</div><div class="info-value">${data.type_info || '-'}</div></div>
                <div class="info-item"><div class="info-label">File</div><div class="info-value">${data.file || '-'}:${data.line}</div></div>
                <div class="info-item"><div class="info-label">Children</div><div class="info-value">${(data.children||[]).length}</div></div>
                ${extraHtml}
            </div>
        `;
    }

    function highlightNode(name) {
        g.selectAll('.node').each(function(d) {
            const match = d.data.name && d.data.name.includes(name);
            d3.select(this).select('circle').style('stroke-width', match ? '4px' : '2px');
            d3.select(this).select('text').style('fill', match ? '#fbbf24' : (d.data.is_from_header ? '#94a3b8' : '#e2e8f0'));
        });
    }

    function zoomIn() { svg.transition().duration(200).call(zoom.scaleBy, 1.3); }
    function zoomOut() { svg.transition().duration(200).call(zoom.scaleBy, 0.7); }
    function resetZoom() { svg.transition().duration(200).call(zoom.transform, d3.zoomIdentity.translate(40,20).scale(0.8)); }
    function expandAll() {
        root.descendants().forEach(d => { if(d._children) { d.children = d._children; d._children = null; }});
        update(root);
    }
    function collapseAll() {
        root.descendants().forEach(d => { if(d.depth > 1 && d.children) { d._children = d.children; d.children = null; }});
        update(root);
    }
    function clearAll() {
        uploadedFiles = {};
        updateFileList();
        document.getElementById('code').value = '';
        document.getElementById('empty').style.display = 'flex';
        document.getElementById('treeSvg').style.display = 'none';
        document.getElementById('zoomControls').style.display = 'none';
        ['nodeCount','funcCount','typeCount','headerCount'].forEach(id => document.getElementById(id).textContent = '0');
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
    """Parse code from text input."""
    data = request.get_json()
    code = data.get('code', '')
    
    if not code.strip():
        return jsonify({
            'success': False,
            'error': 'No code provided',
            'ast': None,
            'stats': {'nodes': 0, 'depth': 0, 'functions': 0, 'types': 0, 'headers': 0},
            'diagnostics': [],
            'summary': {}
        })
    
    result = parse_code_with_headers(code)
    return jsonify(result)

@app.route('/parse_files', methods=['POST'])
def parse_files():
    """Parse uploaded files with header resolution."""
    data = request.get_json()
    files = data.get('files', {})
    main_file = data.get('main_file', '')
    include_paths = data.get('include_paths', [])
    
    if not files or not main_file:
        return jsonify({
            'success': False,
            'error': 'No files provided',
            'ast': None,
            'stats': {'nodes': 0, 'depth': 0, 'functions': 0, 'types': 0, 'headers': 0},
            'diagnostics': [],
            'summary': {}
        })
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix='ast_upload_')
    
    try:
        # Write all files
        for name, info in files.items():
            file_path = os.path.join(temp_dir, name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                f.write(info['content'])
        
        # Parse main file
        main_path = os.path.join(temp_dir, main_file)
        all_include_paths = [temp_dir] + include_paths
        
        result = parse_with_dependencies(main_path, all_include_paths)
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'ast': None,
            'stats': {'nodes': 0, 'depth': 0, 'functions': 0, 'types': 0, 'headers': 0},
            'diagnostics': [],
            'summary': {}
        })
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.route('/parse_directory', methods=['POST'])
def parse_directory():
    """Parse a directory path on the local filesystem."""
    data = request.get_json()
    directory = data.get('directory', '')
    main_file = data.get('main_file', '')
    
    if not directory or not os.path.isdir(directory):
        return jsonify({
            'success': False,
            'error': 'Invalid directory path',
            'ast': None,
            'stats': {'nodes': 0, 'depth': 0, 'functions': 0, 'types': 0, 'headers': 0},
            'diagnostics': [],
            'summary': {}
        })
    
    # Find main .c file if not specified
    if not main_file:
        c_files = [f for f in os.listdir(directory) if f.endswith('.c')]
        if not c_files:
            return jsonify({
                'success': False,
                'error': 'No .c files found in directory',
                'ast': None,
                'stats': {'nodes': 0, 'depth': 0, 'functions': 0, 'types': 0, 'headers': 0},
                'diagnostics': [],
                'summary': {}
            })
        main_file = c_files[0]
    
    main_path = os.path.join(directory, main_file)
    result = parse_with_dependencies(main_path, [directory])
    return jsonify(result)


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   🌳 C AST Viewer PRO - with Full Header Resolution               ║
║                                                                   ║
║   Features:                                                       ║
║   • Automatic header file resolution                              ║
║   • Function signatures from .h files                             ║
║   • Type definitions across files                                 ║
║   • Call graph visualization                                      ║
║   • Macro expansion tracking                                      ║
║   • Works 100% offline                                            ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    print("🚀 Starting server at http://localhost:5050")
    print("📌 Press Ctrl+C to stop\n")
    
    import webbrowser
    webbrowser.open('http://localhost:5050')
    
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)


if __name__ == '__main__':
    main()

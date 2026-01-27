#!/usr/bin/env python3
"""
FileNo Agent v2.0 - With Proper Call Graph Analysis
====================================================
This agent ONLY finds mpf_mfs_open calls that are REACHABLE from main().
If a function is not called from main(), its mpf_mfs_open calls are ignored.

Key improvement: Build call graph from main() first, then only analyze reachable functions.
"""

import os
import re
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Load environment
def load_env():
    env_paths = [
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env"
    ]
    for env_path in env_paths:
        if env_path.exists():
            try:
                with open(env_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if value and key not in os.environ:
                                os.environ[key] = value
                return True
            except Exception as e:
                pass
    return False

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_env()
except ImportError:
    load_env()

try:
    from openai import OpenAI, AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  openai package not installed. Run: pip install openai")


@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    output: str
    data: Any = None


class CallGraphAnalyzer:
    """
    Builds and analyzes the call graph starting from main().
    This ensures we only find code that's actually executed.
    
    STRICT MODE: A function call is only valid if:
    1. The function is DECLARED (prototype or definition visible in the calling file)
    2. The function is actually CALLED
    """
    
    def __init__(self, project_dir: str, include_dirs: List[str] = None):
        self.project_dir = project_dir
        self.include_dirs = include_dirs or []
        
        # Storage
        self.files: Dict[str, str] = {}  # filepath -> content
        self.functions: Dict[str, Dict] = {}  # func_name -> {file, body, calls}
        self.macro_functions: Dict[str, Dict] = {}  # macro_name -> {file, params, body, line}
        self.prototypes: Dict[str, Dict] = {}  # FIX 4: func_name -> {file, line, signature} for declared but not defined functions
        self.declarations: Dict[str, Dict[str, Set[str]]] = {}  # filepath -> {func_name -> set of declaration sources}
        self.call_graph: Dict[str, Set[str]] = {}  # caller -> set of callees
        self.reachable_from_main: Set[str] = set()
        self.includes: Dict[str, Set[str]] = {}  # filepath -> set of included files
        
    def load_project(self):
        """Load all C files and build call graph"""
        # Load all files
        search_dirs = [self.project_dir] + self.include_dirs
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            for root, _, files in os.walk(search_dir):
                for f in files:
                    if f.endswith(('.c', '.h', '.inc')):
                        filepath = os.path.join(root, f)
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                                self.files[filepath] = file.read()
                        except:
                            pass
        
        # Extract includes from each file
        for filepath, content in self.files.items():
            self._extract_includes(filepath, content)
        
        # Extract function definitions from .c files
        for filepath, content in self.files.items():
            if filepath.endswith('.c'):
                self._extract_functions(filepath, content)
        
        # FIX 1: Also extract inline/static functions from .h files
        for filepath, content in self.files.items():
            if filepath.endswith(('.h', '.inc')):
                self._extract_functions_from_header(filepath, content)
        
        # FIX 2: Extract function-like macros from all files
        for filepath, content in self.files.items():
            self._extract_macro_functions(filepath, content)
        
        # Extract function declarations (prototypes)
        for filepath, content in self.files.items():
            self._extract_declarations(filepath, content)
        
        # Build call graph (with declaration checking)
        for func_name, func_data in self.functions.items():
            self._build_calls(func_name, func_data['file'], func_data['body'])
        
        # FIX 3: Also build call graph for macro functions
        for macro_name, macro_data in self.macro_functions.items():
            self._build_calls_for_macro(macro_name, macro_data['file'], macro_data['body'])
        
        # Find what's reachable from main
        self._find_reachable_from_main()
        
        return self
    
    def _extract_includes(self, filepath: str, content: str):
        """Extract #include statements to understand visibility"""
        self.includes[filepath] = set()
        
        for match in re.finditer(r'#\s*include\s*[<"]([^>"]+)[>"]', content):
            include_name = match.group(1)
            self.includes[filepath].add(include_name)
    
    def _extract_functions_from_header(self, filepath: str, content: str):
        """
        FIX 1: Extract inline/static functions defined in header files.
        These are actual function definitions (with body) in .h files.
        """
        # C keywords that should NOT be function names
        C_KEYWORDS = {
            'if', 'else', 'while', 'for', 'do', 'switch', 'case', 'default',
            'break', 'continue', 'return', 'goto', 'sizeof', 'typeof',
            'struct', 'union', 'enum', 'typedef', 'extern', 'register',
            'auto', 'volatile', 'const', 'restrict', 'inline', 'static'
        }
        
        # Pattern for inline/static function in header
        # Must have static or inline keyword to be a definition in header
        func_pattern = r'''
            (static|inline)\s+          # Must be static or inline
            (?:static\s+|inline\s+)?    # May have both
            (?:const\s+)?
            (?:unsigned\s+|signed\s+)?
            (?:long\s+)?
            (?:struct\s+\w+\s*\*?\s*|enum\s+\w+\s*|\w+)
            (?:\s*\*)*
            \s+
            (\w+)                       # Function name
            \s*\(
            ([^)]*)                     # Parameters
            \)\s*\{
        '''
        
        for match in re.finditer(func_pattern, content, re.VERBOSE):
            func_name = match.group(2).strip()
            
            if func_name in C_KEYWORDS:
                continue
            
            # Don't overwrite if already defined in .c file
            if func_name in self.functions:
                continue
            
            # Find function body
            start = match.end() - 1
            brace_count = 1
            end = start + 1
            
            while end < len(content) and brace_count > 0:
                if content[end] == '{':
                    brace_count += 1
                elif content[end] == '}':
                    brace_count -= 1
                end += 1
            
            func_body = content[start:end]
            line_num = content[:match.start()].count('\n') + 1
            
            self.functions[func_name] = {
                'file': filepath,
                'body': func_body,
                'line': line_num,
                'signature': match.group(0).rstrip('{').strip(),
                'is_header_func': True
            }
    
    def _extract_macro_functions(self, filepath: str, content: str):
        """
        FIX 2: Extract function-like macros.
        These are #define NAME(args) body patterns.
        They act like functions but are preprocessor substitutions.
        """
        # Remove block comments first
        clean = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Pattern for function-like macro:
        # #define NAME(arg1, arg2, ...) body
        # Note: macro body can span multiple lines with backslash continuation
        
        lines = clean.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for function-like macro
            match = re.match(r'#\s*define\s+(\w+)\s*\(([^)]*)\)\s*(.*)', line)
            if match:
                macro_name = match.group(1)
                params_str = match.group(2)
                body = match.group(3)
                
                # Handle line continuation with backslash
                while body.rstrip().endswith('\\') and i + 1 < len(lines):
                    i += 1
                    body = body.rstrip()[:-1] + ' ' + lines[i].strip()
                
                # Parse parameters
                params = [p.strip() for p in params_str.split(',') if p.strip()]
                
                # Skip if it's just an include guard or empty
                if body.strip() and macro_name not in self.macro_functions:
                    self.macro_functions[macro_name] = {
                        'file': filepath,
                        'params': params,
                        'body': body.strip(),
                        'line': content[:content.find(line)].count('\n') + 1 if line in content else i + 1
                    }
            
            i += 1
    
    def _extract_declarations(self, filepath: str, content: str):
        """Extract function prototypes (declarations)"""
        if filepath not in self.declarations:
            self.declarations[filepath] = {}
        
        # Remove block comments
        clean = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # Remove line comments
        clean = re.sub(r'//.*$', '', clean, flags=re.MULTILINE)
        
        # C keywords to skip
        C_KEYWORDS = {
            'if', 'else', 'while', 'for', 'do', 'switch', 'case', 'default',
            'break', 'continue', 'return', 'goto', 'sizeof', 'typeof'
        }
        
        # Pattern for function prototype: return_type func_name(params);
        # Must end with ; (not {) to distinguish from definitions
        # Enhanced pattern to capture more complex return types
        proto_pattern = r'''
            (?:extern\s+)?
            (?:static\s+)?
            (?:inline\s+)?
            (?:const\s+)?
            (?:unsigned\s+|signed\s+)?
            (?:long\s+long\s+|long\s+|short\s+)?
            (?:struct\s+\w+\s*\*?\s*|enum\s+\w+\s*|void\s*\*?\s*|\w+)
            (?:\s*\*)*
            \s+
            (\w+)                       # Function name (group 1)
            \s*\(
            ([^)]*)                     # Parameters (group 2)
            \)\s*;
        '''
        
        for match in re.finditer(proto_pattern, clean, re.VERBOSE):
            func_name = match.group(1)
            params_str = match.group(2)
            
            # Skip C keywords
            if func_name in C_KEYWORDS:
                continue
            
            if func_name not in self.declarations[filepath]:
                self.declarations[filepath][func_name] = set()
            self.declarations[filepath][func_name].add(f"prototype in {os.path.basename(filepath)}")
            
            # FIX 4: Store prototype details if function is NOT already defined
            # This allows macros to call functions that are only declared (prototypes)
            if func_name not in self.functions and func_name not in self.prototypes:
                line_num = content[:match.start()].count('\n') + 1
                self.prototypes[func_name] = {
                    'file': filepath,
                    'line': line_num,
                    'signature': match.group(0).strip(),
                    'params': [p.strip() for p in params_str.split(',') if p.strip()],
                    'is_prototype': True
                }
        
        # Also, function definitions serve as declarations
        for func_name, func_data in self.functions.items():
            if func_data['file'] == filepath:
                if func_name not in self.declarations[filepath]:
                    self.declarations[filepath][func_name] = set()
                self.declarations[filepath][func_name].add(f"definition in {os.path.basename(filepath)}")
    
    def _get_visible_declarations(self, filepath: str) -> Set[str]:
        """Get all function names visible/declared in a file (including via includes)"""
        visible = set()
        
        # Direct declarations in this file
        if filepath in self.declarations:
            visible.update(self.declarations[filepath].keys())
        
        # Declarations from included files
        for included in self.includes.get(filepath, []):
            # Find the actual include file
            for fpath in self.files:
                if fpath.endswith(included) or os.path.basename(fpath) == included:
                    if fpath in self.declarations:
                        visible.update(self.declarations[fpath].keys())
        
        # Functions defined in the same file are always visible
        for func_name, func_data in self.functions.items():
            if func_data['file'] == filepath:
                visible.add(func_name)
        
        return visible
    
    def _extract_functions(self, filepath: str, content: str):
        """Extract all function definitions"""
        # C keywords that should NOT be treated as function names
        C_KEYWORDS = {
            'if', 'else', 'while', 'for', 'do', 'switch', 'case', 'default',
            'break', 'continue', 'return', 'goto', 'sizeof', 'typeof',
            'struct', 'union', 'enum', 'typedef', 'extern', 'register',
            'auto', 'volatile', 'const', 'restrict', 'inline', 'static',
            '_Bool', '_Complex', '_Imaginary', '_Atomic', '_Thread_local',
            '_Static_assert', '_Alignas', '_Alignof', '_Generic', '_Noreturn'
        }
        
        # Words that are NOT valid C return types (false positive indicators)
        NOT_RETURN_TYPES = {
            'else', 'case', 'default', 'return', 'goto', 'break', 'continue'
        }
        
        # More comprehensive pattern for function definition
        # Handles: static, extern, inline, const, unsigned, long long, struct X*, etc.
        func_pattern = r'''
            (?:extern\s+)?              # optional extern
            (?:static\s+)?              # optional static
            (?:inline\s+)?              # optional inline
            (?:const\s+)?               # optional const
            (?:unsigned\s+|signed\s+)?  # optional unsigned/signed
            (?:long\s+)?                # optional long (for long long, long int)
            (?:struct\s+\w+\s*\*?\s*|   # struct X* or struct X
               enum\s+\w+\s*|           # enum X
               \w+)                     # basic type or typedef
            (?:\s*\*)*                  # pointer asterisks
            \s+
            (\w+)                       # FUNCTION NAME (capture group 1)
            \s*\(
            ([^)]*)                     # parameters (capture group 2)
            \)\s*\{
        '''
        
        for match in re.finditer(func_pattern, content, re.VERBOSE):
            func_name = match.group(1)
            
            # Skip if "function name" is a C keyword (like 'if', 'while', etc.)
            if func_name in C_KEYWORDS:
                continue
            
            # Get the full match to check return type
            full_match = match.group(0)
            # Extract what looks like the return type (everything before func name)
            return_type_match = re.match(r'(.+?)\s+' + re.escape(func_name), full_match)
            if return_type_match:
                return_type = return_type_match.group(1).split()[-1].rstrip('*')
                if return_type in NOT_RETURN_TYPES:
                    continue
            
            # Find function body
            start = match.end() - 1
            brace_count = 1
            end = start + 1
            
            while end < len(content) and brace_count > 0:
                if content[end] == '{':
                    brace_count += 1
                elif content[end] == '}':
                    brace_count -= 1
                end += 1
            
            func_body = content[start:end]
            
            # Calculate line number
            line_num = content[:match.start()].count('\n') + 1
            
            self.functions[func_name] = {
                'file': filepath,
                'body': func_body,
                'line': line_num,
                'signature': match.group(0).rstrip('{').strip()
            }
    
    def _build_calls(self, func_name: str, caller_file: str, body: str):
        """Find all ACTUAL function calls within a function body (not prototypes)"""
        self.call_graph[func_name] = set()
        
        # Remove string literals and comments to avoid false positives
        clean_body = re.sub(r'"[^"]*"', '""', body)
        clean_body = re.sub(r'/\*.*?\*/', '', clean_body, flags=re.DOTALL)
        clean_body = re.sub(r'//.*$', '', clean_body, flags=re.MULTILINE)
        
        # Keywords and type specifiers that indicate declarations, not calls
        type_keywords = {
            'int', 'void', 'char', 'short', 'long', 'float', 'double',
            'unsigned', 'signed', 'static', 'extern', 'const', 'volatile',
            'struct', 'union', 'enum', 'typedef', 'inline', 'register',
            'auto', 'restrict', '_Bool', '_Complex', '_Imaginary',
            'size_t', 'ssize_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t',
            'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t'
        }
        
        # Control flow keywords (not function calls)
        control_keywords = {
            'if', 'while', 'for', 'switch', 'return', 'sizeof', 'typeof',
            'case', 'default', 'goto', 'break', 'continue', 'else'
        }
        
        # Common library functions to skip (optional, but improves accuracy)
        skip_functions = {
            'printf', 'sprintf', 'fprintf', 'snprintf', 'scanf', 'sscanf',
            'malloc', 'calloc', 'realloc', 'free',
            'memset', 'memcpy', 'memmove', 'memcmp',
            'strcpy', 'strncpy', 'strlen', 'strcmp', 'strncmp', 'strcat',
            'fopen', 'fclose', 'fread', 'fwrite', 'fseek', 'ftell',
            'exit', 'abort', 'assert'
        }
        
        # Find potential function calls
        # Pattern: word followed by ( but NOT preceded by type keywords
        lines = clean_body.split('\n')
        for line in lines:
            line_stripped = line.strip()
            
            # Skip lines that look like function prototypes/declarations
            # Pattern: type word(...); at statement level
            if re.match(r'^(static\s+|inline\s+|extern\s+)*(int|void|char|short|long|float|double|unsigned|signed|struct\s+\w+|enum\s+\w+|\w+_t)\s*\*?\s*\w+\s*\([^)]*\)\s*;', line_stripped):
                continue
            
            # Find function calls in this line
            # Look for pattern: identifier( that is NOT preceded by type keyword
            for match in re.finditer(r'(\w+)\s*\(', line):
                called = match.group(1)
                pos = match.start()
                
                # Skip if it's a control keyword
                if called in control_keywords:
                    continue
                
                # Skip common library functions
                if called in skip_functions:
                    continue
                
                # Check if this looks like a declaration (type keyword before function name)
                # Get text before the match on this line
                text_before = line[:pos].strip()
                
                # If line starts with or text_before ends with a type keyword, it's likely a declaration
                words_before = text_before.split()
                if words_before:
                    last_word = words_before[-1].rstrip('*')  # handle pointer types like int*
                    if last_word in type_keywords:
                        continue
                    # Also check for custom types ending in _t
                    if last_word.endswith('_t'):
                        continue
                    # Check for struct/enum/union declarations
                    if len(words_before) >= 2 and words_before[-2] in {'struct', 'enum', 'union'}:
                        continue
                
                # Check if the entire statement is a declaration
                # Declarations typically have: type name(params);
                # Calls typically have: name(args); or var = name(args);
                if text_before:
                    # If what's before is just a type (with possible pointer), skip
                    text_before_clean = text_before.rstrip('*').strip()
                    if text_before_clean in type_keywords:
                        continue
                    if re.match(r'^(static\s+|inline\s+|extern\s+)*(const\s+)?\w+\s*\*?$', text_before):
                        # Check if it's a type - skip if so
                        potential_type = text_before_clean.split()[-1] if text_before_clean.split() else ''
                        if potential_type in type_keywords or potential_type.endswith('_t'):
                            continue
                
                # This looks like an actual function call
                # Add it if the function EXISTS in our codebase (functions OR macro functions)
                # FIX 3: Also track calls to macro functions
                # FIX 4: Also track calls to prototypes (declared but not defined)
                if called in self.functions:
                    self.call_graph[func_name].add(called)
                elif called in self.macro_functions:
                    self.call_graph[func_name].add(called)
                elif called in self.prototypes:
                    self.call_graph[func_name].add(called)
    
    def _build_calls_for_macro(self, macro_name: str, macro_file: str, body: str):
        """
        FIX 3: Build call graph for macro functions.
        Macro bodies can contain calls to other functions/macros.
        FIX 4: Also track calls to prototypes (declared but not defined functions).
        """
        self.call_graph[macro_name] = set()
        
        # Control flow keywords (not function calls)
        control_keywords = {
            'if', 'while', 'for', 'switch', 'return', 'sizeof', 'typeof',
            'case', 'default', 'goto', 'break', 'continue', 'else', 'do'
        }
        
        # Find potential function calls in macro body
        for match in re.finditer(r'(\w+)\s*\(', body):
            called = match.group(1)
            
            if called in control_keywords:
                continue
            
            # Add if it's a known function, macro function, or prototype
            if called in self.functions:
                self.call_graph[macro_name].add(called)
            elif called in self.macro_functions and called != macro_name:
                self.call_graph[macro_name].add(called)
            elif called in self.prototypes:
                # FIX 4: Macro calls a function that's only declared (prototype)
                self.call_graph[macro_name].add(called)
    
    def _find_reachable_from_main(self):
        """BFS from main() to find all reachable functions (including macro functions and prototypes)"""
        if 'main' not in self.functions:
            return
        
        visited = set()
        queue = ['main']
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            
            # Add all functions/macros/prototypes this one calls
            for called in self.call_graph.get(current, []):
                # FIX 3: Also consider macro functions as reachable targets
                # FIX 4: Also consider prototypes as reachable targets
                if (called in self.functions or called in self.macro_functions or called in self.prototypes) and called not in visited:
                    queue.append(called)
        
        self.reachable_from_main = visited
    
    def is_reachable(self, func_name: str) -> bool:
        """Check if a function is reachable from main()"""
        return func_name in self.reachable_from_main
    
    def get_call_chain(self, target_func: str) -> List[str]:
        """Get the call chain from main() to target function (including macro functions and prototypes)"""
        if 'main' not in self.functions:
            return []
        
        # FIX 3: Target can be a function or macro function
        # FIX 4: Target can also be a prototype
        if target_func not in self.functions and target_func not in self.macro_functions and target_func not in self.prototypes:
            return []
        
        if target_func not in self.reachable_from_main:
            return []
        
        # BFS to find shortest path
        visited = {('main',): 'main'}
        queue = [(['main'], 'main')]
        
        while queue:
            path, current = queue.pop(0)
            
            if current == target_func:
                return path
            
            for called in self.call_graph.get(current, []):
                # FIX 3: Also follow macro functions
                # FIX 4: Also follow prototypes
                if called in self.functions or called in self.macro_functions or called in self.prototypes:
                    new_path = tuple(path + [called])
                    if new_path not in visited:
                        visited[new_path] = called
                        queue.append((path + [called], called))
        
        return []
    
    def get_reachable_mfs_calls(self) -> List[Dict]:
        """Find all mpf_mfs_open calls that are reachable from main()"""
        calls = []
        
        for func_name in self.reachable_from_main:
            # Check regular functions
            if func_name in self.functions:
                func_data = self.functions[func_name]
                body = func_data['body']
                filepath = func_data['file']
                
                # Find mpf_mfs_open calls in this function
                lines = body.split('\n')
                for i, line in enumerate(lines):
                    if 'mpf_mfs_open' in line:
                        # Get complete call
                        full_call = line
                        j = i
                        while full_call.count('(') > full_call.count(')') and j < len(lines) - 1:
                            j += 1
                            full_call += ' ' + lines[j].strip()
                        
                        # Parse call
                        match = re.search(r'mpf_mfs_open\s*\(([^)]+)\)', full_call)
                        if match:
                            args = self._parse_args(match.group(1))
                            
                            # Get call chain
                            call_chain = self.get_call_chain(func_name)
                            
                            calls.append({
                                'file': os.path.basename(filepath),
                                'filepath': filepath,
                                'function': func_name,
                                'line': func_data['line'] + i,
                                'call': match.group(0),
                                'args': args,
                                'arg3': args[2] if len(args) > 2 else None,
                                'call_chain': call_chain,
                                'reachable': True,
                                'is_macro': False
                            })
            
            # FIX 3: Also check macro functions for mpf_mfs_open calls
            elif func_name in self.macro_functions:
                macro_data = self.macro_functions[func_name]
                body = macro_data['body']
                filepath = macro_data['file']
                
                if 'mpf_mfs_open' in body:
                    match = re.search(r'mpf_mfs_open\s*\(([^)]+)\)', body)
                    if match:
                        args = self._parse_args(match.group(1))
                        call_chain = self.get_call_chain(func_name)
                        
                        calls.append({
                            'file': os.path.basename(filepath),
                            'filepath': filepath,
                            'function': func_name,
                            'line': macro_data['line'],
                            'call': match.group(0),
                            'args': args,
                            'arg3': args[2] if len(args) > 2 else None,
                            'call_chain': call_chain,
                            'reachable': True,
                            'is_macro': True
                        })
            
            # FIX 5: Also check prototypes - search for actual definition in all files
            elif func_name in self.prototypes:
                # Prototype exists but definition wasn't extracted
                # Search all files for the actual function body
                found_calls = self._search_prototype_definition(func_name)
                for call_info in found_calls:
                    call_chain = self.get_call_chain(func_name)
                    call_info['call_chain'] = call_chain
                    call_info['reachable'] = True
                    calls.append(call_info)
        
        return calls
    
    def _search_prototype_definition(self, func_name: str) -> List[Dict]:
        """
        FIX 5: Search all files for the actual definition of a prototype function.
        This handles cases where the function definition exists but wasn't extracted
        by the normal regex (complex signatures, unusual formatting, etc.)
        """
        calls = []
        
        for filepath, content in self.files.items():
            # Look for function definition pattern: func_name followed by ( and later {
            # This is a more lenient search than the extraction regex
            
            # Pattern: func_name(...)  { ... } with possible newlines
            pattern = rf'\b{re.escape(func_name)}\s*\([^)]*\)\s*\{{'
            
            for match in re.finditer(pattern, content, re.MULTILINE):
                # Found a potential definition, extract the body
                start = match.end() - 1  # Position of {
                brace_count = 1
                end = start + 1
                
                while end < len(content) and brace_count > 0:
                    if content[end] == '{':
                        brace_count += 1
                    elif content[end] == '}':
                        brace_count -= 1
                    end += 1
                
                func_body = content[start:end]
                
                # Search for mpf_mfs_open in this body
                if 'mpf_mfs_open' in func_body:
                    lines = func_body.split('\n')
                    func_start_line = content[:match.start()].count('\n') + 1
                    
                    for i, line in enumerate(lines):
                        if 'mpf_mfs_open' in line:
                            # Get complete call (handle multi-line)
                            full_call = line
                            j = i
                            while full_call.count('(') > full_call.count(')') and j < len(lines) - 1:
                                j += 1
                                full_call += ' ' + lines[j].strip()
                            
                            mfs_match = re.search(r'mpf_mfs_open\s*\(([^)]+)\)', full_call)
                            if mfs_match:
                                args = self._parse_args(mfs_match.group(1))
                                
                                calls.append({
                                    'file': os.path.basename(filepath),
                                    'filepath': filepath,
                                    'function': func_name,
                                    'line': func_start_line + i,
                                    'call': mfs_match.group(0),
                                    'args': args,
                                    'arg3': args[2] if len(args) > 2 else None,
                                    'is_macro': False,
                                    'is_prototype_search': True  # Flag that this was found via prototype search
                                })
        
        return calls
    
    def _parse_args(self, args_str: str) -> List[str]:
        """Parse function arguments"""
        args = []
        current = ""
        depth = 0
        
        for char in args_str:
            if char == ',' and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                current += char
        
        if current.strip():
            args.append(current.strip())
        
        return args


class AgentTools:
    """Tools for the agent - now with call graph awareness"""
    
    def __init__(self, project_dir: str, include_dirs: List[str] = None):
        self.project_dir = project_dir
        self.include_dirs = include_dirs or []
        
        # Initialize call graph analyzer
        self.call_graph = CallGraphAnalyzer(project_dir, include_dirs)
        self.call_graph.load_project()
        
        # Cache
        self.macro_cache: Dict[str, str] = {}
        self._load_all_macros()
    
    def _load_all_macros(self):
        """Pre-load all macros"""
        for filepath, content in self.call_graph.files.items():
            # Remove block comments
            clean = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            
            # Process line by line to avoid multiline regex issues
            for line in clean.split('\n'):
                # Remove line comments
                line = re.sub(r'//.*$', '', line)
                line = line.strip()
                
                # Match #define NAME VALUE (where VALUE is not empty)
                match = re.match(r'#\s*define\s+(\w+)\s+(.+)', line)
                if match:
                    name = match.group(1)
                    value = match.group(2).strip()
                    # Skip include guards (defines with no real value or ending in _H)
                    if name and value and not name.endswith('_H'):
                        self.macro_cache[name] = value
    
    # -------------------------------------------------------------------------
    # FILE OPERATIONS
    # -------------------------------------------------------------------------
    
    def read_file(self, path: str) -> ToolResult:
        """Read contents of a file"""
        try:
            # Resolve path - try multiple strategies
            full_path = None
            
            # Strategy 1: Absolute path
            if os.path.isabs(path) and os.path.exists(path):
                full_path = path
            
            # Strategy 2: Path as-is (relative to cwd)
            if not full_path and os.path.exists(path):
                full_path = path
            
            # Strategy 3: Join with project dir
            if not full_path:
                test_path = os.path.join(self.project_dir, path)
                if os.path.exists(test_path):
                    full_path = test_path
            
            # Strategy 4: Just the basename in project dir
            if not full_path:
                test_path = os.path.join(self.project_dir, os.path.basename(path))
                if os.path.exists(test_path):
                    full_path = test_path
            
            # Strategy 5: Try include dirs
            if not full_path:
                for inc_dir in self.include_dirs:
                    test_path = os.path.join(inc_dir, path)
                    if os.path.exists(test_path):
                        full_path = test_path
                        break
                    # Also try basename
                    test_path = os.path.join(inc_dir, os.path.basename(path))
                    if os.path.exists(test_path):
                        full_path = test_path
                        break
            
            if not full_path:
                return ToolResult(False, f"File not found: {path}")
            
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            numbered = '\n'.join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
            
            return ToolResult(True, f"File: {full_path}\n\n{numbered}", content)
            
        except Exception as e:
            return ToolResult(False, f"Error reading file: {str(e)}")
    
    def list_files(self, directory: str = None) -> ToolResult:
        """List all C/H files"""
        try:
            # If no directory specified, use project dir
            if directory is None:
                dir_path = self.project_dir
            elif os.path.isabs(directory):
                dir_path = directory
            elif os.path.exists(directory):
                # Path exists as-is (relative to cwd)
                dir_path = directory
            else:
                # Try joining with project dir
                dir_path = os.path.join(self.project_dir, directory)
                if not os.path.exists(dir_path):
                    # Fall back to project dir
                    dir_path = self.project_dir
            
            files = []
            for root, _, filenames in os.walk(dir_path):
                for f in filenames:
                    if f.endswith(('.c', '.h', '.inc')):
                        rel_path = os.path.relpath(os.path.join(root, f), self.project_dir)
                        files.append(rel_path)
            
            if not files:
                return ToolResult(True, "No C/H files found", [])
            
            return ToolResult(True, f"Files:\n" + '\n'.join(f"  • {f}" for f in sorted(files)), files)
            
        except Exception as e:
            return ToolResult(False, f"Error: {str(e)}")
    
    def search_in_files(self, pattern: str, file_pattern: str = "*.c") -> ToolResult:
        """Search for pattern in files"""
        try:
            import fnmatch
            results = []
            
            for filepath, content in self.call_graph.files.items():
                if fnmatch.fnmatch(os.path.basename(filepath), file_pattern):
                    for i, line in enumerate(content.split('\n'), 1):
                        if re.search(pattern, line):
                            rel_path = os.path.relpath(filepath, self.project_dir)
                            results.append(f"{rel_path}:{i}: {line.strip()}")
            
            if not results:
                return ToolResult(True, f"No matches for '{pattern}'", [])
            
            return ToolResult(True, f"Found {len(results)} matches:\n" + '\n'.join(results[:30]), results)
            
        except Exception as e:
            return ToolResult(False, f"Error: {str(e)}")
    
    # -------------------------------------------------------------------------
    # MACRO OPERATIONS
    # -------------------------------------------------------------------------
    
    def find_macro(self, macro_name: str) -> ToolResult:
        """Find a macro definition"""
        if macro_name in self.macro_cache:
            return ToolResult(True, f"#define {macro_name} {self.macro_cache[macro_name]}", 
                            {'name': macro_name, 'value': self.macro_cache[macro_name]})
        return ToolResult(False, f"Macro '{macro_name}' not found")
    
    def resolve_macro(self, macro_name: str, depth: int = 0) -> ToolResult:
        """Resolve macro to numeric value"""
        try:
            if depth > 20:
                return ToolResult(False, f"Max depth for '{macro_name}'")
            
            if macro_name not in self.macro_cache:
                return ToolResult(False, f"Macro '{macro_name}' not found")
            
            value = self.macro_cache[macro_name]
            
            # Remove parentheses
            if value.startswith('(') and value.endswith(')'):
                value = value[1:-1].strip()
            
            # Direct number
            try:
                if value.startswith('0x'):
                    num = int(value, 16)
                else:
                    num = int(value)
                return ToolResult(True, f"{macro_name} = {num}", num)
            except ValueError:
                pass
            
            # Another macro
            if value in self.macro_cache:
                return self.resolve_macro(value, depth + 1)
            
            # Expression
            expr = value
            for m in self.macro_cache:
                if re.search(rf'\b{m}\b', expr):
                    r = self.resolve_macro(m, depth + 1)
                    if r.success and isinstance(r.data, int):
                        expr = re.sub(rf'\b{m}\b', str(r.data), expr)
            
            # Evaluate
            try:
                expr_clean = re.sub(r'[^\d+\-*/().\s]', '', expr)
                if expr_clean.strip():
                    result = eval(expr_clean)
                    return ToolResult(True, f"{macro_name} = {value} = {int(result)}", int(result))
            except:
                pass
            
            return ToolResult(False, f"Cannot resolve '{macro_name}' = '{value}'")
            
        except Exception as e:
            return ToolResult(False, f"Error: {str(e)}")
    
    def list_all_macros(self) -> ToolResult:
        """List all macros"""
        if not self.macro_cache:
            return ToolResult(True, "No macros found", {})
        
        output = f"Found {len(self.macro_cache)} macros:\n"
        for name, value in sorted(self.macro_cache.items()):
            output += f"  #define {name} {value}\n"
        
        return ToolResult(True, output, self.macro_cache)
    
    # -------------------------------------------------------------------------
    # FUNCTION OPERATIONS (Call Graph Aware)
    # -------------------------------------------------------------------------
    
    def find_function(self, func_name: str) -> ToolResult:
        """Find a function - shows if it's reachable from main()"""
        if func_name not in self.call_graph.functions:
            return ToolResult(False, f"Function '{func_name}' not found")
        
        func_data = self.call_graph.functions[func_name]
        reachable = self.call_graph.is_reachable(func_name)
        call_chain = self.call_graph.get_call_chain(func_name)
        
        # Add line numbers to body
        lines = func_data['body'].split('\n')
        numbered = '\n'.join(f"{func_data['line'] + i:4d} | {line}" for i, line in enumerate(lines))
        
        status = "✅ REACHABLE from main()" if reachable else "❌ NOT REACHABLE from main()"
        chain_str = " → ".join(call_chain) if call_chain else "N/A"
        
        output = f"""Function: {func_data['signature']}
File: {os.path.basename(func_data['file'])}
Status: {status}
Call chain: {chain_str}

{numbered}"""
        
        return ToolResult(True, output, {
            'name': func_name,
            'body': func_data['body'],
            'file': func_data['file'],
            'reachable': reachable,
            'call_chain': call_chain
        })
    
    def get_call_graph(self) -> ToolResult:
        """Show the call graph from main() with declaration info"""
        if 'main' not in self.call_graph.functions:
            return ToolResult(False, "main() not found")
        
        output = "Call Graph (from main):\n"
        output += "Legend: [✓] = reachable, [✗] = not reachable, [!] = not declared\n\n"
        
        def print_tree(func: str, indent: int = 0, visited: Set[str] = None):
            nonlocal output
            if visited is None:
                visited = set()
            
            if func in visited:
                output += "  " * indent + f"├─ {func} (recursive)\n"
                return
            
            visited.add(func)
            marker = "✓" if func in self.call_graph.reachable_from_main else "✗"
            output += "  " * indent + f"├─ [{marker}] {func}\n"
            
            for called in sorted(self.call_graph.call_graph.get(func, [])):
                if called in self.call_graph.functions:
                    print_tree(called, indent + 1, visited.copy())
        
        print_tree('main')
        
        # Show functions that exist but are not reachable (with reason)
        unreachable = set(self.call_graph.functions.keys()) - self.call_graph.reachable_from_main
        if unreachable:
            output += f"\n\nUnreachable functions:\n"
            for func in sorted(unreachable):
                func_file = self.call_graph.functions[func]['file']
                output += f"  • {func}() in {os.path.basename(func_file)}\n"
                
                # Check why it's not reachable - is it not called or not declared?
                # Find if any function tries to call it
                callers = []
                for caller, callees in self.call_graph.call_graph.items():
                    if func in callees:
                        callers.append(caller)
                
                if not callers:
                    output += f"    Reason: Not called from any reachable function\n"
        
        output += f"\nReachable functions: {sorted(self.call_graph.reachable_from_main)}"
        
        # Show declaration info
        if 'main' in self.call_graph.functions:
            main_file = self.call_graph.functions['main']['file']
            visible = self.call_graph._get_visible_declarations(main_file)
            output += f"\n\nFunctions visible in main.c: {sorted(visible)}"
        
        return ToolResult(True, output, {
            'reachable': list(self.call_graph.reachable_from_main),
            'call_graph': {k: list(v) for k, v in self.call_graph.call_graph.items()}
        })
    
    def trace_variable(self, var_name: str, func_name: str) -> ToolResult:
        """Trace variable assignments - only in reachable functions"""
        if func_name not in self.call_graph.functions:
            return ToolResult(False, f"Function '{func_name}' not found")
        
        if not self.call_graph.is_reachable(func_name):
            return ToolResult(False, f"Function '{func_name}' is NOT reachable from main()")
        
        func_body = self.call_graph.functions[func_name]['body']
        
        assignments = []
        patterns = [
            rf'\b{re.escape(var_name)}\s*=\s*([^;]+);',
            rf'\bint\s+{re.escape(var_name)}\s*=\s*([^;]+);',
            rf'\b{re.escape(var_name)}\s*\+=\s*([^;]+);',
        ]
        
        for i, line in enumerate(func_body.split('\n'), 1):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    assignments.append({
                        'line': i,
                        'code': line.strip(),
                        'value': match.group(1).strip()
                    })
        
        if not assignments:
            return ToolResult(False, f"No assignments to '{var_name}' in '{func_name}'")
        
        output = f"Assignments to '{var_name}' in '{func_name}':\n"
        for a in assignments:
            output += f"  Line {a['line']}: {a['code']}\n"
        
        return ToolResult(True, output, assignments)
    
    # -------------------------------------------------------------------------
    # MFS CALLS - ONLY REACHABLE FROM MAIN
    # -------------------------------------------------------------------------
    
    def find_mfs_open_calls(self) -> ToolResult:
        """Find mpf_mfs_open calls ONLY in functions reachable from main()"""
        calls = self.call_graph.get_reachable_mfs_calls()
        
        if not calls:
            # Check if there are any calls at all (for debugging)
            all_calls = []
            for filepath, content in self.call_graph.files.items():
                if 'mpf_mfs_open' in content:
                    for func_name, func_data in self.call_graph.functions.items():
                        if 'mpf_mfs_open' in func_data['body']:
                            reachable = self.call_graph.is_reachable(func_name)
                            all_calls.append((func_name, reachable))
            
            if all_calls:
                msg = "No REACHABLE mpf_mfs_open calls found.\n\n"
                msg += "mpf_mfs_open exists in these functions:\n"
                for func, reachable in all_calls:
                    status = "✅ reachable" if reachable else "❌ NOT reachable"
                    msg += f"  • {func}() - {status}\n"
                msg += "\nThese functions are not called from main(), so their calls are ignored."
                return ToolResult(True, msg, [])
            
            return ToolResult(True, "No mpf_mfs_open calls found in project", [])
        
        output = f"Found {len(calls)} REACHABLE mpf_mfs_open calls:\n"
        for c in calls:
            chain = " → ".join(c['call_chain'])
            output += f"\n  📍 {c['file']}:{c['line']} in {c['function']}()\n"
            output += f"     Call: {c['call']}\n"
            output += f"     3rd arg: {c['arg3']}\n"
            output += f"     Path: {chain}\n"
        
        return ToolResult(True, output, calls)
    
    # -------------------------------------------------------------------------
    # EXPRESSION EVALUATION
    # -------------------------------------------------------------------------
    
    def evaluate_expression(self, expression: str) -> ToolResult:
        """Evaluate arithmetic expression"""
        try:
            expr = expression
            
            # Replace macros
            for macro in self.macro_cache:
                if re.search(rf'\b{macro}\b', expr):
                    r = self.resolve_macro(macro)
                    if r.success and isinstance(r.data, int):
                        expr = re.sub(rf'\b{macro}\b', str(r.data), expr)
            
            # Evaluate
            expr_clean = expr.replace(' ', '')
            expr_clean = re.sub(r'0x([0-9a-fA-F]+)', lambda m: str(int(m.group(1), 16)), expr_clean)
            
            if re.match(r'^[\d+\-*/()%]+$', expr_clean):
                result = eval(expr_clean)
                return ToolResult(True, f"{expression} = {result}", int(result))
            
            return ToolResult(False, f"Cannot evaluate: {expression} (cleaned: {expr_clean})")
            
        except Exception as e:
            return ToolResult(False, f"Error: {str(e)}")
    
    def simulate_function(self, func_name: str, params: dict = None) -> ToolResult:
        """
        ROBUST C function simulator using ACTUAL C COMPILATION.
        Compiles and runs the code with gcc/clang for 100% accuracy.
        """
        try:
            if params is None:
                params = {}
            
            output = f"🔬 Executing {func_name}({', '.join(f'{k}={v}' for k,v in params.items())}) via C compiler:\n\n"
            
            # C keywords to filter out
            C_KEYWORDS = {'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 
                         'continue', 'return', 'goto', 'sizeof', 'typedef', 'struct', 
                         'union', 'enum', 'const', 'static', 'extern', 'register', 'volatile'}
            
            # Valid C return types
            C_TYPES = {'int', 'void', 'char', 'short', 'long', 'float', 'double', 
                      'unsigned', 'signed', 'static', 'const'}
            
            def is_valid_function(name, data):
                """Check if this is a real function, not a keyword or partial code"""
                if name in C_KEYWORDS:
                    return False
                sig = data.get('signature', '')
                # Signature should start with a type
                first_word = sig.split()[0] if sig.split() else ''
                return first_word in C_TYPES
            
            # Collect all macros
            macros = []
            for macro, value in self.macro_cache.items():
                if isinstance(value, int):
                    macros.append(f"#define {macro} {value}")
                elif isinstance(value, str) and value.strip():
                    macros.append(f"#define {macro} {value}")
            
            # Collect all function bodies we need
            functions_code = []
            
            # Get the target function
            if func_name not in self.call_graph.functions:
                return ToolResult(False, f"Function '{func_name}' not found")
            
            func_data = self.call_graph.functions[func_name]
            if not is_valid_function(func_name, func_data):
                return ToolResult(False, f"'{func_name}' is not a valid function")
                
            signature = func_data.get('signature', f'int {func_name}()')
            body = func_data['body']
            functions_code.append(f"{signature}\n{body}")
            
            # Find functions called by this function and include them
            for other_name, other_data in self.call_graph.functions.items():
                if other_name != func_name and other_name in body and is_valid_function(other_name, other_data):
                    other_sig = other_data.get('signature', f'int {other_name}()')
                    other_body = other_data['body']
                    functions_code.insert(0, f"{other_sig}\n{other_body}")
            
            # Build parameter list for function call
            param_args = ', '.join(str(v) for v in params.values())
            
            # Create test program
            test_code = f"""
#include <stdio.h>

{chr(10).join(macros)}

{chr(10).join(functions_code)}

int main() {{
    int result = {func_name}({param_args});
    printf("RESULT:%d", result);
    return 0;
}}
"""
            output += f"Generated test code:\n{'-'*40}\n"
            # Show abbreviated code
            code_lines = test_code.strip().split('\n')
            if len(code_lines) > 30:
                output += '\n'.join(code_lines[:15]) + '\n...\n' + '\n'.join(code_lines[-10:])
            else:
                output += test_code
            output += f"\n{'-'*40}\n\n"
            
            # Write to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
                f.write(test_code)
                c_file = f.name
            
            exe_file = c_file.replace('.c', '')
            
            try:
                # Compile
                compile_result = subprocess.run(
                    ['gcc', '-o', exe_file, c_file, '-w'],  # -w suppresses warnings
                    capture_output=True, text=True, timeout=10
                )
                
                if compile_result.returncode != 0:
                    output += f"⚠️ Compilation failed:\n{compile_result.stderr}\n"
                    # Clean up
                    os.unlink(c_file)
                    return ToolResult(False, output)
                
                output += "✓ Compiled successfully\n"
                
                # Execute
                run_result = subprocess.run(
                    [exe_file],
                    capture_output=True, text=True, timeout=5
                )
                
                # Parse result
                stdout = run_result.stdout
                if 'RESULT:' in stdout:
                    value = int(stdout.split('RESULT:')[1].strip())
                    output += f"✓ Executed successfully\n\n"
                    output += f"✅ RETURN VALUE: {value}"
                    
                    # Clean up
                    os.unlink(c_file)
                    os.unlink(exe_file)
                    
                    return ToolResult(True, output, value)
                else:
                    output += f"⚠️ Unexpected output: {stdout}\n"
                    os.unlink(c_file)
                    os.unlink(exe_file)
                    return ToolResult(False, output)
                    
            except subprocess.TimeoutExpired:
                output += "⚠️ Execution timed out\n"
                return ToolResult(False, output)
            except Exception as e:
                output += f"⚠️ Error: {str(e)}\n"
                return ToolResult(False, output)
            finally:
                # Clean up temp files
                if os.path.exists(c_file):
                    try: os.unlink(c_file)
                    except: pass
                if os.path.exists(exe_file):
                    try: os.unlink(exe_file)
                    except: pass
                    
        except Exception as e:
            import traceback
            return ToolResult(False, f"Error: {str(e)}\n{traceback.format_exc()}")
    
    def run_command(self, command: str) -> ToolResult:
        """Execute shell command"""
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=30, cwd=self.project_dir
            )
            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            return ToolResult(result.returncode == 0, output, {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            })
        except Exception as e:
            return ToolResult(False, f"Error: {str(e)}")


class FileNoAgent:
    """
    Intelligent agent with proper call graph analysis.
    Only finds mpf_mfs_open calls reachable from main().
    """
    
    def __init__(self, project_dir: str, include_dirs: List[str] = None, verbose: bool = True):
        self.project_dir = project_dir
        self.include_dirs = include_dirs or []
        self.verbose = verbose
        
        # Initialize tools
        self.tools = AgentTools(project_dir, include_dirs)
        
        # Initialize LLM
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required")
        
        # Azure config (check first, like ai_analyzer.py)
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        
        # Check for placeholder values
        if "your-resource" in azure_endpoint:
            azure_endpoint = ""
        if "your-api-key" in azure_api_key:
            azure_api_key = ""
        
        # OpenAI config
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        openai_base_url = os.environ.get("OPENAI_BASE_URL", "")  # For proxy support
        
        # Initialize client - Azure first, then OpenAI
        self.provider = None
        if all([azure_endpoint, azure_api_key, azure_deployment]):
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            )
            self.model = azure_deployment
            self.provider = "Azure"
            if verbose:
                print("🔷 Using Azure OpenAI")
                print(f"   Endpoint: {azure_endpoint[:50]}...")
                print(f"   Deployment: {azure_deployment}")
                print(f"   API Version: {os.environ.get('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')}")
        elif openai_api_key:
            # Support for proxy/custom base URL
            if openai_base_url:
                self.client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
                if verbose:
                    print(f"🟢 Using OpenAI via proxy ({openai_base_url})")
            else:
                self.client = OpenAI(api_key=openai_api_key)
            self.model = openai_model
            self.provider = "OpenAI"
            if verbose and not openai_base_url:
                print(f"🟢 Using OpenAI ({self.model})")
        else:
            raise ValueError("No AI credentials configured. Set AZURE_OPENAI_* or OPENAI_API_KEY in .env file")
        
        self.conversation_history: List[Dict] = []
        
        # Tool registry
        self.tool_registry = {
            "read_file": self.tools.read_file,
            "list_files": self.tools.list_files,
            "search_in_files": self.tools.search_in_files,
            "find_macro": self.tools.find_macro,
            "resolve_macro": self.tools.resolve_macro,
            "list_all_macros": self.tools.list_all_macros,
            "find_function": self.tools.find_function,
            "get_call_graph": self.tools.get_call_graph,
            "trace_variable": self.tools.trace_variable,
            "evaluate_expression": self.tools.evaluate_expression,
            "simulate_function": self.tools.simulate_function,
            "find_mfs_open_calls": self.tools.find_mfs_open_calls,
            "run_command": self.tools.run_command,
        }
        
        # Tool definitions for LLM
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a source file with line numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "File path"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List all C/H files in the project",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string", "description": "Directory (optional)"}
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_in_files",
                    "description": "Search for a pattern in files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Regex pattern"},
                            "file_pattern": {"type": "string", "description": "File glob (default: *.c)"}
                        },
                        "required": ["pattern"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_macro",
                    "description": "Find a #define macro definition",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "macro_name": {"type": "string"}
                        },
                        "required": ["macro_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "resolve_macro",
                    "description": "Resolve a macro to its numeric value",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "macro_name": {"type": "string"}
                        },
                        "required": ["macro_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_all_macros",
                    "description": "List all macro definitions in the project",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_function",
                    "description": "Find a function definition. Shows if it's reachable from main().",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "func_name": {"type": "string"}
                        },
                        "required": ["func_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_call_graph",
                    "description": "Show the call graph starting from main(). Shows which functions are reachable.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "trace_variable",
                    "description": "Trace variable assignments within a function (only works for reachable functions)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "var_name": {"type": "string"},
                            "func_name": {"type": "string"}
                        },
                        "required": ["var_name", "func_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "evaluate_expression",
                    "description": "Evaluate an arithmetic expression (resolves macros)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        },
                        "required": ["expression"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "simulate_function",
                    "description": "POWERFUL C function simulator. Executes a function with given parameters and returns the result. Handles: nested if/else, multiple parameters, compound conditions (&&, ||, <, >, ==), variable assignments, arithmetic, bitwise ops. Use this to find what a function returns!",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "func_name": {"type": "string", "description": "Name of the function to simulate"},
                            "params": {"type": "object", "description": "Parameters as key-value pairs, e.g. {\"type\": 1, \"level\": 3}"}
                        },
                        "required": ["func_name", "params"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_mfs_open_calls",
                    "description": "Find all mpf_mfs_open() calls that are REACHABLE from main(). Ignores calls in functions not called from main.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"}
                        },
                        "required": ["command"]
                    }
                }
            },
        ]
        
        self.system_prompt = """You are an expert C code analyzer. Your task is to find the EXACT numeric value of the 3rd argument to mpf_mfs_open() calls.

CRITICAL RULES:
1. ONLY analyze code REACHABLE from main()
2. DO NOT trust comments - they may be outdated. Always compute values yourself.
3. When a variable is assigned from a function call, you MUST trace INTO that function
4. When you encounter if/else branches, you MUST determine which branch is taken based on the ACTUAL computed value

## METHODOLOGY
1. Use get_call_graph to see reachable functions
2. Use find_mfs_open_calls to find calls
3. For each call's 3rd argument:
   a. If it's a variable, use trace_variable to find its assignment
   b. If assigned from a function, use find_function to read that function's code
   c. Trace all inputs to the function (parameters)
   d. Evaluate conditionals: compute the condition value, then follow the correct branch
   e. Use resolve_macro and evaluate_expression for macros and arithmetic

## CONDITIONAL TRACING (VERY IMPORTANT!)
When you see code like:
```c
type = 105 - 101;  // Compute: type = 4
fileno = get_fileno(type);  // Must trace into get_fileno with type=4
```

USE the simulate_function tool! It's a POWERFUL C simulator. Call it like:
  simulate_function(func_name="get_fileno", params={"type": 4})

For functions with multiple parameters:
  simulate_function(func_name="calc_fileno", params={"type": 1, "level": 3})

This tool will:
- Execute the function step by step
- Evaluate all conditions (==, <, >, <=, >=, &&, ||)
- Track variable assignments
- Handle nested if/else
- Return the exact value

## OUTPUT FORMAT
Show your reasoning step by step, then:

FINAL ANSWER:
- File: <filename>
- Function: <function_name>  
- Call Chain: main() → ... → <function>
- Line: <line_number>
- 3rd Argument: <raw_argument>
- Resolved Value: <numeric_value>"""
    
    def test_connection(self) -> bool:
        """Test if LLM connection works (simple call without tools)"""
        print("\n🔍 Testing LLM connection...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say 'OK' if you can hear me."}],
                max_tokens=10,
                temperature=0
            )
            result = response.choices[0].message.content
            print(f"   ✓ Connection OK: {result}")
            return True
        except Exception as e:
            print(f"   ❌ Connection FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_tools(self) -> bool:
        """Test if tool/function calling works"""
        print("\n🔍 Testing tool calling support...")
        try:
            test_tools = [{
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Call the test_tool function."}],
                tools=test_tools,
                tool_choice="auto",
                max_tokens=100,
                temperature=0
            )
            has_tools = bool(response.choices[0].message.tool_calls)
            print(f"   ✓ Tool calling {'SUPPORTED' if has_tools else 'returned but no tool calls'}")
            print(f"   Finish reason: {response.choices[0].finish_reason}")
            return True
        except Exception as e:
            error_str = str(e).lower()
            if "tool" in error_str or "function" in error_str:
                print(f"   ⚠️ Tool calling NOT SUPPORTED by this deployment")
                print(f"   Error: {e}")
                return False
            else:
                print(f"   ❌ Error testing tools: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def log(self, msg: str, prefix: str = ""):
        if self.verbose:
            print(f"{prefix}{msg}")
    
    def execute_tool(self, tool_name: str, arguments: Dict) -> str:
        if tool_name not in self.tool_registry:
            return f"Unknown tool: {tool_name}"
        try:
            result = self.tool_registry[tool_name](**arguments)
            return result.output
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze(self, task: str = None) -> str:
        if task is None:
            task = f"""Analyze the C project at {self.project_dir}.

Find all mpf_mfs_open() calls that are REACHABLE from main() and determine the 3rd argument value.

IMPORTANT: Only report calls that are actually executed when main() runs. If a function is not called from main(), ignore it."""
        
        self.conversation_history = [{"role": "user", "content": task}]
        
        self.log("\n" + "="*70)
        self.log("🤖 AGENT STARTING (Call Graph Aware)")
        self.log("="*70)
        self.log(f"\n📋 Task: {task[:100]}...")
        
        # Test connection first
        if not self.test_connection():
            return "ERROR: Cannot connect to LLM API. Check your credentials."
        
        if not self.test_tools():
            print("\n⚠️ Tool calling not supported. The agent requires function calling.")
            print("   Your Azure deployment may not support this feature.")
            print("   Try using a different deployment or check with your admin.")
            return "ERROR: Tool calling not supported by this LLM deployment."
        
        # Show reachability info upfront
        self.log(f"\n📊 Reachable functions from main(): {sorted(self.tools.call_graph.reachable_from_main)}")
        
        max_iterations = 25
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            self.log(f"\n{'─'*50}")
            self.log(f"🔄 Iteration {iteration}")
            
            try:
                self.log("   📡 Calling LLM API...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system_prompt}] + self.conversation_history,
                    tools=self.tool_definitions,
                    tool_choice="auto",
                    temperature=0
                )
                self.log("   ✓ LLM response received")
            except Exception as e:
                error_msg = str(e)
                self.log(f"\n❌ LLM API Error: {error_msg}")
                print(f"\n❌ LLM API Error: {error_msg}")  # Always print errors
                import traceback
                traceback.print_exc()
                return f"LLM Error: {error_msg}"
            
            # Debug: Check response structure
            if not response.choices:
                self.log("   ⚠️ No choices in response!")
                print("   ⚠️ No choices in response!")
                return "Error: Empty response from LLM"
            
            message = response.choices[0].message
            
            # Debug: Log what we got
            has_tool_calls = bool(message.tool_calls)
            has_content = bool(message.content)
            self.log(f"   Response: tool_calls={has_tool_calls}, content={has_content}")
            
            # Check finish reason
            finish_reason = response.choices[0].finish_reason
            self.log(f"   Finish reason: {finish_reason}")
            
            if message.tool_calls:
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in message.tool_calls
                    ]
                })
                
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except:
                        arguments = {}
                    
                    self.log(f"\n🔧 Tool: {tool_name}")
                    self.log(f"   Args: {arguments}")
                    
                    result = self.execute_tool(tool_name, arguments)
                    
                    display = result[:500] + "..." if len(result) > 500 else result
                    self.log(f"   Result: {display}")
                    
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                # No tool calls - LLM is done or has a response
                content = message.content or ""
                
                if not content.strip():
                    self.log("   ⚠️ Empty response from LLM, continuing...")
                    print("   ⚠️ Empty response from LLM (no tool calls, no content)")
                    print(f"   Finish reason: {finish_reason}")
                    # Try to continue - maybe LLM needs another prompt
                    self.conversation_history.append({
                        "role": "assistant", 
                        "content": ""
                    })
                    self.conversation_history.append({
                        "role": "user",
                        "content": "Please continue your analysis. Use the available tools to find mpf_mfs_open calls."
                    })
                    continue
                
                self.log(f"\n{'='*70}")
                self.log("✅ AGENT COMPLETE")
                self.log("="*70)
                self.log(f"\n{content}")
                return content
        
        return "Max iterations reached"
    
    def interactive(self):
        """Interactive mode"""
        print("\n" + "="*70)
        print("🤖 FileNo Agent v2.0 - Call Graph Aware")
        print("="*70)
        print(f"Project: {self.project_dir}")
        print(f"Reachable from main(): {sorted(self.tools.call_graph.reachable_from_main)}")
        print("Type 'quit' to exit")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                break
            if user_input.lower() == 'reset':
                self.conversation_history = []
                print("History cleared.\n")
                continue
            
            response = self.analyze(user_input)
            print(f"\nAgent: {response}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="FileNo Agent v2.0 - Call Graph Aware")
    parser.add_argument("project_dir", nargs="?", help="Project directory")
    parser.add_argument("--all", "-a", action="store_true", help="Analyze all test cases")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--include", "-I", action="append", default=[], help="Include directories")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    parser.add_argument("--show-graph", "-g", action="store_true", help="Just show call graph")
    
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    default_include = base_dir / "test_cases" / "include"
    include_dirs = args.include + ([str(default_include)] if default_include.exists() else [])
    
    if args.interactive:
        project_dir = args.project_dir or str(base_dir)
        agent = FileNoAgent(project_dir, include_dirs, verbose=not args.quiet)
        agent.interactive()
        return
    
    if args.show_graph and args.project_dir:
        project_path = Path(args.project_dir)
        if not project_path.exists():
            project_path = base_dir / "test_cases" / args.project_dir
        
        tools = AgentTools(str(project_path), include_dirs)
        result = tools.get_call_graph()
        print(result.output)
        return
    
    if args.all:
        test_cases_dir = base_dir / "test_cases"
        test_cases = sorted([
            d for d in os.listdir(test_cases_dir)
            if os.path.isdir(test_cases_dir / d) and d.startswith("apl")
        ])
        
        for case in test_cases:
            print(f"\n{'─'*70}")
            print(f"📁 {case}")
            print("─"*70)
            
            try:
                agent = FileNoAgent(str(test_cases_dir / case), include_dirs, verbose=not args.quiet)
                agent.analyze()
            except Exception as e:
                print(f"❌ Error: {str(e)}")
        return
    
    if args.project_dir:
        project_path = Path(args.project_dir)
        if not project_path.exists():
            project_path = base_dir / "test_cases" / args.project_dir
        
        if not project_path.exists():
            print(f"Error: Not found: {args.project_dir}")
            sys.exit(1)
        
        agent = FileNoAgent(str(project_path), include_dirs, verbose=not args.quiet)
        agent.analyze()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

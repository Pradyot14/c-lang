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
    from openai import OpenAI
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
        
        # Extract function definitions
        for filepath, content in self.files.items():
            if filepath.endswith('.c'):
                self._extract_functions(filepath, content)
        
        # Extract function declarations (prototypes)
        for filepath, content in self.files.items():
            self._extract_declarations(filepath, content)
        
        # Build call graph (with declaration checking)
        for func_name, func_data in self.functions.items():
            self._build_calls(func_name, func_data['file'], func_data['body'])
        
        # Find what's reachable from main
        self._find_reachable_from_main()
        
        return self
    
    def _extract_includes(self, filepath: str, content: str):
        """Extract #include statements to understand visibility"""
        self.includes[filepath] = set()
        
        for match in re.finditer(r'#\s*include\s*[<"]([^>"]+)[>"]', content):
            include_name = match.group(1)
            self.includes[filepath].add(include_name)
    
    def _extract_declarations(self, filepath: str, content: str):
        """Extract function prototypes (declarations)"""
        if filepath not in self.declarations:
            self.declarations[filepath] = {}
        
        # Remove block comments
        clean = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        # Remove line comments
        clean = re.sub(r'//.*$', '', clean, flags=re.MULTILINE)
        
        # Pattern for function prototype: return_type func_name(params);
        # Must end with ; (not {) to distinguish from definitions
        proto_pattern = r'(?:extern\s+)?(?:static\s+)?(?:inline\s+)?(\w+(?:\s*\*)?)\s+(\w+)\s*\(([^)]*)\)\s*;'
        
        for match in re.finditer(proto_pattern, clean):
            func_name = match.group(2)
            if func_name not in self.declarations[filepath]:
                self.declarations[filepath][func_name] = set()
            self.declarations[filepath][func_name].add(f"prototype in {os.path.basename(filepath)}")
        
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
        # Pattern for function definition
        func_pattern = r'(?:static\s+)?(?:inline\s+)?(\w+(?:\s*\*)?)\s+(\w+)\s*\(([^)]*)\)\s*\{'
        
        for match in re.finditer(func_pattern, content):
            func_name = match.group(2)
            
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
                # But ONLY add it if the function is DECLARED/VISIBLE in this file
                if called in self.functions:
                    # Check if it's declared in the caller's file (prototype or via includes)
                    visible_funcs = self._get_visible_declarations(caller_file)
                    if called in visible_funcs:
                        self.call_graph[func_name].add(called)
                    # If not visible, we skip it - it's an undeclared call (compile error)
    
    def _find_reachable_from_main(self):
        """BFS from main() to find all reachable functions"""
        if 'main' not in self.functions:
            return
        
        visited = set()
        queue = ['main']
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            
            # Add all functions this one calls
            for called in self.call_graph.get(current, []):
                if called in self.functions and called not in visited:
                    queue.append(called)
        
        self.reachable_from_main = visited
    
    def is_reachable(self, func_name: str) -> bool:
        """Check if a function is reachable from main()"""
        return func_name in self.reachable_from_main
    
    def get_call_chain(self, target_func: str) -> List[str]:
        """Get the call chain from main() to target function"""
        if 'main' not in self.functions or target_func not in self.functions:
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
                if called in self.functions:
                    new_path = tuple(path + [called])
                    if new_path not in visited:
                        visited[new_path] = called
                        queue.append((path + [called], called))
        
        return []
    
    def get_reachable_mfs_calls(self) -> List[Dict]:
        """Find all mpf_mfs_open calls that are reachable from main()"""
        calls = []
        
        for func_name in self.reachable_from_main:
            if func_name not in self.functions:
                continue
                
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
                            'reachable': True
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
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        
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
        
        self.system_prompt = """You are an expert C code analyzer. Your task is to find the numeric value of the 3rd argument to mpf_mfs_open() calls.

CRITICAL: You must ONLY analyze code that is REACHABLE from main(). If a function containing mpf_mfs_open() is not called from main(), its calls should be IGNORED.

## METHODOLOGY
1. First, use get_call_graph to see what's reachable from main()
2. Use find_mfs_open_calls to find ONLY reachable calls
3. For each call, trace the 3rd argument to its value
4. Use tools to verify - never guess

## IMPORTANT
- If find_mfs_open_calls returns no results, it means either:
  a) There are no mpf_mfs_open calls in the project, or
  b) The functions containing mpf_mfs_open are not called from main()
- Only report values for REACHABLE calls
- Show the call chain from main() to the call

## OUTPUT FORMAT
FINAL ANSWER:
- File: <filename>
- Function: <function_name>
- Call Chain: main() → ... → <function>
- Line: <line_number>
- 3rd Argument: <raw_argument>
- Resolved Value: <numeric_value or UNDEFINED or NOT_REACHABLE>"""
    
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
        
        # Show reachability info upfront
        self.log(f"\n📊 Reachable functions from main(): {sorted(self.tools.call_graph.reachable_from_main)}")
        
        max_iterations = 25
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            self.log(f"\n{'─'*50}")
            self.log(f"🔄 Iteration {iteration}")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system_prompt}] + self.conversation_history,
                    tools=self.tool_definitions,
                    tool_choice="auto",
                    temperature=0
                )
            except Exception as e:
                return f"LLM Error: {str(e)}"
            
            message = response.choices[0].message
            
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
                self.log(f"\n{'='*70}")
                self.log("✅ AGENT COMPLETE")
                self.log("="*70)
                self.log(f"\n{message.content}")
                return message.content
        
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

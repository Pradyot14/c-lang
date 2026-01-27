#!/usr/bin/env python3
"""
FileNo Agent v3.0 - Robust Multi-File Call Graph Analysis
==========================================================
FIXES:
1. Finds ALL functions across ALL .c files
2. Builds complete call graph without strict declaration checking
3. Traces ALL paths from main() to mpf_mfs_open()
4. Handles cross-file function calls properly
"""

import os
import re
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime

# Load environment
def load_env():
    env_paths = [
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
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
            except:
                pass
    return False

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
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
    success: bool
    output: str
    data: Any = None


class RobustCallGraphAnalyzer:
    """
    Robust call graph analyzer that finds ALL functions and ALL calls.
    No strict declaration checking - if it looks like a call, track it.
    """
    
    def __init__(self, project_dir: str, include_dirs: List[str] = None):
        self.project_dir = project_dir
        self.include_dirs = include_dirs or []
        
        self.files: Dict[str, str] = {}
        self.functions: Dict[str, Dict] = {}
        self.call_graph: Dict[str, Set[str]] = {}
        self.reverse_graph: Dict[str, Set[str]] = {}
        self.reachable_from_main: Set[str] = set()
        self.macros: Dict[str, str] = {}
        
    def load_project(self):
        """Load all C/H files"""
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
        
        print(f"📂 Loaded {len(self.files)} files")
        
        # Extract everything
        self._extract_macros()
        self._extract_all_functions()
        self._build_call_graph()
        self._find_reachable_from_main()
        
        return self
    
    def _extract_macros(self):
        """Extract all #define macros"""
        for filepath, content in self.files.items():
            # Remove comments
            clean = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
            clean = re.sub(r'//.*$', '', clean, flags=re.MULTILINE)
            
            for line in clean.split('\n'):
                line = line.strip()
                match = re.match(r'#\s*define\s+(\w+)\s+(.+)', line)
                if match:
                    name = match.group(1)
                    value = match.group(2).strip()
                    if name and value and not name.endswith('_H'):
                        self.macros[name] = value
        
        print(f"📝 Found {len(self.macros)} macros")
    
    def _extract_all_functions(self):
        """Extract ALL function definitions from ALL .c files"""
        # C keywords that should NOT be function names
        skip_keywords = {
            'if', 'else', 'while', 'for', 'do', 'switch', 'case', 'default',
            'break', 'continue', 'return', 'goto', 'sizeof', 'typeof'
        }
        
        for filepath, content in self.files.items():
            if not filepath.endswith('.c'):
                continue
            
            # DON'T remove comments before pattern matching - use original content
            # We'll remove comments from the body AFTER extraction
            
            # Pattern for function definition
            # Handles: static, inline, const, unsigned, struct X*, etc.
            func_pattern = r'''
                (?:extern\s+)?
                (?:static\s+)?
                (?:inline\s+)?
                (?:const\s+)?
                (?:unsigned\s+|signed\s+)?
                (?:long\s+)?
                (?:struct\s+\w+\s*\*?\s*|enum\s+\w+\s*|\w+)
                (?:\s*\*)*
                \s+
                (\w+)                       # Function name (group 1)
                \s*\(
                ([^)]*)                     # Parameters (group 2)
                \)\s*\{
            '''
            
            for match in re.finditer(func_pattern, content, re.VERBOSE):
                func_name = match.group(1).strip()
                params_str = match.group(2).strip()
                
                # Skip keywords
                if func_name in skip_keywords:
                    continue
                
                # Extract return type from the full match
                full_match = match.group(0)
                ret_type_match = re.match(r'(.+?)\s+' + re.escape(func_name), full_match.replace('\n', ' '))
                ret_type = ret_type_match.group(1).strip() if ret_type_match else 'int'
                
                # Find function body using brace counting from ORIGINAL content
                start = match.end() - 1  # Position of opening {
                brace_count = 1
                end = start + 1
                
                while end < len(content) and brace_count > 0:
                    if content[end] == '{':
                        brace_count += 1
                    elif content[end] == '}':
                        brace_count -= 1
                    end += 1
                
                func_body = content[start:end]
                
                # NOW remove comments from the body
                func_body_clean = re.sub(r'/\*.*?\*/', ' ', func_body, flags=re.DOTALL)
                func_body_clean = re.sub(r'//.*$', '', func_body_clean, flags=re.MULTILINE)
                
                line_num = content[:match.start()].count('\n') + 1
                
                # Parse parameters
                params = self._parse_params(params_str)
                
                # Store function (don't overwrite if already found)
                if func_name not in self.functions:
                    self.functions[func_name] = {
                        'file': filepath,
                        'body': func_body_clean,  # Use cleaned body
                        'line': line_num,
                        'ret_type': ret_type,
                        'params': params,
                        'signature': f"{ret_type} {func_name}({params_str})"
                    }
        
        print(f"📦 Found {len(self.functions)} functions: {list(self.functions.keys())}")
    
    def _parse_params(self, params_str: str) -> List[str]:
        """Parse parameter names from parameter string"""
        if not params_str or params_str.strip() == 'void':
            return []
        
        params = []
        for p in params_str.split(','):
            p = p.strip()
            if p:
                # Get last word as parameter name
                parts = p.replace('*', ' ').split()
                if parts:
                    name = parts[-1].strip()
                    if name and name != 'void':
                        params.append(name)
        return params
    
    def _build_call_graph(self):
        """Build call graph - find ALL function calls in ALL functions"""
        # Keywords to skip
        skip_keywords = {
            'if', 'while', 'for', 'switch', 'return', 'sizeof', 'typeof', 'defined',
            'case', 'default', 'goto', 'break', 'continue', 'else', 'do'
        }
        
        # Common library functions to skip
        skip_stdlib = {
            'printf', 'sprintf', 'fprintf', 'snprintf', 'scanf', 'sscanf',
            'malloc', 'calloc', 'realloc', 'free',
            'memset', 'memcpy', 'memmove', 'memcmp',
            'strcpy', 'strncpy', 'strlen', 'strcmp', 'strncmp', 'strcat', 'strstr',
            'fopen', 'fclose', 'fread', 'fwrite', 'fseek', 'ftell', 'fgets',
            'exit', 'abort', 'assert', 'perror', 'atoi', 'atof', 'strtol'
        }
        
        # Type keywords that indicate declarations
        type_keywords = {
            'int', 'void', 'char', 'short', 'long', 'float', 'double',
            'unsigned', 'signed', 'static', 'extern', 'const', 'struct', 
            'enum', 'union', 'typedef', 'inline', 'register', 'volatile',
            'size_t', 'ssize_t', 'int8_t', 'int16_t', 'int32_t', 'int64_t',
            'uint8_t', 'uint16_t', 'uint32_t', 'uint64_t'
        }
        
        for func_name, func_data in self.functions.items():
            self.call_graph[func_name] = set()
            body = func_data['body']
            
            # Remove strings and comments
            clean = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', body)
            clean = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", clean)
            clean = re.sub(r'/\*.*?\*/', '', clean, flags=re.DOTALL)
            clean = re.sub(r'//.*$', '', clean, flags=re.MULTILINE)
            
            # Process line by line for better context
            for line in clean.split('\n'):
                line_stripped = line.strip()
                
                # Skip empty lines
                if not line_stripped:
                    continue
                
                # Skip lines that are clearly declarations/prototypes
                # Pattern: extern ... name(...);
                if re.match(r'^\s*extern\s+', line_stripped):
                    continue
                
                # Skip function prototypes: type name(params);
                if re.match(r'^(static\s+|inline\s+|extern\s+)*(const\s+)?(unsigned\s+|signed\s+)?(int|void|char|short|long|float|double|struct\s+\w+|enum\s+\w+|\w+_t)\s*\*?\s*\w+\s*\([^)]*\)\s*;', line_stripped):
                    continue
                
                # Find function calls in this line
                for match in re.finditer(r'\b([a-zA-Z_]\w*)\s*\(', line):
                    called = match.group(1)
                    
                    # Skip keywords and stdlib
                    if called in skip_keywords or called in skip_stdlib:
                        continue
                    
                    # Skip if it's the same function (recursion is fine, but we need real calls)
                    # Actually, keep recursion - it's valid
                    
                    # Check context before the call
                    pos = match.start()
                    before = line[:pos].strip()
                    
                    # Skip if preceded by type keyword (declaration pattern)
                    is_declaration = False
                    if before:
                        words = before.split()
                        if words:
                            last_word = words[-1].rstrip('*')
                            if last_word in type_keywords:
                                is_declaration = True
                            # Also check for _t suffix (custom types)
                            if last_word.endswith('_t'):
                                is_declaration = True
                    
                    # Check if the line looks like a standalone declaration
                    # e.g., "int func(int x);" - note the semicolon after )
                    if not is_declaration:
                        # Find the closing paren and check what's after
                        after_match = line[match.end()-1:]  # start from the (
                        paren_depth = 1
                        idx = 1
                        while idx < len(after_match) and paren_depth > 0:
                            if after_match[idx] == '(':
                                paren_depth += 1
                            elif after_match[idx] == ')':
                                paren_depth -= 1
                            idx += 1
                        
                        # Check what comes after the closing paren
                        if idx < len(after_match):
                            after_paren = after_match[idx:].strip()
                            # If it's just a semicolon (and nothing else meaningful before it)
                            # AND there's a type before, it's likely a prototype
                            if after_paren.startswith(';'):
                                # Check if entire line looks like: type name(...);
                                if before and any(w.rstrip('*') in type_keywords or w.endswith('_t') for w in before.split()):
                                    is_declaration = True
                    
                    if not is_declaration and called in self.functions:
                        self.call_graph[func_name].add(called)
                        
                        # Build reverse graph
                        if called not in self.reverse_graph:
                            self.reverse_graph[called] = set()
                        self.reverse_graph[called].add(func_name)
        
        # Print call graph summary
        print(f"🔗 Call graph built:")
        for fn, calls in self.call_graph.items():
            known_calls = [c for c in calls if c in self.functions]
            if known_calls:
                print(f"   {fn}() → {known_calls}")
    
    def _find_reachable_from_main(self):
        """BFS from main() to find all reachable functions"""
        if 'main' not in self.functions:
            print("⚠️  No main() found - marking all functions as reachable")
            self.reachable_from_main = set(self.functions.keys())
            return
        
        visited = set()
        queue = ['main']
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            
            # Add all functions this one calls (that we know about)
            for called in self.call_graph.get(current, []):
                if called in self.functions and called not in visited:
                    queue.append(called)
        
        self.reachable_from_main = visited
        print(f"✅ Reachable from main(): {sorted(self.reachable_from_main)}")
    
    def is_reachable(self, func_name: str) -> bool:
        return func_name in self.reachable_from_main
    
    def get_all_paths_to_function(self, target: str, max_paths: int = 10) -> List[List[str]]:
        """Find ALL paths from main() to target function"""
        if 'main' not in self.functions:
            return []
        
        paths = []
        
        def dfs(current: str, path: List[str], visited: Set[str]):
            if len(paths) >= max_paths:
                return
            
            if current == target:
                paths.append(path.copy())
                return
            
            for called in self.call_graph.get(current, []):
                if called not in visited and called in self.functions:
                    visited.add(called)
                    path.append(called)
                    dfs(called, path, visited)
                    path.pop()
                    visited.remove(called)
        
        dfs('main', ['main'], {'main'})
        return paths
    
    def get_mfs_calls(self) -> List[Dict]:
        """Find all mpf_mfs_open calls in reachable functions"""
        calls = []
        
        for func_name in self.reachable_from_main:
            if func_name not in self.functions:
                continue
            
            func_data = self.functions[func_name]
            body = func_data['body']
            filepath = func_data['file']
            
            # Find mpf_mfs_open calls
            lines = body.split('\n')
            for i, line in enumerate(lines):
                if 'mpf_mfs_open' in line:
                    # Get complete call (may span lines)
                    full_call = line
                    j = i
                    while full_call.count('(') > full_call.count(')') and j < len(lines) - 1:
                        j += 1
                        full_call += ' ' + lines[j].strip()
                    
                    match = re.search(r'mpf_mfs_open\s*\(([^)]+)\)', full_call)
                    if match:
                        args = self._parse_call_args(match.group(1))
                        
                        # Get ALL paths to this function
                        all_paths = self.get_all_paths_to_function(func_name)
                        if not all_paths:
                            all_paths = [[func_name]]
                        
                        calls.append({
                            'file': os.path.basename(filepath),
                            'filepath': filepath,
                            'function': func_name,
                            'line': func_data['line'] + i,
                            'call': match.group(0),
                            'args': args,
                            'arg3': args[2] if len(args) > 2 else None,
                            'all_paths': all_paths,
                            'call_chain': all_paths[0] if all_paths else [func_name]
                        })
        
        return calls
    
    def _parse_call_args(self, args_str: str) -> List[str]:
        """Parse function call arguments"""
        args = []
        current = ""
        depth = 0
        
        for char in args_str:
            if char == ',' and depth == 0:
                args.append(current.strip())
                current = ""
            else:
                if char in '([':
                    depth += 1
                elif char in ')]':
                    depth -= 1
                current += char
        
        if current.strip():
            args.append(current.strip())
        
        return args


class AgentTools:
    """Tools for the agent"""
    
    def __init__(self, project_dir: str, include_dirs: List[str] = None):
        self.project_dir = project_dir
        self.include_dirs = include_dirs or []
        
        self.call_graph = RobustCallGraphAnalyzer(project_dir, include_dirs)
        self.call_graph.load_project()
    
    def read_file(self, path: str) -> ToolResult:
        """Read a file"""
        try:
            for test_path in [
                path,
                os.path.join(self.project_dir, path),
                os.path.join(self.project_dir, os.path.basename(path))
            ] + [os.path.join(d, path) for d in self.include_dirs] + \
              [os.path.join(d, os.path.basename(path)) for d in self.include_dirs]:
                if os.path.exists(test_path):
                    with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    lines = content.split('\n')
                    numbered = '\n'.join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
                    return ToolResult(True, f"File: {test_path}\n\n{numbered}", content)
            
            return ToolResult(False, f"File not found: {path}")
        except Exception as e:
            return ToolResult(False, f"Error: {str(e)}")
    
    def list_files(self) -> ToolResult:
        files = list(self.call_graph.files.keys())
        rel_files = [os.path.relpath(f, self.project_dir) for f in files]
        return ToolResult(True, f"Files:\n" + '\n'.join(f"  • {f}" for f in sorted(rel_files)), rel_files)
    
    def find_macro(self, macro_name: str) -> ToolResult:
        if macro_name in self.call_graph.macros:
            return ToolResult(True, f"#define {macro_name} {self.call_graph.macros[macro_name]}", 
                            {'name': macro_name, 'value': self.call_graph.macros[macro_name]})
        return ToolResult(False, f"Macro '{macro_name}' not found")
    
    def resolve_macro(self, macro_name: str, depth: int = 0) -> ToolResult:
        if depth > 20:
            return ToolResult(False, f"Max depth for '{macro_name}'")
        
        if macro_name not in self.call_graph.macros:
            try:
                if macro_name.startswith('0x'):
                    return ToolResult(True, f"{macro_name} = {int(macro_name, 16)}", int(macro_name, 16))
                return ToolResult(True, f"{macro_name} = {int(macro_name)}", int(macro_name))
            except:
                return ToolResult(False, f"Macro '{macro_name}' not found")
        
        value = self.call_graph.macros[macro_name]
        
        while value.startswith('(') and value.endswith(')'):
            value = value[1:-1].strip()
        
        try:
            if value.startswith('0x'):
                num = int(value, 16)
            else:
                num = int(value)
            return ToolResult(True, f"{macro_name} = {num}", num)
        except:
            pass
        
        if value in self.call_graph.macros:
            return self.resolve_macro(value, depth + 1)
        
        expr = value
        for m in self.call_graph.macros:
            if re.search(rf'\b{m}\b', expr):
                r = self.resolve_macro(m, depth + 1)
                if r.success and isinstance(r.data, int):
                    expr = re.sub(rf'\b{m}\b', str(r.data), expr)
        
        try:
            expr_clean = re.sub(r'[^\d+\-*/().\s]', '', expr)
            if expr_clean.strip():
                result = eval(expr_clean)
                return ToolResult(True, f"{macro_name} = {value} = {int(result)}", int(result))
        except:
            pass
        
        return ToolResult(False, f"Cannot resolve '{macro_name}' = '{value}'")
    
    def find_function(self, func_name: str) -> ToolResult:
        if func_name not in self.call_graph.functions:
            return ToolResult(False, f"Function '{func_name}' not found")
        
        func_data = self.call_graph.functions[func_name]
        reachable = self.call_graph.is_reachable(func_name)
        all_paths = self.call_graph.get_all_paths_to_function(func_name)
        
        lines = func_data['body'].split('\n')
        numbered = '\n'.join(f"{func_data['line'] + i:4d} | {line}" for i, line in enumerate(lines))
        
        status = "✅ REACHABLE" if reachable else "❌ NOT REACHABLE"
        paths_str = '\n'.join(' → '.join(p) for p in all_paths[:5]) if all_paths else "N/A"
        
        output = f"""Function: {func_data['signature']}
File: {os.path.basename(func_data['file'])}
Status: {status}
Paths from main():
{paths_str}

{numbered}"""
        
        return ToolResult(True, output, {
            'name': func_name,
            'body': func_data['body'],
            'file': func_data['file'],
            'reachable': reachable,
            'all_paths': all_paths
        })
    
    def get_call_graph(self) -> ToolResult:
        output = "Call Graph from main():\n" + "=" * 50 + "\n\n"
        
        def print_tree(func: str, indent: int = 0, visited: Set[str] = None):
            nonlocal output
            if visited is None:
                visited = set()
            
            if func in visited:
                output += "  " * indent + f"├─ {func} (recursive)\n"
                return
            
            visited.add(func)
            marker = "✓" if func in self.call_graph.reachable_from_main else "✗"
            output += "  " * indent + f"├─ [{marker}] {func}()\n"
            
            for called in sorted(self.call_graph.call_graph.get(func, [])):
                if called in self.call_graph.functions:
                    print_tree(called, indent + 1, visited.copy())
        
        if 'main' in self.call_graph.functions:
            print_tree('main')
        else:
            output += "⚠️ No main() found\n"
        
        output += f"\n\nReachable: {sorted(self.call_graph.reachable_from_main)}"
        output += f"\nAll functions: {sorted(self.call_graph.functions.keys())}"
        
        return ToolResult(True, output)
    
    def find_mfs_open_calls(self) -> ToolResult:
        calls = self.call_graph.get_mfs_calls()
        
        if not calls:
            return ToolResult(True, "No mpf_mfs_open calls found in reachable functions", [])
        
        output = f"Found {len(calls)} mpf_mfs_open calls:\n"
        for c in calls:
            output += f"\n📍 {c['file']}:{c['line']} in {c['function']}()\n"
            output += f"   Call: {c['call']}\n"
            output += f"   3rd arg: {c['arg3']}\n"
            output += f"   Paths:\n"
            for path in c['all_paths'][:3]:
                output += f"      • {' → '.join(path)}\n"
        
        return ToolResult(True, output, calls)
    
    def trace_variable(self, var_name: str, func_name: str) -> ToolResult:
        if func_name not in self.call_graph.functions:
            return ToolResult(False, f"Function '{func_name}' not found")
        
        func_body = self.call_graph.functions[func_name]['body']
        
        assignments = []
        patterns = [
            rf'\b{re.escape(var_name)}\s*=\s*([^;]+);',
            rf'\b(?:int|char|long)\s+{re.escape(var_name)}\s*=\s*([^;]+);',
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
            params = self.call_graph.functions[func_name].get('params', [])
            if var_name in params:
                return ToolResult(True, f"'{var_name}' is a PARAMETER of {func_name}(). Trace callers to find value.", 
                                {'is_param': True, 'param_index': params.index(var_name)})
            return ToolResult(False, f"No assignments to '{var_name}' in '{func_name}'")
        
        output = f"Assignments to '{var_name}' in '{func_name}':\n"
        for a in assignments:
            output += f"  Line {a['line']}: {a['code']}\n"
        
        return ToolResult(True, output, assignments)
    
    def evaluate_expression(self, expression: str) -> ToolResult:
        try:
            expr = expression
            
            for macro in self.call_graph.macros:
                if re.search(rf'\b{macro}\b', expr):
                    r = self.resolve_macro(macro)
                    if r.success and isinstance(r.data, int):
                        expr = re.sub(rf'\b{macro}\b', str(r.data), expr)
            
            expr_clean = re.sub(r'0x([0-9a-fA-F]+)', lambda m: str(int(m.group(1), 16)), expr)
            expr_clean = re.sub(r'[^\d+\-*/()%\s]', '', expr_clean)
            
            if expr_clean.strip():
                result = eval(expr_clean)
                return ToolResult(True, f"{expression} = {int(result)}", int(result))
            
            return ToolResult(False, f"Cannot evaluate: {expression}")
        except Exception as e:
            return ToolResult(False, f"Error: {str(e)}")


class FileNoAgentV3:
    """FileNo Agent v3 - Robust multi-file analysis"""
    
    def __init__(self, project_dir: str, include_dirs: List[str] = None, verbose: bool = True):
        self.project_dir = project_dir
        self.include_dirs = include_dirs or []
        self.verbose = verbose
        
        self.tools = AgentTools(project_dir, include_dirs)
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required")
        
        # Azure OpenAI config (check first, like agent.py)
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
        
        self.tool_registry = {
            "read_file": self.tools.read_file,
            "list_files": lambda: self.tools.list_files(),
            "find_macro": self.tools.find_macro,
            "resolve_macro": self.tools.resolve_macro,
            "find_function": self.tools.find_function,
            "get_call_graph": lambda: self.tools.get_call_graph(),
            "find_mfs_open_calls": lambda: self.tools.find_mfs_open_calls(),
            "trace_variable": self.tools.trace_variable,
            "evaluate_expression": self.tools.evaluate_expression,
        }
        
        self.tool_definitions = [
            {"type": "function", "function": {"name": "read_file", "description": "Read a source file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "list_files", "description": "List all files", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "find_macro", "description": "Find macro definition", "parameters": {"type": "object", "properties": {"macro_name": {"type": "string"}}, "required": ["macro_name"]}}},
            {"type": "function", "function": {"name": "resolve_macro", "description": "Resolve macro to number", "parameters": {"type": "object", "properties": {"macro_name": {"type": "string"}}, "required": ["macro_name"]}}},
            {"type": "function", "function": {"name": "find_function", "description": "Find function with all paths from main()", "parameters": {"type": "object", "properties": {"func_name": {"type": "string"}}, "required": ["func_name"]}}},
            {"type": "function", "function": {"name": "get_call_graph", "description": "Show complete call graph from main()", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "find_mfs_open_calls", "description": "Find all mpf_mfs_open calls with ALL paths", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "trace_variable", "description": "Trace variable assignments", "parameters": {"type": "object", "properties": {"var_name": {"type": "string"}, "func_name": {"type": "string"}}, "required": ["var_name", "func_name"]}}},
            {"type": "function", "function": {"name": "evaluate_expression", "description": "Evaluate arithmetic expression", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
        ]
        
        self.system_prompt = """You are an expert C code analyzer. Find the EXACT numeric value of the 3rd argument to mpf_mfs_open().

METHODOLOGY:
1. Use get_call_graph to see ALL functions reachable from main()
2. Use find_mfs_open_calls to find ALL mpf_mfs_open calls with their paths
3. For each call's 3rd argument:
   - If macro: use resolve_macro
   - If variable: use trace_variable, then resolve the value
   - If expression: use evaluate_expression
   - If parameter: trace back through the call chain to find the value passed

IMPORTANT: A function may be reached via MULTIPLE paths with DIFFERENT argument values. Report ALL possible values.

OUTPUT FORMAT:
For each mpf_mfs_open call:
- File: <file>
- Function: <function>
- Line: <line>
- All Paths: <list all paths from main()>
- 3rd Argument: <raw arg>
- Resolved Values: <all possible numeric values>"""
    
    def execute_tool(self, name: str, args: Dict) -> str:
        if name not in self.tool_registry:
            return f"Unknown tool: {name}"
        try:
            func = self.tool_registry[name]
            if callable(func):
                if args:
                    result = func(**args)
                else:
                    result = func()
            else:
                result = func
            return result.output if hasattr(result, 'output') else str(result)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def analyze(self, task: str = None) -> str:
        if task is None:
            task = f"Analyze {self.project_dir}. Find ALL mpf_mfs_open calls reachable from main() and resolve their 3rd argument values."
        
        messages = [{"role": "user", "content": task}]
        
        print(f"\n{'='*60}")
        print("🤖 Agent v3.0 Starting")
        print('='*60)
        
        for i in range(25):
            print(f"\n--- Iteration {i+1} ---")
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system_prompt}] + messages,
                    tools=self.tool_definitions,
                    tool_choice="auto",
                    temperature=0
                )
            except Exception as e:
                return f"API Error: {str(e)}"
            
            msg = response.choices[0].message
            
            if msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content, "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]})
                
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    print(f"🔧 {name}({args})")
                    
                    result = self.execute_tool(name, args)
                    print(f"   → {result[:200]}..." if len(result) > 200 else f"   → {result}")
                    
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                print(f"\n{'='*60}")
                print("✅ Analysis Complete")
                print('='*60)
                print(msg.content)
                return msg.content
        
        return "Max iterations reached"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="FileNo Agent v3.0")
    parser.add_argument("project_dir", nargs="?", help="Project directory")
    parser.add_argument("--include", "-I", action="append", default=[], help="Include dirs")
    parser.add_argument("--graph", "-g", action="store_true", help="Show call graph only")
    
    args = parser.parse_args()
    
    if not args.project_dir:
        parser.print_help()
        return
    
    base_dir = Path(__file__).parent
    project_path = Path(args.project_dir)
    if not project_path.exists():
        project_path = base_dir / "test_cases" / args.project_dir
    
    include_dirs = args.include + [str(base_dir / "test_cases" / "include")]
    
    if args.graph:
        tools = AgentTools(str(project_path), include_dirs)
        print(tools.get_call_graph().output)
        return
    
    agent = FileNoAgentV3(str(project_path), include_dirs)
    agent.analyze()


if __name__ == "__main__":
    main()

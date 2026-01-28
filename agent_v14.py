#!/usr/bin/env python3
"""
FileNo Agent v14.0 - Tool-Based LLM Analyzer with Full Tracing
===============================================================
Shows:
1. All traced paths from entry to target
2. LLM iterations with tool calls step by step
3. Full reasoning chain for each resolution
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
from enum import Enum


# ============================================================================
# COLORS FOR TERMINAL OUTPUT
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def colored(text: str, color: str) -> str:
    return f"{color}{text}{Colors.ENDC}"


# ============================================================================
# ENVIRONMENT
# ============================================================================

def load_env():
    for p in [Path(__file__).parent / ".env", Path.cwd() / ".env"]:
        if p.exists():
            for line in open(p):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = v.strip().strip('"\'')

load_env()

try:
    from openai import OpenAI, AzureOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("⚠️  pip install openai")

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class CallType(Enum):
    DIRECT = "direct"
    MACRO = "macro"
    POINTER = "pointer"
    CALLBACK = "callback"
    ENTRY = "entry"


@dataclass
class FunctionDef:
    name: str
    file: str
    line: int
    body: str
    params: List[str] = field(default_factory=list)
    calls: Set[str] = field(default_factory=set)


@dataclass
class MacroDef:
    name: str
    file: str
    body: str
    params: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)


@dataclass
class PathStep:
    name: str
    type: CallType
    file: Optional[str] = None


@dataclass
class MFSCallSite:
    file: str
    line: int
    function: str
    full_call: str
    args: List[str]
    path: List[PathStep]
    fileno_raw: str = ""
    fileno_resolved: str = ""


# ============================================================================
# C PARSER
# ============================================================================

class CParser:
    KEYWORDS = frozenset([
        'if', 'else', 'while', 'for', 'do', 'switch', 'case', 'default',
        'break', 'continue', 'return', 'goto', 'sizeof', 'typeof',
        'struct', 'union', 'enum', 'typedef', 'extern', 'static',
        'void', 'int', 'char', 'short', 'long', 'float', 'double',
        'signed', 'unsigned', 'NULL', 'true', 'false', 'const',
        'printf', 'fprintf', 'malloc', 'free', 'memcpy', 'memset',
        'strlen', 'strcpy', 'strcmp', 'fopen', 'fclose'
    ])
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.files: Dict[str, str] = {}
        self.functions: Dict[str, FunctionDef] = {}
        self.macros: Dict[str, MacroDef] = {}
        self.value_macros: Dict[str, str] = {}
        self.func_pointers: Dict[str, Set[str]] = defaultdict(set)
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
    
    def clean_code(self, code: str) -> str:
        code = re.sub(r'/\*[\s\S]*?\*/', ' ', code)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', code)
        code = re.sub(r"'(?:[^'\\]|\\.)*'", "''", code)
        return code
    
    def is_valid_func(self, name: str) -> bool:
        return name not in self.KEYWORDS and len(name) >= 2 and not name[0].isdigit()
    
    def find_calls(self, code: str) -> List[str]:
        calls = []
        for m in re.finditer(r'\b([a-zA-Z_]\w*)\s*\(', code):
            if self.is_valid_func(m.group(1)):
                calls.append(m.group(1))
        return list(dict.fromkeys(calls))
    
    def parse_macros(self, code: str, filename: str):
        code_exp = re.sub(r'\\\n', ' ', code)
        
        for line in code_exp.split('\n'):
            line = line.strip()
            if not line.startswith('#'):
                continue
            
            # Function-like: #define NAME(args) body (NO space before paren)
            func_match = re.match(r'#\s*define\s+(\w+)\(([^)]*)\)\s*(.*)', line)
            if func_match:
                name = func_match.group(1)
                params = [p.strip() for p in func_match.group(2).split(',') if p.strip()]
                body = re.sub(r'/\*.*?\*/', '', func_match.group(3))
                body = re.sub(r'//.*$', '', body).strip()
                calls = self.find_calls(body)
                self.macros[name] = MacroDef(name, filename, body, params, calls)
                continue
            
            # Value macro
            val_match = re.match(r'#\s*define\s+(\w+)\s+(.+)', line)
            if val_match and not re.match(r'#\s*define\s+\w+\(', line):
                name = val_match.group(1)
                value = re.sub(r'/\*.*?\*/', '', val_match.group(2))
                value = re.sub(r'//.*$', '', value).strip()
                if value:
                    self.value_macros[name] = value
    
    def parse_func_pointers(self, code: str):
        clean = self.clean_code(code)
        for pat in [r'\.(\w+)\s*=\s*&?(\w+)\s*[,;}\)]', r'->(\w+)\s*=\s*&?(\w+)\s*[,;}\)]']:
            for m in re.finditer(pat, clean):
                if self.is_valid_func(m.group(2)):
                    self.func_pointers[m.group(1)].add(m.group(2))
    
    def find_brace_end(self, code: str, start: int) -> int:
        depth, i = 1, start
        while i < len(code) and depth > 0:
            if code[i] == '{': depth += 1
            elif code[i] == '}': depth -= 1
            i += 1
        return i if depth == 0 else -1
    
    def parse_functions(self, code: str, filename: str):
        clean = self.clean_code(code)
        
        for m in re.finditer(r'\b([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*\{', clean):
            name = m.group(1)
            if not self.is_valid_func(name):
                continue
            
            body_end = self.find_brace_end(clean, m.end())
            if body_end == -1:
                continue
            
            body = clean[m.end()-1:body_end]
            line = code[:m.start()].count('\n') + 1
            
            params = []
            params_str = m.group(2).strip()
            if params_str and params_str != 'void':
                for p in params_str.split(','):
                    parts = p.strip().split()
                    if parts:
                        params.append(parts[-1].strip('*'))
            
            calls = set(self.find_calls(body))
            
            # Expand macro calls
            for call in list(calls):
                if call in self.macros:
                    for mc in self.macros[call].calls:
                        calls.add(mc)
                if call in self.func_pointers:
                    calls.update(self.func_pointers[call])
            
            if name not in self.functions or filename.endswith('.c'):
                self.functions[name] = FunctionDef(name, filename, line, body, params, calls)
    
    def load_file(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            filename = os.path.basename(filepath)
            self.files[filename] = content
            self.parse_macros(content, filename)
            self.parse_func_pointers(content)
            self.parse_functions(content, filename)
        except Exception as e:
            if self.verbose:
                print(f"❌ Error: {filepath}: {e}")
    
    def load_directory(self, dirpath: str):
        if not os.path.isdir(dirpath):
            return
        # Headers first
        for root, _, files in os.walk(dirpath):
            for f in files:
                if f.endswith(('.h', '.hpp', '.inc')):
                    self.load_file(os.path.join(root, f))
        # Then sources
        for root, _, files in os.walk(dirpath):
            for f in files:
                if f.endswith(('.c', '.cpp')):
                    self.load_file(os.path.join(root, f))
    
    def build_call_graph(self):
        # Detect callbacks
        self._detect_callbacks()
        for name, func in self.functions.items():
            self.call_graph[name] = func.calls
    
    def _detect_callbacks(self):
        """Detect function names passed as arguments (callbacks)"""
        all_func_names = set(self.functions.keys())
        for func_name, func in self.functions.items():
            for call_match in re.finditer(r'\b(\w+)\s*\(([^)]*)\)', func.body):
                args_str = call_match.group(2)
                for arg in args_str.split(','):
                    arg = arg.strip()
                    if arg in all_func_names and arg != func_name:
                        func.calls.add(arg)


# ============================================================================
# PATH FINDER
# ============================================================================

class PathFinder:
    def __init__(self, parser: CParser):
        self.parser = parser
    
    def find_paths(self, entry: str, target: str, max_depth: int = 30) -> List[List[PathStep]]:
        if entry not in self.parser.call_graph:
            return []
        
        paths = []
        
        def dfs(current: str, path: List[PathStep], visited: Set[str]):
            if len(paths) >= 100 or len(path) > max_depth or current in visited:
                return
            
            if current == target:
                paths.append(list(path))
                return
            
            visited.add(current)
            
            for callee in self.parser.call_graph.get(current, set()):
                func = self.parser.functions.get(callee)
                step = PathStep(callee, CallType.DIRECT, func.file if func else None)
                path.append(step)
                dfs(callee, path, visited)
                path.pop()
            
            visited.discard(current)
        
        entry_file = self.parser.functions.get(entry, FunctionDef(entry, "", 0, "")).file
        dfs(entry, [PathStep(entry, CallType.ENTRY, entry_file)], set())
        
        return paths
    
    def print_paths(self, paths: List[List[PathStep]], entry: str, target: str):
        """Print all found paths in a tree-like format"""
        print(colored(f"\n  🛤️  Found {len(paths)} path(s) from {entry}() to {target}():", Colors.CYAN))
        print()
        
        for i, path in enumerate(paths, 1):
            print(colored(f"  ╔══ Path {i} ══╗", Colors.YELLOW))
            
            for j, step in enumerate(path):
                is_last = (j == len(path) - 1)
                connector = "╚═▶" if is_last else "╠═▶"
                indent = "  ║  " * j
                
                file_info = colored(f" [{step.file}]", Colors.DIM) if step.file else ""
                
                if step.type == CallType.ENTRY:
                    func_str = colored(f"{step.name}()", Colors.GREEN + Colors.BOLD)
                elif is_last:
                    func_str = colored(f"{step.name}()", Colors.RED + Colors.BOLD)
                else:
                    func_str = colored(f"{step.name}()", Colors.BLUE)
                
                print(f"  {indent}{connector} {func_str}{file_info}")
            
            print()


# ============================================================================
# MFS EXTRACTOR
# ============================================================================

class MFSExtractor:
    def __init__(self, parser: CParser):
        self.parser = parser
    
    def extract_from_paths(self, paths: List[List[PathStep]]) -> List[MFSCallSite]:
        calls = []
        seen = set()
        
        for path in paths:
            if len(path) < 2 or path[-1].name != 'mpf_mfs_open':
                continue
            
            caller = path[-2]
            func = self.parser.functions.get(caller.name)
            if not func:
                continue
            
            pattern = r'mpf_mfs_open\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\)'
            
            for m in re.finditer(pattern, func.body):
                line = func.line + func.body[:m.start()].count('\n')
                key = f"{func.file}:{line}"
                
                if key in seen:
                    continue
                seen.add(key)
                
                args = [m.group(i).strip() for i in range(1, 7)]
                calls.append(MFSCallSite(
                    func.file, line, caller.name, m.group(0),
                    args, path[:-1], args[2] if len(args) > 2 else ""
                ))
        
        return calls
    
    def print_calls(self, calls: List[MFSCallSite]):
        """Print extracted MFS calls"""
        print(colored(f"\n  📍 Found {len(calls)} mpf_mfs_open call(s):", Colors.CYAN))
        print()
        
        for i, call in enumerate(calls, 1):
            print(colored(f"  ┌─ Call {i} ─┐", Colors.YELLOW))
            print(f"  │ File: {colored(call.file, Colors.BLUE)}:{call.line}")
            print(f"  │ Function: {colored(call.function + '()', Colors.GREEN)}")
            print(f"  │ 3rd Arg (fileno): {colored(call.fileno_raw, Colors.RED + Colors.BOLD)}")
            print(f"  │ Full: {call.full_call[:60]}...")
            print(f"  └{'─' * 20}┘")
            print()


# ============================================================================
# TOOL-BASED VALUE RESOLVER WITH DETAILED TRACING
# ============================================================================

class ToolBasedResolver:
    """Provides tools for LLM to resolve values step-by-step with full tracing."""
    
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_macro_value",
                "description": "Look up the value/definition of a C macro. Returns the macro body/value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "macro_name": {"type": "string", "description": "Name of the macro to look up"}
                    },
                    "required": ["macro_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_variable_assignment",
                "description": "Find what value a LOCAL variable is assigned to within a function.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {"type": "string", "description": "Name of the function"},
                        "variable_name": {"type": "string", "description": "Name of the variable"}
                    },
                    "required": ["function_name", "variable_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_function_params",
                "description": "Get the parameter names of a function.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {"type": "string", "description": "Name of the function"}
                    },
                    "required": ["function_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_call_sites",
                "description": "Find where a function is called and what arguments are passed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {"type": "string", "description": "Name of the function"}
                    },
                    "required": ["function_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "evaluate_expression",
                "description": "Evaluate a simple arithmetic expression with numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Expression like '9000 + 1'"}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "check_if_parameter",
                "description": "Check if a variable is a function parameter or local variable.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {"type": "string"},
                        "variable_name": {"type": "string"}
                    },
                    "required": ["function_name", "variable_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "find_macro_that_calls",
                "description": "Find macros that expand to call a specific function.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {"type": "string"}
                    },
                    "required": ["function_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_macro_call_sites",
                "description": "Find where a macro is invoked and what arguments are passed.",
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
                "name": "submit_answer",
                "description": "Submit the final resolved numeric value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "description": "The final number"},
                        "reasoning": {"type": "string", "description": "Brief explanation"}
                    },
                    "required": ["value"]
                }
            }
        }
    ]
    
    def __init__(self, parser: CParser, verbose: bool = True):
        self.parser = parser
        self.verbose = verbose
        self.client = None
        self.model = None
        self.provider = None  # 'openai', 'azure', 'groq'
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM client - tries Groq first, then OpenAI, then Azure"""
        
        # Try Groq first (fast and free tier available)
        if HAS_GROQ:
            groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("groq")
            if groq_key:
                try:
                    self.client = Groq(api_key=groq_key)
                    self.model = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
                    self.provider = "groq"
                    if self.verbose:
                        print(colored(f"    🚀 Using Groq ({self.model})", Colors.GREEN))
                    return
                except Exception as e:
                    if self.verbose:
                        print(colored(f"    ⚠️ Groq init failed: {e}", Colors.YELLOW))
        
        if not HAS_OPENAI:
            return
        
        # Try Azure OpenAI
        try:
            ae = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
            ak = os.environ.get("AZURE_OPENAI_API_KEY", "")
            ad = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
            
            if "your-resource" not in ae and all([ae, ak, ad]):
                self.client = AzureOpenAI(
                    azure_endpoint=ae, api_key=ak,
                    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
                )
                self.model = ad
                self.provider = "azure"
                if self.verbose:
                    print(colored(f"    🔷 Using Azure OpenAI ({self.model})", Colors.BLUE))
                return
        except Exception as e:
            if self.verbose:
                print(colored(f"    ⚠️ Azure init failed: {e}", Colors.YELLOW))
        
        # Try OpenAI
        try:
            ok = os.environ.get("OPENAI_API_KEY", "")
            if ok:
                self.client = OpenAI(api_key=ok)
                self.model = os.environ.get("OPENAI_MODEL", "gpt-4o")
                self.provider = "openai"
                if self.verbose:
                    print(colored(f"    🟢 Using OpenAI ({self.model})", Colors.GREEN))
                return
        except Exception as e:
            if self.verbose:
                print(colored(f"    ⚠️ OpenAI init failed: {e}", Colors.YELLOW))
    
    # ==================== TOOL IMPLEMENTATIONS ====================
    
    def tool_get_macro_value(self, macro_name: str) -> str:
        if macro_name in self.parser.value_macros:
            return self.parser.value_macros[macro_name]
        if macro_name in self.parser.macros:
            m = self.parser.macros[macro_name]
            return f"FUNCTION_MACRO({', '.join(m.params)}): {m.body}"
        return "NOT_FOUND"
    
    def tool_get_variable_assignment(self, function_name: str, variable_name: str) -> str:
        func = self.parser.functions.get(function_name)
        if not func:
            return f"FUNCTION_NOT_FOUND: {function_name}"
        
        matches = list(re.finditer(rf'\b{re.escape(variable_name)}\s*=\s*([^;]+);', func.body))
        if matches:
            return matches[-1].group(1).strip()
        
        decl_match = re.search(rf'(?:int|long|short|char|\w+_t)\s+{re.escape(variable_name)}\s*=\s*([^;]+);', func.body)
        if decl_match:
            return decl_match.group(1).strip()
        
        return "NOT_FOUND"
    
    def tool_get_function_params(self, function_name: str) -> str:
        func = self.parser.functions.get(function_name)
        if not func:
            return f"FUNCTION_NOT_FOUND: {function_name}"
        return json.dumps(func.params)
    
    def tool_get_call_sites(self, function_name: str) -> str:
        results = []
        
        for caller_name, caller_func in self.parser.functions.items():
            if function_name not in caller_func.calls:
                continue
            
            pattern = rf'\b{re.escape(function_name)}\s*\(([^)]*)\)'
            for m in re.finditer(pattern, caller_func.body):
                args_str = m.group(1)
                args = self._split_args(args_str)
                results.append({
                    "caller": caller_name,
                    "caller_file": caller_func.file,
                    "args": args
                })
        
        if not results:
            return "NO_CALL_SITES_FOUND"
        
        return json.dumps(results, indent=2)
    
    def tool_evaluate_expression(self, expression: str) -> str:
        try:
            clean = re.sub(r'[^\d\s\+\-\*\/\(\)]', '', expression)
            if not clean.strip():
                return f"INVALID_EXPRESSION: {expression}"
            result = eval(clean)
            return str(int(result))
        except Exception as e:
            return f"EVAL_ERROR: {e}"
    
    def tool_check_if_parameter(self, function_name: str, variable_name: str) -> str:
        func = self.parser.functions.get(function_name)
        if not func:
            return f"FUNCTION_NOT_FOUND: {function_name}"
        
        if variable_name in func.params:
            idx = func.params.index(variable_name)
            return f"YES:parameter_index_{idx}"
        else:
            return f"NO:local_variable"
    
    def tool_find_macro_that_calls(self, function_name: str) -> str:
        results = []
        for macro_name, macro in self.parser.macros.items():
            if function_name in macro.calls:
                results.append({
                    "macro_name": macro_name,
                    "params": macro.params,
                    "body": macro.body
                })
        
        if not results:
            return "NO_MACROS_FOUND"
        return json.dumps(results, indent=2)
    
    def tool_get_macro_call_sites(self, macro_name: str) -> str:
        if macro_name not in self.parser.macros:
            return f"MACRO_NOT_FOUND: {macro_name}"
        
        macro = self.parser.macros[macro_name]
        results = []
        
        for func_name, func in self.parser.functions.items():
            pattern = rf'\b{re.escape(macro_name)}\s*\(([^)]*)\)'
            for m in re.finditer(pattern, func.body):
                args_str = m.group(1)
                args = self._split_args(args_str)
                results.append({
                    "caller_function": func_name,
                    "caller_file": func.file,
                    "args": args,
                    "macro_params": macro.params
                })
        
        if not results:
            return "NO_MACRO_CALL_SITES_FOUND"
        return json.dumps(results, indent=2)
    
    def _split_args(self, args_str: str) -> List[str]:
        args = []
        depth = 0
        current = ""
        for c in args_str:
            if c == '(':
                depth += 1
                current += c
            elif c == ')':
                depth -= 1
                current += c
            elif c == ',' and depth == 0:
                if current.strip():
                    args.append(current.strip())
                current = ""
            else:
                current += c
        if current.strip():
            args.append(current.strip())
        return args
    
    def execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "get_macro_value":
            return self.tool_get_macro_value(args["macro_name"])
        elif tool_name == "get_variable_assignment":
            return self.tool_get_variable_assignment(args["function_name"], args["variable_name"])
        elif tool_name == "get_function_params":
            return self.tool_get_function_params(args["function_name"])
        elif tool_name == "get_call_sites":
            return self.tool_get_call_sites(args["function_name"])
        elif tool_name == "evaluate_expression":
            return self.tool_evaluate_expression(args["expression"])
        elif tool_name == "check_if_parameter":
            return self.tool_check_if_parameter(args["function_name"], args["variable_name"])
        elif tool_name == "find_macro_that_calls":
            return self.tool_find_macro_that_calls(args["function_name"])
        elif tool_name == "get_macro_call_sites":
            return self.tool_get_macro_call_sites(args["macro_name"])
        elif tool_name == "submit_answer":
            return f"ANSWER:{args['value']}"
        else:
            return f"UNKNOWN_TOOL: {tool_name}"
    
    # ==================== MAIN RESOLVER WITH TRACING ====================
    
    def resolve(self, mfs_call: MFSCallSite) -> str:
        raw_value = mfs_call.fileno_raw
        
        print(colored(f"\n  ┌{'─'*50}┐", Colors.CYAN))
        print(colored(f"  │ RESOLVING: {raw_value:<38} │", Colors.CYAN))
        print(colored(f"  │ Function: {mfs_call.function:<38} │", Colors.CYAN))
        print(colored(f"  └{'─'*50}┘", Colors.CYAN))
        
        # Quick check: already a number?
        if re.match(r'^-?\d+$', raw_value):
            print(colored(f"    ✓ Direct number: {raw_value}", Colors.GREEN))
            return raw_value
        
        # Quick check: simple macro?
        simple = self._try_simple_resolve(raw_value)
        if simple:
            print(colored(f"    ✓ Simple resolution: {raw_value} → {simple}", Colors.GREEN))
            return simple
        
        # Use LLM with tools
        if not self.client:
            print(colored("    ✗ No LLM available", Colors.RED))
            return f"{raw_value} (unresolved - no LLM)"
        
        print(colored("    🤖 Starting LLM resolution...", Colors.YELLOW))
        return self._resolve_with_llm(mfs_call)
    
    def _try_simple_resolve(self, value: str, depth: int = 0) -> Optional[str]:
        if depth > 20:
            return None
        
        value = value.strip()
        
        if re.match(r'^-?\d+$', value):
            return value
        
        if re.match(r'^0x[0-9a-fA-F]+$', value, re.I):
            try:
                return str(int(value, 16))
            except:
                pass
        
        if value in self.parser.value_macros:
            return self._try_simple_resolve(self.parser.value_macros[value], depth + 1)
        
        if value.startswith('(') and value.endswith(')'):
            return self._try_simple_resolve(value[1:-1].strip(), depth + 1)
        
        if any(op in value for op in ['+', '-', '*', '/']):
            expr = value
            for mn, mv in self.parser.value_macros.items():
                if re.search(rf'\b{mn}\b', expr):
                    resolved = self._try_simple_resolve(mv, depth + 1)
                    if resolved and re.match(r'^-?\d+$', resolved):
                        expr = re.sub(rf'\b{mn}\b', resolved, expr)
            
            clean = re.sub(r'[^\d\s\+\-\*\/\(\)]', '', expr)
            if clean.strip() and re.match(r'^[\d\s\+\-\*\/\(\)]+$', clean):
                try:
                    return str(int(eval(clean)))
                except:
                    pass
        
        return None
    
    def _resolve_with_llm(self, mfs_call: MFSCallSite, max_iterations: int = 15) -> str:
        path_str = " → ".join(s.name for s in mfs_call.path)
        
        system_prompt = """You are a C code analyzer. Your task is to determine the numeric value of the 3rd argument (fileno) in an mpf_mfs_open() call.

IMPORTANT RULES:
1. If the value is a NUMBER, submit it.
2. If it's a MACRO name, use get_macro_value.
3. If it's a VARIABLE:
   a. Use check_if_parameter first
   b. If LOCAL: use get_variable_assignment
   c. If PARAMETER: use get_call_sites to trace arguments
4. If get_call_sites returns NO_CALL_SITES_FOUND, use find_macro_that_calls then get_macro_call_sites.
5. For expressions like "(9000 + 1)", use evaluate_expression.

Keep tracing until you reach a final numeric value."""

        user_prompt = f"""Find the numeric value of the 3rd argument in this mpf_mfs_open call:

File: {mfs_call.file}
Function: {mfs_call.function}
Call: {mfs_call.full_call}
3rd argument: {mfs_call.fileno_raw}
Call path: {path_str}

Trace this value until you find the final number."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        for iteration in range(max_iterations):
            print(colored(f"\n    ╔══ Iteration {iteration + 1} ══╗", Colors.YELLOW))
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.TOOLS,
                    tool_choice="auto",
                    temperature=0
                )
                
                msg = response.choices[0].message
                
                # Show LLM thinking if any
                if msg.content:
                    content_preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    print(colored(f"    │ LLM Thinking: {content_preview}", Colors.DIM))
                
                # Check if done
                if msg.content and "ANSWER:" in str(msg.content):
                    match = re.search(r'ANSWER:(\d+)', str(msg.content))
                    if match:
                        result = match.group(1)
                        print(colored(f"    ╚══ RESOLVED: {result} ══╝", Colors.GREEN + Colors.BOLD))
                        return result
                
                # Process tool calls
                if msg.tool_calls:
                    messages.append(msg)
                    
                    for tc_idx, tool_call in enumerate(msg.tool_calls):
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        
                        # Print tool call
                        args_str = ", ".join(f"{k}={v}" for k, v in func_args.items())
                        print(colored(f"    │ 🔧 Tool Call {tc_idx + 1}: {func_name}({args_str})", Colors.BLUE))
                        
                        result = self.execute_tool(func_name, func_args)
                        
                        # Print result (truncated)
                        result_preview = result[:100] + "..." if len(result) > 100 else result
                        print(colored(f"    │    ↳ Result: {result_preview}", Colors.CYAN))
                        
                        # Check if answer submitted
                        if result.startswith("ANSWER:"):
                            final_val = result.replace("ANSWER:", "")
                            print(colored(f"    ╚══ RESOLVED: {final_val} ══╝", Colors.GREEN + Colors.BOLD))
                            return final_val
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })
                else:
                    # No tool calls
                    if msg.content:
                        nums = re.findall(r'\b(\d{4,})\b', msg.content)
                        if nums:
                            print(colored(f"    ╚══ RESOLVED (from text): {nums[0]} ══╝", Colors.GREEN))
                            return nums[0]
                    print(colored("    │ No tool calls and no answer found", Colors.RED))
                    break
                    
            except Exception as e:
                print(colored(f"    │ ⚠️ Error: {e}", Colors.RED))
                break
        
        print(colored(f"    ╚══ UNRESOLVED after {max_iterations} iterations ══╝", Colors.RED))
        return f"{mfs_call.fileno_raw} (unresolved)"


# ============================================================================
# MAIN AGENT
# ============================================================================

class FileNoAgent:
    def __init__(self, project_dirs: List[str], verbose: bool = True, use_llm: bool = True):
        self.project_dirs = project_dirs
        self.verbose = verbose
        self.use_llm = use_llm
        
        self.parser = CParser(verbose)
        self.paths: List[List[PathStep]] = []
        self.mfs_calls: List[MFSCallSite] = []
    
    def analyze(self, entry: str = "main", target: str = "mpf_mfs_open") -> Dict:
        # Phase 1: Load & Parse
        print(colored("\n" + "═"*60, Colors.HEADER))
        print(colored(" PHASE 1: LOADING & PARSING FILES", Colors.HEADER + Colors.BOLD))
        print(colored("═"*60, Colors.HEADER))
        
        for d in self.project_dirs:
            self.parser.load_directory(d)
        
        if not self.parser.files:
            return {"success": False, "error": "No files"}
        
        self.parser.build_call_graph()
        
        print(f"\n  📂 Files loaded: {colored(str(len(self.parser.files)), Colors.GREEN)}")
        print(f"  📊 Functions found: {colored(str(len(self.parser.functions)), Colors.GREEN)}")
        print(f"  📋 Value macros: {colored(str(len(self.parser.value_macros)), Colors.GREEN)}")
        print(f"  📦 Function macros: {colored(str(len(self.parser.macros)), Colors.GREEN)}")
        
        # Phase 2: Find Paths
        print(colored("\n" + "═"*60, Colors.HEADER))
        print(colored(f" PHASE 2: FINDING PATHS {entry}() → {target}()", Colors.HEADER + Colors.BOLD))
        print(colored("═"*60, Colors.HEADER))
        
        finder = PathFinder(self.parser)
        self.paths = finder.find_paths(entry, target)
        finder.print_paths(self.paths, entry, target)
        
        # Phase 3: Extract MFS Calls
        print(colored("\n" + "═"*60, Colors.HEADER))
        print(colored(" PHASE 3: EXTRACTING mpf_mfs_open CALLS", Colors.HEADER + Colors.BOLD))
        print(colored("═"*60, Colors.HEADER))
        
        extractor = MFSExtractor(self.parser)
        self.mfs_calls = extractor.extract_from_paths(self.paths)
        extractor.print_calls(self.mfs_calls)
        
        # Phase 4: Resolve Values
        print(colored("\n" + "═"*60, Colors.HEADER))
        print(colored(" PHASE 4: RESOLVING FILENO VALUES (LLM + TOOLS)", Colors.HEADER + Colors.BOLD))
        print(colored("═"*60, Colors.HEADER))
        
        resolver = ToolBasedResolver(self.parser, self.verbose)
        
        for call in self.mfs_calls:
            call.fileno_resolved = resolver.resolve(call)
        
        return self._build_result()
    
    def _build_result(self) -> Dict:
        return {
            "success": True,
            "files": len(self.parser.files),
            "functions": len(self.parser.functions),
            "paths": len(self.paths),
            "mfs_calls": [
                {
                    "file": c.file,
                    "line": c.line,
                    "function": c.function,
                    "fileno_raw": c.fileno_raw,
                    "fileno_resolved": c.fileno_resolved,
                    "path": " → ".join(s.name for s in c.path)
                }
                for c in self.mfs_calls
            ]
        }
    
    def print_results(self):
        print(colored("\n" + "═"*60, Colors.GREEN))
        print(colored(" FINAL RESULTS", Colors.GREEN + Colors.BOLD))
        print(colored("═"*60, Colors.GREEN))
        
        print(f"\n  📂 Files: {len(self.parser.files)}")
        print(f"  📊 Functions: {len(self.parser.functions)}")
        print(f"  🛤️  Paths: {len(self.paths)}")
        print(f"  📍 Calls: {len(self.mfs_calls)}")
        
        print(colored("\n  " + "─"*56, Colors.DIM))
        
        for c in self.mfs_calls:
            print(f"\n  📄 {colored(c.file, Colors.BLUE)}:{c.line} in {colored(c.function + '()', Colors.GREEN)}")
            
            if c.fileno_raw != c.fileno_resolved and "(unresolved)" not in c.fileno_resolved:
                print(f"     Fileno: {c.fileno_raw} → {colored(c.fileno_resolved, Colors.GREEN + Colors.BOLD)} ✅")
            else:
                print(f"     Fileno: {colored(c.fileno_resolved, Colors.YELLOW)}")
            
            path_str = " → ".join(s.name for s in c.path)
            print(f"     Path: {colored(path_str, Colors.DIM)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="FileNo Agent v14 - Full Tracing")
    parser.add_argument("project_dir")
    parser.add_argument("include_dirs", nargs="*")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--entry", default="main")
    parser.add_argument("--target", default="mpf_mfs_open")
    
    args = parser.parse_args()
    
    print(colored("\n" + "╔" + "═"*58 + "╗", Colors.CYAN + Colors.BOLD))
    print(colored("║" + " FileNo Agent v14.0 - Tool-Based LLM with Full Tracing ".center(58) + "║", Colors.CYAN + Colors.BOLD))
    print(colored("╚" + "═"*58 + "╝", Colors.CYAN + Colors.BOLD))
    
    dirs = [args.project_dir] + (args.include_dirs or [])
    agent = FileNoAgent(dirs, args.verbose, not args.no_llm)
    agent.analyze(args.entry, args.target)
    agent.print_results()


if __name__ == "__main__":
    main()

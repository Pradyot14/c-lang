#!/usr/bin/env python3
"""
FileNo Agent v15.1 - Smart Path-Aware LLM Resolution
=====================================================
Key improvement: LLM gets PATH context + smart tools that use the path.
Tools are designed to help LLM trace backward on the KNOWN path efficiently.

Tools:
- get_function_body(func_name) - Get specific function's code
- get_call_args(caller, callee) - What does caller pass to callee on THIS path?
- get_macros(expression) - Get macro values for tokens in expression
- evaluate_expression(expr) - Evaluate arithmetic
- submit_answer(value, reasoning) - Final answer
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
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
            
            func_match = re.match(r'#\s*define\s+(\w+)\(([^)]*)\)\s*(.*)', line)
            if func_match:
                name = func_match.group(1)
                params = [p.strip() for p in func_match.group(2).split(',') if p.strip()]
                body = re.sub(r'/\*.*?\*/', '', func_match.group(3))
                body = re.sub(r'//.*$', '', body).strip()
                calls = self.find_calls(body)
                self.macros[name] = MacroDef(name, filename, body, params, calls)
                continue
            
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
        for root, _, files in os.walk(dirpath):
            for f in files:
                if f.endswith(('.h', '.hpp', '.inc')):
                    self.load_file(os.path.join(root, f))
        for root, _, files in os.walk(dirpath):
            for f in files:
                if f.endswith(('.c', '.cpp')):
                    self.load_file(os.path.join(root, f))
    
    def build_call_graph(self):
        self._detect_callbacks()
        for name, func in self.functions.items():
            self.call_graph[name] = func.calls
    
    def _detect_callbacks(self):
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
        print(colored(f"\n  📍 Found {len(calls)} mpf_mfs_open call(s):", Colors.CYAN))
        print()
        
        for i, call in enumerate(calls, 1):
            print(colored(f"  ┌─ Call {i} ─┐", Colors.YELLOW))
            print(f"  │ File: {colored(call.file, Colors.BLUE)}:{call.line}")
            print(f"  │ Function: {colored(call.function + '()', Colors.GREEN)}")
            print(f"  │ 3rd Arg (fileno): {colored(call.fileno_raw, Colors.RED + Colors.BOLD)}")
            print(f"  │ Path: {' → '.join(s.name for s in call.path)}")
            print(f"  └{'─' * 20}┘")
            print()


# ============================================================================
# SMART PATH-AWARE VALUE RESOLVER
# ============================================================================

class SmartPathResolver:
    """
    Smart LLM resolution with path-aware tools.
    LLM gets the path context and can request specific info using tools.
    """
    
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_function_body",
                "description": "Get the code body of a specific function from the call path. Use this to see the function's implementation, local variables, assignments, and how it calls other functions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "function_name": {"type": "string", "description": "Name of the function to get code for"}
                    },
                    "required": ["function_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_call_args",
                "description": "Get the arguments that a caller function passes when calling a callee function. This is useful to trace parameter values backward through the call path.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "caller": {"type": "string", "description": "The calling function name"},
                        "callee": {"type": "string", "description": "The called function name"}
                    },
                    "required": ["caller", "callee"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_macros",
                "description": "Get macro definitions for identifiers in an expression. Returns the values of any macros found.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Expression containing potential macro names"}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "evaluate_expression",
                "description": "Evaluate a numeric arithmetic expression like '9000 + 5' or '(4000 + 1)'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Arithmetic expression to evaluate"}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": "Submit the final resolved numeric value(s). Call this when you have determined the fileno value(s).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "string", "description": "Comma-separated numeric values (e.g., '6001' or '6001,6002' for conditionals)"},
                        "reasoning": {"type": "string", "description": "Brief explanation of how you traced to this value"}
                    },
                    "required": ["values"]
                }
            }
        }
    ]
    
    def __init__(self, parser: CParser, verbose: bool = True):
        self.parser = parser
        self.verbose = verbose
        self.client = None
        self.model = None
        self.provider = None
        self.current_path: List[PathStep] = []
        self._init_llm()
    
    def _init_llm(self):
        if HAS_GROQ:
            groq_key = os.environ.get("GROQ_API_KEY") or os.environ.get("groq")
            if groq_key:
                try:
                    self.client = Groq(api_key=groq_key)
                    self.model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
                    self.provider = "groq"
                    if self.verbose:
                        print(colored(f"    🚀 Using Groq ({self.model})", Colors.GREEN))
                    return
                except Exception as e:
                    pass
        
        if HAS_OPENAI:
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
            except:
                pass
            
            try:
                ok = os.environ.get("OPENAI_API_KEY", "")
                if ok:
                    self.client = OpenAI(api_key=ok)
                    self.model = os.environ.get("OPENAI_MODEL", "gpt-4o")
                    self.provider = "openai"
                    if self.verbose:
                        print(colored(f"    🟢 Using OpenAI ({self.model})", Colors.GREEN))
            except:
                pass
    
    # ==================== TOOL IMPLEMENTATIONS ====================
    
    def tool_get_function_body(self, function_name: str) -> str:
        """Get function body with parameter info"""
        func = self.parser.functions.get(function_name)
        if not func:
            return f"FUNCTION_NOT_FOUND: {function_name}"
        
        params_str = ", ".join(func.params) if func.params else "void"
        
        # Clean up body for readability
        body = func.body.strip()
        if len(body) > 1500:
            body = body[:1500] + "\n... (truncated)"
        
        return f"""FUNCTION: {function_name}({params_str})
FILE: {func.file}
PARAMETERS: {func.params if func.params else 'none'}
CODE:
{body}"""
    
    def tool_get_call_args(self, caller: str, callee: str) -> str:
        """Get what arguments caller passes to callee"""
        caller_func = self.parser.functions.get(caller)
        if not caller_func:
            return f"CALLER_NOT_FOUND: {caller}"
        
        callee_func = self.parser.functions.get(callee)
        callee_params = callee_func.params if callee_func else []
        
        # Find the call in caller's body
        pattern = rf'\b{re.escape(callee)}\s*\(([^)]*)\)'
        matches = list(re.finditer(pattern, caller_func.body))
        
        if not matches:
            return f"CALL_NOT_FOUND: {caller} does not call {callee}"
        
        results = []
        for m in matches:
            args_str = m.group(1)
            args = self._split_args(args_str)
            
            # Map args to params
            arg_mapping = []
            for i, arg in enumerate(args):
                param_name = callee_params[i] if i < len(callee_params) else f"arg{i}"
                arg_mapping.append(f"  {param_name} = {arg.strip()}")
            
            results.append(f"{callee}({args_str.strip()})\nArguments:\n" + "\n".join(arg_mapping))
        
        return f"In {caller}(), found {len(matches)} call(s) to {callee}:\n\n" + "\n\n".join(results)
    
    def tool_get_macros(self, expression: str) -> str:
        """Get macro values for tokens in expression"""
        tokens = re.findall(r'\b([A-Z_][A-Z0-9_]*)\b', expression)
        
        if not tokens:
            return f"No macro-like tokens found in: {expression}"
        
        results = []
        for token in set(tokens):
            if token in self.parser.value_macros:
                value = self.parser.value_macros[token]
                # Try to resolve nested macros
                resolved = self._try_simple_resolve(value)
                if resolved and resolved != value:
                    results.append(f"{token} = {value} = {resolved}")
                else:
                    results.append(f"{token} = {value}")
            elif token in self.parser.macros:
                m = self.parser.macros[token]
                results.append(f"{token}({', '.join(m.params)}) = {m.body}  [function macro]")
            else:
                results.append(f"{token} = NOT_FOUND")
        
        return "MACRO VALUES:\n" + "\n".join(results)
    
    def tool_evaluate_expression(self, expression: str) -> str:
        """Evaluate arithmetic expression"""
        try:
            # First resolve any macros
            expr = expression
            for macro_name, macro_val in self.parser.value_macros.items():
                if re.search(rf'\b{macro_name}\b', expr):
                    resolved = self._try_simple_resolve(macro_val)
                    if resolved and re.match(r'^-?\d+$', resolved):
                        expr = re.sub(rf'\b{macro_name}\b', resolved, expr)
            
            clean = re.sub(r'[^\d\s\+\-\*\/\(\)]', '', expr)
            if not clean.strip():
                return f"CANNOT_EVALUATE: {expression} (no numeric content after macro resolution)"
            
            result = eval(clean)
            return f"{expression} = {int(result)}"
        except Exception as e:
            return f"EVAL_ERROR: {expression} -> {e}"
    
    def _split_args(self, args_str: str) -> List[str]:
        """Split function arguments handling nested parentheses"""
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
    
    def _try_simple_resolve(self, value: str, depth: int = 0) -> Optional[str]:
        """Try to resolve value without LLM"""
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
    
    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool and return result"""
        if tool_name == "get_function_body":
            return self.tool_get_function_body(args["function_name"])
        elif tool_name == "get_call_args":
            return self.tool_get_call_args(args["caller"], args["callee"])
        elif tool_name == "get_macros":
            return self.tool_get_macros(args["expression"])
        elif tool_name == "evaluate_expression":
            return self.tool_evaluate_expression(args["expression"])
        elif tool_name == "submit_answer":
            return f"ANSWER:{args['values']}"
        else:
            return f"UNKNOWN_TOOL: {tool_name}"
    
    # ==================== MAIN RESOLVER ====================
    
    def resolve(self, mfs_call: MFSCallSite) -> str:
        """Resolve the fileno value for an MFS call"""
        raw_value = mfs_call.fileno_raw
        self.current_path = mfs_call.path
        
        path_str = " → ".join(s.name for s in mfs_call.path)
        
        print(colored(f"\n  ┌{'─'*60}┐", Colors.CYAN))
        print(colored(f"  │ RESOLVING: {raw_value:<47} │", Colors.CYAN))
        print(colored(f"  │ In function: {mfs_call.function:<44} │", Colors.CYAN))
        print(colored(f"  │ Path: {path_str:<52} │", Colors.CYAN))
        print(colored(f"  └{'─'*60}┘", Colors.CYAN))
        
        # Quick resolution for direct numbers
        if re.match(r'^-?\d+$', raw_value):
            print(colored(f"    ✓ Direct number: {raw_value}", Colors.GREEN))
            return raw_value
        
        # Try simple macro resolution
        simple = self._try_simple_resolve(raw_value)
        if simple:
            print(colored(f"    ✓ Simple resolution: {raw_value} → {simple}", Colors.GREEN))
            return simple
        
        # Use LLM for complex cases
        if not self.client:
            print(colored("    ✗ No LLM available", Colors.RED))
            return f"{raw_value} (unresolved - no LLM)"
        
        print(colored("    🤖 Starting smart LLM resolution...", Colors.YELLOW))
        return self._resolve_with_llm(mfs_call)
    
    def _build_path_context(self, mfs_call: MFSCallSite) -> str:
        """Build path context string for LLM"""
        lines = []
        
        for i, step in enumerate(mfs_call.path):
            func = self.parser.functions.get(step.name)
            if func:
                params = f"({', '.join(func.params)})" if func.params else "()"
                lines.append(f"  {i+1}. {step.name}{params} [{step.file}]")
            else:
                lines.append(f"  {i+1}. {step.name}() [{step.file or 'unknown'}]")
        
        # Add the final mpf_mfs_open call
        lines.append(f"  {len(mfs_call.path)+1}. mpf_mfs_open() ← TARGET CALL")
        
        return "\n".join(lines)
    
    def _resolve_with_llm(self, mfs_call: MFSCallSite, max_iterations: int = 10) -> str:
        """Use LLM with smart tools to resolve value"""
        
        path_context = self._build_path_context(mfs_call)
        
        # Get parameter info for the function containing the call
        func = self.parser.functions.get(mfs_call.function)
        func_params = func.params if func else []
        
        system_prompt = """You are a C code analyzer. Your task is to find the numeric value of a variable used in an mpf_mfs_open() call.

You have access to tools to inspect the code:
- get_function_body(func_name): Get a function's code to see variables, assignments, conditionals
- get_call_args(caller, callee): See what arguments a caller passes to a callee function
- get_macros(expression): Get macro values for identifiers
- evaluate_expression(expr): Calculate arithmetic expressions
- submit_answer(values, reasoning): Submit final answer when done

STRATEGY:
1. The value you need to resolve is in a function on the call path
2. If it's a PARAMETER of that function, use get_call_args to see what the caller passes
3. If it's a LOCAL VARIABLE, use get_function_body to see how it's assigned
4. If it's a MACRO, use get_macros to get its value
5. Keep tracing BACKWARD through the path until you reach a numeric value
6. For CONDITIONAL assignments (if/else/switch), report ALL possible values

Be efficient - request only what you need. Submit your answer as soon as you find the numeric value(s)."""

        user_prompt = f"""Find the numeric value of '{mfs_call.fileno_raw}' used in mpf_mfs_open().

CALL SITE:
  Function: {mfs_call.function}()
  File: {mfs_call.file}:{mfs_call.line}
  The mpf_mfs_open() call uses '{mfs_call.fileno_raw}' as the 3rd argument (fileno)

CALL PATH (trace backward from bottom to top):
{path_context}

The function {mfs_call.function}() has parameters: {func_params if func_params else 'none'}

Start by checking if '{mfs_call.fileno_raw}' is a parameter or local variable of {mfs_call.function}()."""

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
                
                # Show LLM's thinking if any
                if msg.content:
                    content_preview = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    print(colored(f"    │ 💭 {content_preview}", Colors.DIM))
                
                if msg.tool_calls:
                    messages.append(msg)
                    
                    for tc in msg.tool_calls:
                        func_name = tc.function.name
                        func_args = json.loads(tc.function.arguments)
                        
                        # Display tool call
                        args_str = ", ".join(f"{k}='{v}'" for k, v in func_args.items())
                        print(colored(f"    │ 🔧 {func_name}({args_str})", Colors.BLUE))
                        
                        # Execute tool
                        result = self.execute_tool(func_name, func_args)
                        
                        # Show result (truncated)
                        result_lines = result.split('\n')
                        if len(result_lines) <= 3:
                            for line in result_lines:
                                print(colored(f"    │    ↳ {line}", Colors.CYAN))
                        else:
                            print(colored(f"    │    ↳ {result_lines[0]}", Colors.CYAN))
                            print(colored(f"    │      ... ({len(result_lines)} lines)", Colors.DIM))
                        
                        # Check for answer
                        if result.startswith("ANSWER:"):
                            final_val = result.replace("ANSWER:", "")
                            reasoning = func_args.get("reasoning", "")
                            print(colored(f"    ╚══ RESOLVED: {final_val} ══╝", Colors.GREEN + Colors.BOLD))
                            if reasoning:
                                print(colored(f"        Reasoning: {reasoning}", Colors.DIM))
                            return final_val
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result
                        })
                else:
                    # No tool calls - try to extract answer from text
                    if msg.content:
                        nums = re.findall(r'\b(\d{4,})\b', msg.content)
                        if nums:
                            print(colored(f"    ╚══ RESOLVED (from text): {nums[0]} ══╝", Colors.GREEN))
                            return nums[0]
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
        print(colored("\n" + "═"*70, Colors.HEADER))
        print(colored(" PHASE 1: LOADING & PARSING FILES", Colors.HEADER + Colors.BOLD))
        print(colored("═"*70, Colors.HEADER))
        
        for d in self.project_dirs:
            self.parser.load_directory(d)
        
        if not self.parser.files:
            return {"success": False, "error": "No files"}
        
        self.parser.build_call_graph()
        
        print(f"\n  📂 Files loaded: {colored(str(len(self.parser.files)), Colors.GREEN)}")
        print(f"  📊 Functions found: {colored(str(len(self.parser.functions)), Colors.GREEN)}")
        print(f"  📋 Value macros: {colored(str(len(self.parser.value_macros)), Colors.GREEN)}")
        print(f"  📦 Function macros: {colored(str(len(self.parser.macros)), Colors.GREEN)}")
        
        print(colored("\n" + "═"*70, Colors.HEADER))
        print(colored(f" PHASE 2: FINDING PATHS {entry}() → {target}()", Colors.HEADER + Colors.BOLD))
        print(colored("═"*70, Colors.HEADER))
        
        finder = PathFinder(self.parser)
        self.paths = finder.find_paths(entry, target)
        finder.print_paths(self.paths, entry, target)
        
        print(colored("\n" + "═"*70, Colors.HEADER))
        print(colored(" PHASE 3: EXTRACTING mpf_mfs_open CALLS", Colors.HEADER + Colors.BOLD))
        print(colored("═"*70, Colors.HEADER))
        
        extractor = MFSExtractor(self.parser)
        self.mfs_calls = extractor.extract_from_paths(self.paths)
        extractor.print_calls(self.mfs_calls)
        
        print(colored("\n" + "═"*70, Colors.HEADER))
        print(colored(" PHASE 4: SMART PATH-AWARE VALUE RESOLUTION", Colors.HEADER + Colors.BOLD))
        print(colored("═"*70, Colors.HEADER))
        
        resolver = SmartPathResolver(self.parser, self.verbose)
        
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
        print(colored("\n" + "═"*70, Colors.GREEN))
        print(colored(" FINAL RESULTS", Colors.GREEN + Colors.BOLD))
        print(colored("═"*70, Colors.GREEN))
        
        print(f"\n  📂 Files: {len(self.parser.files)}")
        print(f"  📊 Functions: {len(self.parser.functions)}")
        print(f"  🛤️  Paths: {len(self.paths)}")
        print(f"  📍 Calls: {len(self.mfs_calls)}")
        
        print(colored("\n  " + "─"*66, Colors.DIM))
        
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
    parser = argparse.ArgumentParser(description="FileNo Agent v15.1 - Smart Path-Aware Resolution")
    parser.add_argument("project_dir")
    parser.add_argument("include_dirs", nargs="*")
    parser.add_argument("-v", "--verbose", action="store_true", default=True)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--entry", default="main")
    parser.add_argument("--target", default="mpf_mfs_open")
    
    args = parser.parse_args()
    
    print(colored("\n" + "╔" + "═"*68 + "╗", Colors.CYAN + Colors.BOLD))
    print(colored("║" + " FileNo Agent v15.1 - Smart Path-Aware LLM Resolution ".center(68) + "║", Colors.CYAN + Colors.BOLD))
    print(colored("╚" + "═"*68 + "╝", Colors.CYAN + Colors.BOLD))
    
    dirs = [args.project_dir] + (args.include_dirs or [])
    agent = FileNoAgent(dirs, args.verbose, not args.no_llm)
    agent.analyze(args.entry, args.target)
    agent.print_results()


if __name__ == "__main__":
    main()

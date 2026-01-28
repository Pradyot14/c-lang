#!/usr/bin/env python3
"""
FileNo Agent v16.0 - Full Path-Aware Resolution
================================================
Key improvement: Extract call arguments DURING path tracing.
LLM receives complete call chain with arguments - no blind discovery.

The path tracing now captures:
- Function name, file, params
- What arguments each function passes to the next
- Complete chain: main() --[args]--> func1() --[args]--> func2() --> mpf_mfs_open()

LLM just reasons over the pre-extracted data. Tools only for complex cases.
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
from copy import deepcopy


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
    """Enhanced path step with call argument info"""
    name: str
    type: CallType
    file: Optional[str] = None
    params: List[str] = field(default_factory=list)
    # What this function passes when calling the NEXT function in path
    call_to_next: Optional[str] = None  # e.g., "shared_open_file(fno)"
    args_to_next: List[str] = field(default_factory=list)  # e.g., ["fno"]


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
    
    def get_call_args(self, caller_name: str, callee_name: str) -> Tuple[str, List[str]]:
        """Get what arguments caller passes to callee"""
        caller = self.functions.get(caller_name)
        if not caller:
            return "", []
        
        pattern = rf'\b{re.escape(callee_name)}\s*\(([^)]*)\)'
        match = re.search(pattern, caller.body)
        
        if not match:
            return "", []
        
        args_str = match.group(1).strip()
        args = self._split_args(args_str)
        
        return f"{callee_name}({args_str})", args
    
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
    
    def get_variable_assignments(self, func_name: str, var_name: str) -> List[Dict]:
        """Get all assignments to a variable in a function, including conditionals"""
        func = self.functions.get(func_name)
        if not func:
            return []
        
        assignments = []
        
        # Find all assignments
        for m in re.finditer(rf'\b{re.escape(var_name)}\s*=\s*([^;]+);', func.body):
            value = m.group(1).strip()
            # Check if inside if/else/switch
            before = func.body[:m.start()]
            
            # Simple heuristic for conditionals
            context = "default"
            if_match = re.search(r'if\s*\([^)]+\)\s*\{[^}]*$', before)
            else_match = re.search(r'else\s*\{[^}]*$', before)
            switch_match = re.search(r'case\s+\w+\s*:[^}]*$', before)
            
            if if_match:
                context = "conditional (if)"
            elif else_match:
                context = "conditional (else)"
            elif switch_match:
                context = "conditional (switch)"
            
            assignments.append({"value": value, "context": context})
        
        # Check declaration with initialization
        decl_match = re.search(rf'(?:int|long|short|char|\w+_t)\s+{re.escape(var_name)}\s*=\s*([^;]+);', func.body)
        if decl_match:
            assignments.insert(0, {"value": decl_match.group(1).strip(), "context": "initialization"})
        
        return assignments


# ============================================================================
# ENHANCED PATH FINDER - Extracts call arguments along path
# ============================================================================

class EnhancedPathFinder:
    """Path finder that extracts call arguments during traversal"""
    
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
                # Deep copy to avoid mutation issues with shared PathStep objects
                paths.append(deepcopy(path))
                return
            
            visited.add(current)
            
            for callee in self.parser.call_graph.get(current, set()):
                func = self.parser.functions.get(callee)
                
                # Get call arguments from current to callee
                call_str, call_args = self.parser.get_call_args(current, callee)
                
                step = PathStep(
                    name=callee,
                    type=CallType.DIRECT,
                    file=func.file if func else None,
                    params=func.params if func else [],
                    call_to_next=None,
                    args_to_next=[]
                )
                
                # Update PREVIOUS step with call info to THIS step
                if path:
                    path[-1].call_to_next = call_str
                    path[-1].args_to_next = call_args
                
                path.append(step)
                dfs(callee, path, visited)
                path.pop()
                
                # Restore previous step's call info
                if path:
                    path[-1].call_to_next = None
                    path[-1].args_to_next = []
            
            visited.discard(current)
        
        entry_func = self.parser.functions.get(entry)
        entry_step = PathStep(
            name=entry,
            type=CallType.ENTRY,
            file=entry_func.file if entry_func else None,
            params=entry_func.params if entry_func else []
        )
        
        dfs(entry, [entry_step], set())
        
        # Post-process: fill in call info for complete paths
        for path in paths:
            for i in range(len(path) - 1):
                if not path[i].call_to_next:
                    call_str, call_args = self.parser.get_call_args(path[i].name, path[i+1].name)
                    path[i].call_to_next = call_str
                    path[i].args_to_next = call_args
        
        return paths
    
    def print_paths(self, paths: List[List[PathStep]], entry: str, target: str):
        print(colored(f"\n  🛤️  Found {len(paths)} path(s) from {entry}() to {target}():", Colors.CYAN))
        print()
        
        for i, path in enumerate(paths, 1):
            print(colored(f"  ╔══ Path {i} ══╗", Colors.YELLOW))
            
            for j, step in enumerate(path):
                is_last = (j == len(path) - 1)
                
                # Build function signature
                params_str = f"({', '.join(step.params)})" if step.params else "()"
                file_info = colored(f" [{step.file}]", Colors.DIM) if step.file else ""
                
                if step.type == CallType.ENTRY:
                    func_str = colored(f"{step.name}{params_str}", Colors.GREEN + Colors.BOLD)
                elif is_last:
                    func_str = colored(f"{step.name}{params_str}", Colors.RED + Colors.BOLD)
                else:
                    func_str = colored(f"{step.name}{params_str}", Colors.BLUE)
                
                # Print function
                indent = "  │  " * j
                connector = "╚═▶" if is_last else "╠═▶"
                print(f"  {indent}{connector} {func_str}{file_info}")
                
                # Print call to next (if not last)
                if step.call_to_next and not is_last:
                    call_indent = "  │  " * (j + 1)
                    call_info = colored(f"↓ calls: {step.call_to_next}", Colors.DIM)
                    print(f"  {call_indent}{call_info}")
            
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
# FULL CONTEXT RESOLVER - Gives LLM everything upfront
# ============================================================================

class FullContextResolver:
    """
    v16: LLM receives complete pre-extracted context.
    No blind discovery - all call args are pre-computed from path tracing.
    """
    
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_variable_assignments",
                "description": "Get ALL assignments to a variable in a function, including conditional branches (if/else/switch). Use this when you see a variable that's not a parameter and need to know its possible values.",
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
                "name": "get_macro_value",
                "description": "Get the value of a macro. Use when you encounter a MACRO_NAME that needs resolution.",
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
                "name": "evaluate_expression",
                "description": "Evaluate an arithmetic expression like '6000 + 1' or '(9000 + 5)'.",
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
                "name": "submit_answer",
                "description": "Submit the final resolved numeric value(s). Use comma-separated values if multiple (e.g., from conditionals).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "string", "description": "Comma-separated numeric values"},
                        "trace": {"type": "string", "description": "Brief trace showing: value ← source ← source"}
                    },
                    "required": ["values", "trace"]
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
                except:
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
    
    # ==================== TOOL IMPLEMENTATIONS ====================
    
    def tool_get_variable_assignments(self, func_name: str, var_name: str) -> str:
        assignments = self.parser.get_variable_assignments(func_name, var_name)
        
        if not assignments:
            # Check if it's a parameter
            func = self.parser.functions.get(func_name)
            if func and var_name in func.params:
                idx = func.params.index(var_name)
                return f"'{var_name}' is a PARAMETER (index {idx}) of {func_name}(), not a local variable."
            return f"No assignments found for '{var_name}' in {func_name}()"
        
        lines = [f"Assignments to '{var_name}' in {func_name}():"]
        for a in assignments:
            lines.append(f"  • {var_name} = {a['value']}  [{a['context']}]")
        
        return "\n".join(lines)
    
    def tool_get_macro_value(self, macro_name: str) -> str:
        if macro_name in self.parser.value_macros:
            raw = self.parser.value_macros[macro_name]
            resolved = self._try_simple_resolve(raw)
            if resolved and resolved != raw:
                return f"{macro_name} = {raw} = {resolved}"
            return f"{macro_name} = {raw}"
        
        if macro_name in self.parser.macros:
            m = self.parser.macros[macro_name]
            return f"{macro_name}({', '.join(m.params)}) = {m.body}  [function macro]"
        
        return f"MACRO NOT FOUND: {macro_name}"
    
    def tool_evaluate_expression(self, expression: str) -> str:
        try:
            expr = expression
            for mn, mv in self.parser.value_macros.items():
                if re.search(rf'\b{mn}\b', expr):
                    resolved = self._try_simple_resolve(mv)
                    if resolved and re.match(r'^-?\d+$', resolved):
                        expr = re.sub(rf'\b{mn}\b', resolved, expr)
            
            clean = re.sub(r'[^\d\s\+\-\*\/\(\)]', '', expr)
            if not clean.strip():
                return f"Cannot evaluate: {expression}"
            
            result = eval(clean)
            return f"{expression} = {int(result)}"
        except Exception as e:
            return f"Error evaluating '{expression}': {e}"
    
    def execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name == "get_variable_assignments":
            return self.tool_get_variable_assignments(args["function_name"], args["variable_name"])
        elif tool_name == "get_macro_value":
            return self.tool_get_macro_value(args["macro_name"])
        elif tool_name == "evaluate_expression":
            return self.tool_evaluate_expression(args["expression"])
        elif tool_name == "submit_answer":
            return f"ANSWER:{args['values']}"
        else:
            return f"Unknown tool: {tool_name}"
    
    # ==================== CONTEXT BUILDER ====================
    
    def _build_full_context(self, mfs_call: MFSCallSite) -> str:
        """Build complete pre-extracted context for LLM"""
        lines = []
        
        lines.append("=" * 60)
        lines.append("COMPLETE CALL CHAIN (pre-extracted from path tracing)")
        lines.append("=" * 60)
        lines.append("")
        
        # Build the chain with argument mappings
        for i, step in enumerate(mfs_call.path):
            func = self.parser.functions.get(step.name)
            params_str = f"({', '.join(step.params)})" if step.params else "()"
            
            lines.append(f"[{i+1}] {step.name}{params_str}")
            lines.append(f"    File: {step.file}")
            
            if step.params:
                lines.append(f"    Parameters: {step.params}")
            
            # Show what this function calls (with args)
            if step.call_to_next:
                lines.append(f"    ↓ Calls: {step.call_to_next}")
                
                # Map arguments to next function's parameters
                if i + 1 < len(mfs_call.path):
                    next_step = mfs_call.path[i + 1]
                    if next_step.params and step.args_to_next:
                        mappings = []
                        for j, arg in enumerate(step.args_to_next):
                            if j < len(next_step.params):
                                param = next_step.params[j]
                                mappings.append(f"{param}={arg}")
                        if mappings:
                            lines.append(f"    ↓ Arg mapping: {', '.join(mappings)}")
            
            lines.append("")
        
        # Final call to mpf_mfs_open
        lines.append(f"[{len(mfs_call.path)+1}] mpf_mfs_open(..., {mfs_call.fileno_raw}, ...)")
        lines.append(f"    File: {mfs_call.file}:{mfs_call.line}")
        lines.append(f"    3rd argument (fileno): {mfs_call.fileno_raw}")
        lines.append("")
        
        # Relevant macros
        lines.append("=" * 60)
        lines.append("POTENTIALLY RELEVANT MACROS")
        lines.append("=" * 60)
        
        # Find macros that might be relevant
        relevant_macros = []
        for name, value in self.parser.value_macros.items():
            # Check if macro name appears anywhere in the trace
            for step in mfs_call.path:
                if step.args_to_next:
                    for arg in step.args_to_next:
                        if name in arg or name == arg:
                            resolved = self._try_simple_resolve(value)
                            if resolved:
                                relevant_macros.append(f"  {name} = {value} → {resolved}")
                            else:
                                relevant_macros.append(f"  {name} = {value}")
        
        # Also check the fileno_raw
        if mfs_call.fileno_raw in self.parser.value_macros:
            value = self.parser.value_macros[mfs_call.fileno_raw]
            resolved = self._try_simple_resolve(value)
            if resolved:
                relevant_macros.append(f"  {mfs_call.fileno_raw} = {value} → {resolved}")
        
        if relevant_macros:
            lines.extend(list(set(relevant_macros)))
        else:
            lines.append("  (use get_macro_value tool if you encounter macro names)")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def _build_backward_trace(self, mfs_call: MFSCallSite) -> str:
        """Build backward trace showing the argument chain"""
        lines = []
        lines.append("BACKWARD TRACE (what you need to resolve):")
        lines.append("")
        
        # Start from the fileno value
        current_var = mfs_call.fileno_raw
        current_func = mfs_call.function
        
        lines.append(f"  mpf_mfs_open uses: {current_var}")
        
        # Check if it's already a number
        if re.match(r'^-?\d+$', current_var):
            lines.append(f"  └─ '{current_var}' is a DIRECT NUMBER")
            lines.append(f"     └─ ✅ RESOLVED: {current_var}")
            lines.append("")
            return "\n".join(lines)
        
        # Check if it's a macro at the top level
        if current_var in self.parser.value_macros:
            resolved = self._try_simple_resolve(current_var)
            if resolved:
                lines.append(f"  └─ '{current_var}' is a MACRO = {resolved}")
                lines.append(f"     └─ ✅ RESOLVED: {resolved}")
            else:
                lines.append(f"  └─ '{current_var}' is a MACRO = {self.parser.value_macros[current_var]}")
                lines.append(f"     └─ Use get_macro_value(\"{current_var}\") to resolve")
            lines.append("")
            return "\n".join(lines)
        
        # Walk backward through path
        for i in range(len(mfs_call.path) - 1, -1, -1):
            step = mfs_call.path[i]
            
            if step.name == current_func:
                # Check if current_var is a parameter
                if current_var in step.params:
                    param_idx = step.params.index(current_var)
                    lines.append(f"  └─ '{current_var}' is PARAMETER[{param_idx}] of {step.name}()")
                    
                    # Find what caller passes
                    if i > 0:
                        caller = mfs_call.path[i - 1]
                        if caller.args_to_next and param_idx < len(caller.args_to_next):
                            arg_passed = caller.args_to_next[param_idx]
                            lines.append(f"     └─ {caller.name}() passes: {arg_passed}")
                            current_var = arg_passed
                            current_func = caller.name
                            
                            # Check if the passed arg is a number
                            if re.match(r'^-?\d+$', current_var):
                                lines.append(f"  └─ '{current_var}' is a DIRECT NUMBER")
                                lines.append(f"     └─ ✅ RESOLVED: {current_var}")
                                break
                            
                            # Check if the passed arg is a macro
                            if current_var in self.parser.value_macros:
                                resolved = self._try_simple_resolve(current_var)
                                if resolved:
                                    lines.append(f"  └─ '{current_var}' is a MACRO = {resolved}")
                                    lines.append(f"     └─ ✅ RESOLVED: {resolved}")
                                else:
                                    lines.append(f"  └─ '{current_var}' is a MACRO = {self.parser.value_macros[current_var]}")
                                break
                # Check if it's a known macro
                elif current_var in self.parser.value_macros:
                    resolved = self._try_simple_resolve(current_var)
                    if resolved:
                        lines.append(f"  └─ '{current_var}' is a MACRO = {resolved}")
                        lines.append(f"     └─ ✅ RESOLVED: {resolved}")
                    else:
                        lines.append(f"  └─ '{current_var}' is a MACRO = {self.parser.value_macros[current_var]}")
                        lines.append(f"     └─ Use get_macro_value(\"{current_var}\") to resolve")
                    break
                # Otherwise it's a local variable
                else:
                    lines.append(f"  └─ '{current_var}' is LOCAL VARIABLE in {step.name}()")
                    lines.append(f"     └─ Use get_variable_assignments(\"{step.name}\", \"{current_var}\") to find its value")
                    break
        
        lines.append("")
        return "\n".join(lines)
    
    # ==================== MAIN RESOLVER ====================
    
    def resolve(self, mfs_call: MFSCallSite) -> str:
        raw_value = mfs_call.fileno_raw
        path_str = " → ".join(s.name for s in mfs_call.path)
        
        print(colored(f"\n  ┌{'─'*65}┐", Colors.CYAN))
        print(colored(f"  │ {'RESOLVING: ' + raw_value:<63} │", Colors.CYAN))
        print(colored(f"  │ {'In: ' + mfs_call.function + '()':<63} │", Colors.CYAN))
        print(colored(f"  │ {'Path: ' + path_str:<63} │", Colors.CYAN))
        print(colored(f"  └{'─'*65}┘", Colors.CYAN))
        
        # Quick resolution
        if re.match(r'^-?\d+$', raw_value):
            print(colored(f"    ✓ Direct number: {raw_value}", Colors.GREEN))
            return raw_value
        
        simple = self._try_simple_resolve(raw_value)
        if simple:
            print(colored(f"    ✓ Simple resolution: {raw_value} → {simple}", Colors.GREEN))
            return simple
        
        if not self.client:
            print(colored("    ✗ No LLM available", Colors.RED))
            return f"{raw_value} (unresolved - no LLM)"
        
        print(colored("    🤖 Starting full-context LLM resolution...", Colors.YELLOW))
        return self._resolve_with_llm(mfs_call)
    
    def _resolve_with_llm(self, mfs_call: MFSCallSite, max_iterations: int = 8) -> str:
        full_context = self._build_full_context(mfs_call)
        backward_trace = self._build_backward_trace(mfs_call)
        
        system_prompt = """You are a C code analyzer. You receive a COMPLETE call chain with pre-extracted argument mappings.

Your task: Find the numeric value of the fileno argument used in mpf_mfs_open().

The call chain shows:
- Each function and its parameters
- What arguments each function passes to the next (already extracted!)
- Argument-to-parameter mappings

STRATEGY:
1. Follow the BACKWARD TRACE provided - it shows the variable chain
2. If a variable is a PARAMETER, the caller's argument is already shown
3. If a variable is a LOCAL VARIABLE, use get_variable_assignments() to find its value
4. If you see a MACRO_NAME (all caps), use get_macro_value() to resolve it
5. For arithmetic expressions, use evaluate_expression()
6. Submit your answer with submit_answer(values, trace)

Most cases can be solved by following the pre-extracted chain. Use tools only when needed."""

        user_prompt = f"""Resolve the numeric value of '{mfs_call.fileno_raw}' in mpf_mfs_open().

{full_context}
{backward_trace}

Follow the trace and resolve to numeric value(s). Submit answer when done."""

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
                
                if msg.content:
                    preview = msg.content[:250] + "..." if len(msg.content) > 250 else msg.content
                    print(colored(f"    │ 💭 {preview}", Colors.DIM))
                
                if msg.tool_calls:
                    messages.append(msg)
                    
                    for tc in msg.tool_calls:
                        func_name = tc.function.name
                        func_args = json.loads(tc.function.arguments)
                        
                        args_str = ", ".join(f'{k}="{v}"' for k, v in func_args.items())
                        print(colored(f"    │ 🔧 {func_name}({args_str})", Colors.BLUE))
                        
                        result = self.execute_tool(func_name, func_args)
                        
                        for line in result.split('\n')[:4]:
                            print(colored(f"    │    ↳ {line}", Colors.CYAN))
                        if len(result.split('\n')) > 4:
                            print(colored(f"    │    ... ({len(result.split(chr(10)))} lines)", Colors.DIM))
                        
                        if result.startswith("ANSWER:"):
                            final = result.replace("ANSWER:", "")
                            trace = func_args.get("trace", "")
                            print(colored(f"    ╚══ RESOLVED: {final} ══╝", Colors.GREEN + Colors.BOLD))
                            if trace:
                                print(colored(f"        Trace: {trace}", Colors.DIM))
                            return final
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result
                        })
                else:
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
    def __init__(self, project_dirs: List[str], verbose: bool = True):
        self.project_dirs = project_dirs
        self.verbose = verbose
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
        print(colored(f" PHASE 2: FINDING PATHS WITH ARGUMENT EXTRACTION", Colors.HEADER + Colors.BOLD))
        print(colored("═"*70, Colors.HEADER))
        
        finder = EnhancedPathFinder(self.parser)
        self.paths = finder.find_paths(entry, target)
        finder.print_paths(self.paths, entry, target)
        
        print(colored("\n" + "═"*70, Colors.HEADER))
        print(colored(" PHASE 3: EXTRACTING mpf_mfs_open CALLS", Colors.HEADER + Colors.BOLD))
        print(colored("═"*70, Colors.HEADER))
        
        extractor = MFSExtractor(self.parser)
        self.mfs_calls = extractor.extract_from_paths(self.paths)
        extractor.print_calls(self.mfs_calls)
        
        print(colored("\n" + "═"*70, Colors.HEADER))
        print(colored(" PHASE 4: FULL-CONTEXT VALUE RESOLUTION", Colors.HEADER + Colors.BOLD))
        print(colored("═"*70, Colors.HEADER))
        
        resolver = FullContextResolver(self.parser, self.verbose)
        
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
    p = argparse.ArgumentParser(description="FileNo Agent v16 - Full Context Resolution")
    p.add_argument("project_dir")
    p.add_argument("include_dirs", nargs="*")
    p.add_argument("-v", "--verbose", action="store_true", default=True)
    p.add_argument("--entry", default="main")
    p.add_argument("--target", default="mpf_mfs_open")
    
    args = p.parse_args()
    
    print(colored("\n" + "╔" + "═"*68 + "╗", Colors.CYAN + Colors.BOLD))
    print(colored("║" + " FileNo Agent v16.0 - Full Path Context LLM Resolution ".center(68) + "║", Colors.CYAN + Colors.BOLD))
    print(colored("╚" + "═"*68 + "╝", Colors.CYAN + Colors.BOLD))
    
    dirs = [args.project_dir] + (args.include_dirs or [])
    agent = FileNoAgent(dirs, args.verbose)
    agent.analyze(args.entry, args.target)
    agent.print_results()


if __name__ == "__main__":
    main()

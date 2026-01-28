#!/usr/bin/env python3
"""
FileNo Agent v17.0 - HTML-Matched Parser
=========================================
Parser logic exactly matches the working HTML tracer.
Key changes from v16:
1. Call graph stores Map with type info (direct/macro/pointer/callback)
2. Callback detection pattern: var->member() 
3. Macro expansion matches HTML exactly
4. Function parsing simplified to match HTML
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
    MACRO_INVOKE = "macro-invoke"
    POINTER = "pointer"
    CALLBACK = "callback"
    ENTRY = "entry"


@dataclass
class CallInfo:
    """Info about a function call - matches HTML structure"""
    type: CallType
    via: Optional[str] = None  # For macro/pointer/callback - what it came through


@dataclass
class FunctionDef:
    name: str
    file: str
    body: str
    params: List[str] = field(default_factory=list)
    calls: Dict[str, CallInfo] = field(default_factory=dict)  # callee -> CallInfo


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
    via: Optional[str] = None
    file: Optional[str] = None
    params: List[str] = field(default_factory=list)
    call_to_next: Optional[str] = None
    args_to_next: List[str] = field(default_factory=list)


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
# C PARSER - EXACT MATCH TO HTML LOGIC
# ============================================================================

class CParser:
    """Parser that exactly matches the HTML tracer logic"""
    
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
        self.call_graph: Dict[str, Dict[str, CallInfo]] = {}  # Changed to match HTML
        self.log: List[str] = []
    
    def add_log(self, log_type: str, msg: str):
        self.log.append(f"[{log_type}] {msg}")
        if self.verbose:
            print(f"  [{log_type}] {msg}")
    
    # ==================== CLEAN CODE (matches HTML) ====================
    def clean(self, code: str) -> str:
        """Remove comments and strings - EXACT match to HTML"""
        # Remove multi-line comments
        code = re.sub(r'/\*[\s\S]*?\*/', ' ', code)
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        # Replace strings with empty
        code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', code)
        code = re.sub(r"'(?:[^'\\]|\\.)*'", "''", code)
        return code
    
    # ==================== IS VALID FUNC (matches HTML) ====================
    def is_valid_func(self, name: str) -> bool:
        """Check if name is a valid function name - EXACT match to HTML"""
        if name in self.KEYWORDS:
            return False
        if name[0].isdigit():
            return False
        if len(name) < 2:
            return False
        return True
    
    # ==================== FIND CALLS (matches HTML) ====================
    def find_calls(self, code: str) -> List[str]:
        """Find function calls in code - EXACT match to HTML"""
        calls = []
        for m in re.finditer(r'\b([a-zA-Z_]\w*)\s*\(', code):
            name = m.group(1)
            if self.is_valid_func(name):
                calls.append(name)
        # Return unique calls preserving order
        return list(dict.fromkeys(calls))
    
    # ==================== FIND BRACE (matches HTML) ====================
    def find_brace(self, code: str, start: int, limit: int = 50000) -> int:
        """Find matching brace with limit - EXACT match to HTML"""
        depth = 1
        i = start
        end = min(len(code), start + limit)
        while i < end and depth > 0:
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
            i += 1
        return i if depth == 0 else -1
    
    # ==================== PARSE MACROS (matches HTML) ====================
    def parse_macros(self, code: str, filename: str):
        """Parse macros - EXACT match to HTML logic"""
        # Handle line continuations
        code_exp = re.sub(r'\\\n', ' ', code)
        
        # Function-like: #define NAME(args) body
        # HTML uses: /#\s*define\s+(\w+)\s*\(([^)]*)\)\s*(.+?)(?=\n|$)/g
        for m in re.finditer(r'#\s*define\s+(\w+)\s*\(([^)]*)\)\s*(.+?)(?=\n|$)', code_exp):
            name = m.group(1)
            body = m.group(3).strip()
            calls = self.find_calls(body)
            self.macros[name] = MacroDef(name, filename, body, [], calls)
            if calls:
                self.add_log('macro', f"{name} -> {', '.join(calls)}")
        
        # Object-like: #define NAME value (skip function-like)
        # HTML uses: /#\s*define\s+(\w+)\s+([^(\n][^\n]*)/g
        # Key: [^(\n] means first char after space must NOT be ( or newline
        for m in re.finditer(r'#\s*define\s+(\w+)\s+([^(\n][^\n]*)', code_exp):
            name = m.group(1)
            if name not in self.macros:
                body = m.group(2).strip()
                calls = self.find_calls(body)
                self.macros[name] = MacroDef(name, filename, body, [], calls)
                # Also store as value macro for resolution
                self.value_macros[name] = body
    
    # ==================== PARSE FUNC POINTERS (matches HTML) ====================
    def parse_func_pointers(self, code: str, filename: str):
        """Parse function pointer assignments - EXACT match to HTML"""
        patterns = [
            r'\.(\w+)\s*=\s*&?(\w+)\s*[,;}\)]',
            r'(\w+)\s*=\s*&?(\w+)\s*;'
        ]
        
        for pat in patterns:
            for m in re.finditer(pat, code):
                var_name = m.group(1)
                func_name = m.group(2)
                if self.is_valid_func(func_name):
                    self.func_pointers[var_name].add(func_name)
    
    # ==================== PARSE FUNCTIONS (matches HTML) ====================
    def parse_functions(self, code: str, filename: str):
        """Parse functions - EXACT match to HTML logic"""
        clean = self.clean(code)
        
        # Simpler regex for function definitions - EXACT match to HTML
        for m in re.finditer(r'\b([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*\{', clean):
            name = m.group(1)
            
            if not self.is_valid_func(name):
                continue
            
            body_start = m.end()
            body_end = self.find_brace(clean, body_start)
            
            if body_end == -1:
                continue
            
            body = clean[body_start - 1:body_end]
            direct_calls = self.find_calls(body)
            
            # Parse parameters
            params = []
            params_str = m.group(2).strip()
            if params_str and params_str != 'void':
                for p in params_str.split(','):
                    parts = p.strip().split()
                    if parts:
                        params.append(parts[-1].strip('*[]'))
            
            # ========== KEY DIFFERENCE: Expand macro calls like HTML ==========
            all_calls: Dict[str, CallInfo] = {}
            
            for call in direct_calls:
                if call in self.macros:
                    # Macro invocation
                    macro = self.macros[call]
                    all_calls[call] = CallInfo(CallType.MACRO_INVOKE)
                    # Add functions called by the macro
                    for mc in macro.calls:
                        all_calls[mc] = CallInfo(CallType.MACRO, via=call)
                elif call in self.func_pointers:
                    # Function pointer
                    for target in self.func_pointers[call]:
                        all_calls[target] = CallInfo(CallType.POINTER, via=call)
                else:
                    # Direct call
                    all_calls[call] = CallInfo(CallType.DIRECT)
            
            # ========== KEY DIFFERENCE: Check for callback patterns like HTML ==========
            # Pattern: var->member() where member is a known function pointer field
            for cb in re.finditer(r'(\w+)->(\w+)\s*\(', body):
                member = cb.group(2)
                if member in self.func_pointers:
                    for target in self.func_pointers[member]:
                        all_calls[target] = CallInfo(CallType.CALLBACK, via=member)
            
            # Store function - HTML always overwrites: this.functions.set(name, ...)
            self.functions[name] = FunctionDef(name, filename, body, params, all_calls)
    
    # ==================== BUILD CALL GRAPH (EXACT match to HTML) ====================
    def build_graph(self):
        """Build call graph - EXACT match to HTML buildGraph()"""
        # HTML just does:
        # for (const [name, func] of this.functions) {
        #     this.callGraph.set(name, func.calls);
        # }
        for name, func in self.functions.items():
            self.call_graph[name] = func.calls
        self.add_log('info', f"Built call graph with {len(self.functions)} functions")
    
    # ==================== PARSE FILE (matches HTML) ====================
    def parse_file(self, filepath: str):
        """Parse a single file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            filename = os.path.basename(filepath)
            self.files[filename] = content
            
            self.add_log('info', f"Parsing {filename} ({len(content)} bytes)")
            
            self.parse_macros(content, filename)
            self.parse_func_pointers(content, filename)
            self.parse_functions(content, filename)
            
        except Exception as e:
            self.add_log('error', f"Error parsing {filepath}: {e}")
    
    def load_directory(self, dirpath: str):
        """Load all C/H files from directory"""
        if not os.path.isdir(dirpath):
            return
        
        # Headers first (for macros)
        for root, _, files in os.walk(dirpath):
            for f in files:
                if f.endswith(('.h', '.hpp', '.inc')):
                    self.parse_file(os.path.join(root, f))
        
        # Then source files
        for root, _, files in os.walk(dirpath):
            for f in files:
                if f.endswith(('.c', '.cpp')):
                    self.parse_file(os.path.join(root, f))
    
    # ==================== HELPER METHODS ====================
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
        """Get all assignments to a variable in a function"""
        func = self.functions.get(func_name)
        if not func:
            return []
        
        assignments = []
        
        # Find all assignments
        for m in re.finditer(rf'\b{re.escape(var_name)}\s*=\s*([^;]+);', func.body):
            value = m.group(1).strip()
            before = func.body[:m.start()]
            
            context = "default"
            if re.search(r'if\s*\([^)]+\)\s*\{[^}]*$', before):
                context = "conditional (if)"
            elif re.search(r'else\s*\{[^}]*$', before):
                context = "conditional (else)"
            elif re.search(r'case\s+\w+\s*:[^}]*$', before):
                context = "conditional (switch)"
            
            assignments.append({"value": value, "context": context})
        
        # Check declaration with initialization
        decl_match = re.search(rf'(?:int|long|short|char|\w+_t)\s+{re.escape(var_name)}\s*=\s*([^;]+);', func.body)
        if decl_match:
            assignments.insert(0, {"value": decl_match.group(1).strip(), "context": "initialization"})
        
        return assignments


# ============================================================================
# PATH FINDER - MATCHES HTML LOGIC
# ============================================================================

class PathFinder:
    """Path finder that matches HTML DFS logic exactly"""
    
    def __init__(self, parser: CParser):
        self.parser = parser
    
    def find_paths(self, entry: str, target: str, max_depth: int = 30, 
                   max_paths: int = 100, max_time_ms: int = 5000) -> List[List[PathStep]]:
        """Find all paths - EXACT match to HTML findPaths()"""
        import time
        
        paths = []
        start_time = time.time() * 1000
        
        if entry not in self.parser.call_graph:
            self.parser.add_log('error', f"Entry function '{entry}' not found")
            return paths
        
        def dfs(current: str, path: List[PathStep], visited: Set[str]):
            # Time limit
            if (time.time() * 1000 - start_time) > max_time_ms:
                return
            # Path limit
            if len(paths) >= max_paths:
                return
            # Depth limit
            if len(path) > max_depth:
                return
            # Cycle detection
            if current in visited:
                return
            
            if current == target:
                paths.append(deepcopy(path))
                return
            
            visited.add(current)
            
            calls = self.parser.call_graph.get(current)
            if calls:
                # Iterate like HTML: for (const [callee, info] of calls)
                for callee, info in calls.items():
                    # Get call arguments
                    call_str, call_args = self.parser.get_call_args(current, callee)
                    
                    # Update previous step with call info
                    if path:
                        path[-1].call_to_next = call_str
                        path[-1].args_to_next = call_args
                    
                    # Get callee function info
                    callee_func = self.parser.functions.get(callee)
                    
                    step = PathStep(
                        name=callee,
                        type=info.type,
                        via=info.via,
                        file=callee_func.file if callee_func else None,
                        params=callee_func.params if callee_func else []
                    )
                    
                    path.append(step)
                    dfs(callee, path, visited)
                    path.pop()
                    
                    # Clear call info when backtracking
                    if path:
                        path[-1].call_to_next = None
                        path[-1].args_to_next = []
            
            visited.discard(current)
        
        # Create entry step
        entry_func = self.parser.functions.get(entry)
        entry_step = PathStep(
            name=entry,
            type=CallType.ENTRY,
            file=entry_func.file if entry_func else None,
            params=entry_func.params if entry_func else []
        )
        
        dfs(entry, [entry_step], set())
        
        elapsed = int(time.time() * 1000 - start_time)
        self.parser.add_log('info', f"Found {len(paths)} paths in {elapsed}ms")
        
        return paths
    
    def print_paths(self, paths: List[List[PathStep]], entry: str, target: str):
        """Print paths in a readable format"""
        print(colored(f"\n  🛤️  Found {len(paths)} path(s) from {entry}() to {target}():", Colors.CYAN))
        print()
        
        for i, path in enumerate(paths, 1):
            print(colored(f"  ╔══ Path {i} ══╗", Colors.YELLOW))
            
            for j, step in enumerate(path):
                is_last = (j == len(path) - 1)
                
                # Color based on type
                if step.type == CallType.ENTRY:
                    color = Colors.CYAN
                    prefix = "►"
                elif step.name == target:
                    color = Colors.GREEN
                    prefix = "★"
                elif step.type == CallType.MACRO:
                    color = Colors.HEADER
                    prefix = "M"
                elif step.type == CallType.CALLBACK:
                    color = Colors.YELLOW
                    prefix = "C"
                else:
                    color = Colors.DIM
                    prefix = "│"
                
                params = f"({', '.join(step.params)})" if step.params else "()"
                via = f" [via {step.via}]" if step.via else ""
                
                print(colored(f"  {prefix} [{j}] {step.name}{params}", color), end="")
                if step.file:
                    print(colored(f" [{step.file}]", Colors.DIM), end="")
                print(via)
                
                if step.call_to_next:
                    print(colored(f"      ↓ calls: {step.call_to_next}", Colors.DIM))
                    if step.args_to_next:
                        print(colored(f"      ↓ args: {step.args_to_next}", Colors.DIM))
            
            print()


# ============================================================================
# MFS EXTRACTOR
# ============================================================================

class MFSExtractor:
    """Extract mpf_mfs_open calls from paths"""
    
    def __init__(self, parser: CParser):
        self.parser = parser
    
    def extract_from_paths(self, paths: List[List[PathStep]]) -> List[MFSCallSite]:
        """Extract MFS call sites from found paths"""
        calls = []
        seen = set()
        
        for path in paths:
            if not path:
                continue
            
            # Find the function that calls mpf_mfs_open
            caller_step = None
            for step in path:
                if step.call_to_next and 'mpf_mfs_open' in step.call_to_next:
                    caller_step = step
                    break
            
            if not caller_step:
                # Last step before mpf_mfs_open
                for i, step in enumerate(path):
                    if step.name == 'mpf_mfs_open' and i > 0:
                        caller_step = path[i - 1]
                        break
            
            if not caller_step:
                continue
            
            func = self.parser.functions.get(caller_step.name)
            if not func:
                continue
            
            # Find mpf_mfs_open calls in this function
            for m in re.finditer(r'mpf_mfs_open\s*\(([^)]+)\)', func.body):
                full_call = m.group(0)
                args_str = m.group(1)
                args = self._split_args(args_str)
                
                # 3rd argument is fileno (index 2)
                fileno_raw = args[2].strip() if len(args) > 2 else "UNKNOWN"
                
                key = (caller_step.name, full_call)
                if key in seen:
                    continue
                seen.add(key)
                
                # Get line number
                line = func.body[:m.start()].count('\n') + 1
                
                calls.append(MFSCallSite(
                    file=func.file,
                    line=line,
                    function=caller_step.name,
                    full_call=full_call,
                    args=args,
                    path=path[:-1],  # Exclude mpf_mfs_open itself
                    fileno_raw=fileno_raw
                ))
        
        return calls
    
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


# ============================================================================
# FULL CONTEXT RESOLVER
# ============================================================================

class FullContextResolver:
    """Resolver with full pre-extracted context for LLM"""
    
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_variable_assignments",
                "description": "Get all assignments to a variable in a function",
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
                "description": "Get the value of a macro definition",
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
                "description": "Evaluate arithmetic expression with macros",
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
                "description": "Submit the final resolved value(s)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values": {"type": "string"},
                        "trace": {"type": "string"}
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
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM client"""
        if HAS_GROQ:
            try:
                gk = os.environ.get("GROQ_API_KEY", "")
                if gk:
                    self.client = Groq(api_key=gk)
                    self.model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
                    self.provider = "groq"
                    if self.verbose:
                        print(colored(f"    🚀 Using Groq ({self.model})", Colors.GREEN))
                    return
            except Exception as e:
                if self.verbose:
                    print(colored(f"    ⚠️ Groq init failed: {e}", Colors.YELLOW))
        
        if HAS_OPENAI:
            try:
                ae = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
                ak = os.environ.get("AZURE_OPENAI_API_KEY", "")
                ad = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
                
                if self.verbose:
                    print(colored(f"    📋 Azure config check:", Colors.DIM))
                    print(colored(f"       ENDPOINT: {'✓ set' if ae else '✗ missing'}", Colors.DIM))
                    print(colored(f"       API_KEY: {'✓ set' if ak else '✗ missing'}", Colors.DIM))
                    print(colored(f"       DEPLOYMENT: {'✓ set' if ad else '✗ missing'}", Colors.DIM))
                
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
                else:
                    if self.verbose:
                        if "your-resource" in ae:
                            print(colored(f"    ⚠️ Azure endpoint contains placeholder", Colors.YELLOW))
                        if not all([ae, ak, ad]):
                            print(colored(f"    ⚠️ Azure config incomplete", Colors.YELLOW))
            except Exception as e:
                if self.verbose:
                    print(colored(f"    ⚠️ Azure init failed: {e}", Colors.YELLOW))
            
            try:
                ok = os.environ.get("OPENAI_API_KEY", "")
                if ok:
                    self.client = OpenAI(api_key=ok)
                    self.model = os.environ.get("OPENAI_MODEL", "gpt-4o")
                    self.provider = "openai"
                    if self.verbose:
                        print(colored(f"    🟢 Using OpenAI ({self.model})", Colors.GREEN))
            except Exception as e:
                if self.verbose:
                    print(colored(f"    ⚠️ OpenAI init failed: {e}", Colors.YELLOW))
        
        if not self.client and self.verbose:
            print(colored(f"    ❌ No LLM provider available!", Colors.RED))
            print(colored(f"       Set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT in .env", Colors.RED))
    
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
        
        paren_match = re.match(r'^\((.+)\)$', value)
        if paren_match:
            return self._try_simple_resolve(paren_match.group(1), depth + 1)
        
        if value in self.parser.value_macros:
            return self._try_simple_resolve(self.parser.value_macros[value], depth + 1)
        
        # Try arithmetic
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
            return f"{macro_name}(...) = {m.body}  [function macro]"
        
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
        
        for i, step in enumerate(mfs_call.path):
            params_str = f"({', '.join(step.params)})" if step.params else "()"
            
            lines.append(f"[{i+1}] {step.name}{params_str}")
            lines.append(f"    File: {step.file}")
            
            if step.params:
                lines.append(f"    Parameters: {step.params}")
            
            if step.via:
                lines.append(f"    Called via: {step.via} ({step.type.value})")
            
            if step.call_to_next:
                lines.append(f"    ↓ Calls: {step.call_to_next}")
                
                if i + 1 < len(mfs_call.path):
                    next_step = mfs_call.path[i + 1]
                    if next_step.params and step.args_to_next:
                        mappings = []
                        for j, arg in enumerate(step.args_to_next):
                            if j < len(next_step.params):
                                mappings.append(f"{next_step.params[j]}={arg}")
                        if mappings:
                            lines.append(f"    ↓ Arg mapping: {', '.join(mappings)}")
            
            lines.append("")
        
        lines.append(f"[{len(mfs_call.path)+1}] mpf_mfs_open(..., {mfs_call.fileno_raw}, ...)")
        lines.append(f"    File: {mfs_call.file}:{mfs_call.line}")
        lines.append(f"    3rd argument (fileno): {mfs_call.fileno_raw}")
        lines.append("")
        
        lines.append("=" * 60)
        lines.append("POTENTIALLY RELEVANT MACROS")
        lines.append("=" * 60)
        
        relevant_macros = set()
        for step in mfs_call.path:
            if step.args_to_next:
                for arg in step.args_to_next:
                    for name, value in self.parser.value_macros.items():
                        if name in arg or name == arg:
                            resolved = self._try_simple_resolve(value)
                            if resolved:
                                relevant_macros.add(f"  {name} = {value} → {resolved}")
                            else:
                                relevant_macros.add(f"  {name} = {value}")
        
        if mfs_call.fileno_raw in self.parser.value_macros:
            value = self.parser.value_macros[mfs_call.fileno_raw]
            resolved = self._try_simple_resolve(value)
            if resolved:
                relevant_macros.add(f"  {mfs_call.fileno_raw} = {value} → {resolved}")
        
        if relevant_macros:
            lines.extend(list(relevant_macros))
        else:
            lines.append("  (use get_macro_value tool if needed)")
        
        lines.append("")
        return "\n".join(lines)
    
    def _build_backward_trace(self, mfs_call: MFSCallSite) -> str:
        """Build backward trace showing the argument chain"""
        lines = []
        lines.append("BACKWARD TRACE (what you need to resolve):")
        lines.append("")
        
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
                if current_var in step.params:
                    param_idx = step.params.index(current_var)
                    lines.append(f"  └─ '{current_var}' is PARAMETER[{param_idx}] of {step.name}()")
                    
                    if i > 0:
                        caller = mfs_call.path[i - 1]
                        if caller.args_to_next and param_idx < len(caller.args_to_next):
                            arg_passed = caller.args_to_next[param_idx]
                            lines.append(f"     └─ {caller.name}() passes: {arg_passed}")
                            current_var = arg_passed
                            current_func = caller.name
                            
                            if re.match(r'^-?\d+$', current_var):
                                lines.append(f"  └─ '{current_var}' is a DIRECT NUMBER")
                                lines.append(f"     └─ ✅ RESOLVED: {current_var}")
                                break
                            
                            if current_var in self.parser.value_macros:
                                resolved = self._try_simple_resolve(current_var)
                                if resolved:
                                    lines.append(f"  └─ '{current_var}' is a MACRO = {resolved}")
                                    lines.append(f"     └─ ✅ RESOLVED: {resolved}")
                                else:
                                    lines.append(f"  └─ '{current_var}' is a MACRO = {self.parser.value_macros[current_var]}")
                                break
                elif current_var in self.parser.value_macros:
                    resolved = self._try_simple_resolve(current_var)
                    if resolved:
                        lines.append(f"  └─ '{current_var}' is a MACRO = {resolved}")
                        lines.append(f"     └─ ✅ RESOLVED: {resolved}")
                    else:
                        lines.append(f"  └─ '{current_var}' is a MACRO = {self.parser.value_macros[current_var]}")
                    break
                else:
                    lines.append(f"  └─ '{current_var}' is LOCAL VARIABLE in {step.name}()")
                    lines.append(f"     └─ Use get_variable_assignments(\"{step.name}\", \"{current_var}\")")
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
        
        # Only skip LLM for direct numbers like "2001"
        if re.match(r'^-?\d+$', raw_value):
            print(colored(f"    ✓ Direct number: {raw_value}", Colors.GREEN))
            return raw_value
        
        if not self.client:
            print(colored("    ✗ No LLM available", Colors.RED))
            return f"{raw_value} (unresolved - no LLM)"
        
        # Let LLM do ALL the resolution work
        print(colored("    🤖 LLM resolving...", Colors.YELLOW))
        return self._resolve_with_llm(mfs_call)
    
    def _resolve_with_llm(self, mfs_call: MFSCallSite, max_iterations: int = 10) -> str:
        full_context = self._build_full_context(mfs_call)
        backward_trace = self._build_backward_trace(mfs_call)
        
        system_prompt = """You are a C code analyzer. Your task is to find the numeric value of the fileno (3rd) argument passed to mpf_mfs_open().

IMPORTANT: Think step by step before taking any action.

AVAILABLE TOOLS:
1. get_variable_assignments(function_name, variable_name) - Get all assignments to a variable in a function
2. get_macro_value(macro_name) - Get the value/definition of a macro
3. evaluate_expression(expression) - Evaluate arithmetic expression (e.g., "7000 + 1")
4. submit_answer(values, trace) - Submit your final answer with the resolution trace

STRATEGY:
1. First, THINK about what you see in the call chain and backward trace
2. Identify where the value comes from:
   - If it's a PARAMETER, look at what the caller passes
   - If it's a LOCAL VARIABLE, use get_variable_assignments()
   - If it's a MACRO (usually ALL_CAPS), use get_macro_value()
   - If it involves arithmetic, use evaluate_expression()
3. Chain your reasoning - follow the value back to its source
4. Once you have the final numeric value, use submit_answer()

THINKING FORMAT:
Before each action, explain your reasoning:
- What do I know so far?
- What do I need to find out?
- Which tool should I use and why?

Always submit your final answer using submit_answer(values="<number>", trace="<how you got there>")"""

        user_prompt = f"""Resolve the numeric value of '{mfs_call.fileno_raw}' in mpf_mfs_open().

{full_context}
{backward_trace}

Think step by step and resolve to the final numeric value."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        for iteration in range(max_iterations):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.TOOLS,
                    tool_choice="auto"
                )
                
                msg = response.choices[0].message
                
                # Print LLM's thinking if present
                if msg.content:
                    # Show abbreviated thinking
                    thinking = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    print(colored(f"      💭 {thinking}", Colors.DIM))
                
                if msg.tool_calls:
                    # Build assistant message with content + tool_calls
                    assistant_msg = {
                        "role": "assistant",
                        "content": msg.content,  # Include thinking
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in msg.tool_calls
                        ]
                    }
                    messages.append(assistant_msg)
                    
                    # Process each tool call
                    for tc in msg.tool_calls:
                        func_name = tc.function.name
                        func_args = json.loads(tc.function.arguments)
                        
                        print(colored(f"      🔧 [{iteration+1}] {func_name}({func_args})", Colors.YELLOW))
                        
                        result = self.execute_tool(func_name, func_args)
                        
                        print(colored(f"         → {result[:100]}{'...' if len(result) > 100 else ''}", Colors.DIM))
                        
                        # Check if this is the final answer
                        if result.startswith("ANSWER:"):
                            answer = result[7:]
                            print(colored(f"    ✓ RESOLVED: {answer}", Colors.GREEN))
                            return answer
                        
                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result
                        })
                else:
                    # No tool calls - LLM responded with text only
                    if msg.content:
                        # Try to extract number from response
                        numbers = re.findall(r'\b\d{4,}\b', msg.content)  # 4+ digit numbers likely fileno
                        if numbers:
                            print(colored(f"    ✓ Extracted from response: {numbers[-1]}", Colors.GREEN))
                            return numbers[-1]
                        
                        # Add response to messages and prompt for tool use
                        messages.append({"role": "assistant", "content": msg.content})
                        messages.append({
                            "role": "user", 
                            "content": "Please use submit_answer() tool to submit your final numeric answer."
                        })
                    else:
                        break
                    
            except Exception as e:
                print(colored(f"    ✗ LLM error: {e}", Colors.RED))
                break
        
        return f"{mfs_call.fileno_raw} (unresolved)"



# ============================================================================
# MAIN
# ============================================================================

def main():
    print(colored("\n" + "=" * 70, Colors.CYAN))
    print(colored("  FileNo Agent v17.0 - HTML-Matched Parser", Colors.CYAN + Colors.BOLD))
    print(colored("=" * 70, Colors.CYAN))
    
    if len(sys.argv) < 2:
        print(f"\nUsage: {sys.argv[0]} <directory> [include_dir] [--entry=main] [--target=mpf_mfs_open]")
        print("\nExamples:")
        print(f"  {sys.argv[0]} ./src")
        print(f"  {sys.argv[0]} ./src ./include")
        print(f"  {sys.argv[0]} ./src --entry=app_main --target=mfs_open")
        sys.exit(1)
    
    # Parse arguments
    directories = []
    entry_func = "main"
    target_func = "mpf_mfs_open"
    
    for arg in sys.argv[1:]:
        if arg.startswith("--entry="):
            entry_func = arg.split("=", 1)[1]
        elif arg.startswith("--target="):
            target_func = arg.split("=", 1)[1]
        elif not arg.startswith("-"):
            directories.append(arg)
    
    if not directories:
        print(colored("Error: No directory specified", Colors.RED))
        sys.exit(1)
    
    # Parse files
    print(colored("\n📁 Loading files...", Colors.YELLOW))
    parser = CParser(verbose=True)
    
    for d in directories:
        if os.path.isdir(d):
            print(colored(f"  Loading: {d}", Colors.DIM))
            parser.load_directory(d)
        elif os.path.isfile(d):
            print(colored(f"  Loading: {d}", Colors.DIM))
            parser.parse_file(d)
        else:
            print(colored(f"  ⚠️ Not found: {d}", Colors.YELLOW))
    
    print(colored(f"\n📊 Parsed: {len(parser.files)} files, {len(parser.functions)} functions, {len(parser.macros)} macros", Colors.GREEN))
    
    # Build call graph
    print(colored("\n🔗 Building call graph...", Colors.YELLOW))
    parser.build_graph()
    
    # Find paths
    print(colored(f"\n🔍 Finding paths: {entry_func}() → {target_func}()", Colors.YELLOW))
    finder = PathFinder(parser)
    paths = finder.find_paths(entry_func, target_func)
    
    if not paths:
        print(colored(f"\n❌ No paths found from {entry_func}() to {target_func}()", Colors.RED))
        print(colored("\nDebug info:", Colors.YELLOW))
        print(f"  Entry '{entry_func}' exists: {entry_func in parser.functions}")
        print(f"  Target '{target_func}' in call graph: {any(target_func in calls for calls in parser.call_graph.values())}")
        
        if entry_func in parser.functions:
            func = parser.functions[entry_func]
            print(f"  {entry_func}() calls: {list(func.calls.keys())[:10]}")
        sys.exit(1)
    
    # Print paths
    finder.print_paths(paths, entry_func, target_func)
    
    # Extract MFS calls
    print(colored("\n📍 Extracting mpf_mfs_open() calls...", Colors.YELLOW))
    extractor = MFSExtractor(parser)
    mfs_calls = extractor.extract_from_paths(paths)
    
    print(colored(f"  Found {len(mfs_calls)} unique call site(s)", Colors.GREEN))
    
    if not mfs_calls:
        print(colored("\n⚠️ No mpf_mfs_open calls found in paths", Colors.YELLOW))
        sys.exit(0)
    
    # Initialize resolver
    print(colored("\n🤖 Initializing resolver...", Colors.YELLOW))
    resolver = FullContextResolver(parser, verbose=True)
    
    # Resolve each call
    print(colored("\n" + "=" * 70, Colors.CYAN))
    print(colored("  RESOLVING FILENO VALUES", Colors.CYAN + Colors.BOLD))
    print(colored("=" * 70, Colors.CYAN))
    
    results = []
    for i, call in enumerate(mfs_calls, 1):
        print(colored(f"\n[{i}/{len(mfs_calls)}] {call.file}:{call.line}", Colors.YELLOW))
        print(colored(f"    Call: {call.full_call}", Colors.DIM))
        
        resolved = resolver.resolve(call)
        call.fileno_resolved = resolved
        results.append(call)
    
    # Summary
    print(colored("\n" + "=" * 70, Colors.CYAN))
    print(colored("  SUMMARY", Colors.CYAN + Colors.BOLD))
    print(colored("=" * 70, Colors.CYAN))
    
    for call in results:
        status = "✓" if call.fileno_resolved and "unresolved" not in call.fileno_resolved else "✗"
        color = Colors.GREEN if status == "✓" else Colors.RED
        print(colored(f"\n  {status} {call.file}:{call.line} in {call.function}()", color))
        print(colored(f"    Path: {' → '.join(s.name for s in call.path)}", Colors.DIM))
        print(colored(f"    fileno: {call.fileno_raw} → {call.fileno_resolved}", color))
    
    # Stats
    resolved_count = sum(1 for c in results if c.fileno_resolved and "unresolved" not in c.fileno_resolved)
    print(colored(f"\n📊 Resolved: {resolved_count}/{len(results)}", Colors.GREEN if resolved_count == len(results) else Colors.YELLOW))


if __name__ == "__main__":
    main()

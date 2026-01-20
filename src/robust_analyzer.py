"""
Ultimate Robust Data Flow Analyzer v3.0
Handles ALL patterns:
- Nested macros (A → B → C → value)
- Multi-level function tracing (main → func1 → func2 → mpf_mfs_open)
- Switch-case statements
- Complex if-else chains
- Cross-file analysis
- Full parameter tracking through call chains
- Arithmetic expressions
"""
import os
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field

from macro_extractor import MacroExtractor
from ast_parser import ASTParser, FunctionCall


@dataclass
class AnalysisResult:
    """Result of analyzing a single mpf_mfs_open call."""
    file_path: str
    line_number: int
    containing_function: str
    raw_argument: str
    resolved_value: Optional[int]
    confidence: float
    resolution_chain: List[str]
    all_possible_values: List[int] = field(default_factory=list)
    analysis_method: str = "AST"


class UltimateMacroResolver:
    """Fully resolves nested macros and expressions."""
    
    def __init__(self):
        self.macros: Dict[str, str] = {}
        
    def add_macro(self, name: str, value: str):
        self.macros[name] = value.strip()
    
    def extract_from_content(self, content: str):
        """Extract all macro definitions from content."""
        patterns = [
            r'#\s*define\s+(\w+)\s+\(([^)]+)\)',
            r'#\s*define\s+(\w+)\s+(\d+)',
            r'#\s*define\s+(\w+)\s+([A-Z_][A-Z0-9_]*)',
            r'#\s*define\s+(\w+)\s+([A-Z_][A-Z0-9_]*\s*[-+*/]\s*\d+)',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                name, value = match.groups()
                if name not in self.macros:
                    self.macros[name] = value.strip()
    
    def extract_from_file(self, filepath: str):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                self.extract_from_content(f.read())
        except:
            pass
    
    def extract_from_directory(self, dirpath: str):
        if not os.path.isdir(dirpath):
            return
        for root, _, files in os.walk(dirpath):
            for f in files:
                if f.endswith(('.h', '.c')):
                    self.extract_from_file(os.path.join(root, f))
    
    def resolve(self, value: str, visited: Set[str] = None) -> Tuple[Optional[int], List[str]]:
        """Fully resolve a value to numeric. Returns (value, trace)."""
        if visited is None:
            visited = set()
        
        trace = []
        value = str(value).strip()
        
        # Remove parentheses
        while value.startswith('(') and value.endswith(')'):
            value = value[1:-1].strip()
        
        # Direct number
        try:
            num = int(value)
            trace.append(f"{value}")
            return num, trace
        except ValueError:
            pass
        
        # Single macro
        if value in self.macros and value not in visited:
            visited.add(value)
            trace.append(f"{value}")
            result, sub_trace = self.resolve(self.macros[value], visited)
            trace.extend(sub_trace)
            return result, trace
        
        # Expression with operators
        if re.search(r'[-+*/]', value):
            return self._eval_expr(value, visited, trace)
        
        return None, [f"Unknown: {value}"]
    
    def _eval_expr(self, expr: str, visited: Set[str], trace: List[str]) -> Tuple[Optional[int], List[str]]:
        """Evaluate expression with macro substitution."""
        result_expr = expr
        trace.append(f"Expr: {expr}")
        
        # Find and replace all macros
        for macro in re.findall(r'[A-Z_][A-Z0-9_]*', expr):
            if macro in self.macros and macro not in visited:
                visited_copy = visited.copy()
                visited_copy.add(macro)
                resolved, _ = self.resolve(self.macros[macro], visited_copy)
                if resolved is not None:
                    result_expr = re.sub(r'\b' + macro + r'\b', str(resolved), result_expr)
        
        # Evaluate
        try:
            clean = re.sub(r'[()]', '', result_expr)
            result = int(eval(clean, {"__builtins__": {}}, {}))
            trace.append(f"= {result}")
            return result, trace
        except:
            return None, trace


class UltimateDataFlowAnalyzer:
    """Ultimate analyzer handling all data flow patterns."""
    
    def __init__(self, include_paths: List[str] = None):
        self.include_paths = include_paths or []
        self.macro_resolver = UltimateMacroResolver()
        self.ast_parser = ASTParser(include_paths)
        
        # Storage
        self.file_contents: Dict[str, str] = {}
        self.function_bodies: Dict[str, str] = {}
        self.function_files: Dict[str, str] = {}
        self.function_params: Dict[str, List[str]] = {}
        self.function_calls: Dict[str, List[Tuple[str, List[str]]]] = {}  # func -> [(called, [args])]
        self.function_declarations: Dict[str, str] = {}  # func_name -> header_file (prototypes from .h)
        self.undeclared_functions: List[Tuple[str, str, str]] = []  # (func_name, called_from, file)
        
        # Standard library functions that don't need declaration check
        self.stdlib_functions = {
            'printf', 'fprintf', 'sprintf', 'snprintf', 'scanf', 'fscanf', 'sscanf',
            'malloc', 'calloc', 'realloc', 'free',
            'memcpy', 'memset', 'memmove', 'memcmp',
            'strcpy', 'strncpy', 'strcat', 'strncat', 'strcmp', 'strncmp', 'strlen', 'strstr',
            'fopen', 'fclose', 'fread', 'fwrite', 'fgets', 'fputs', 'fseek', 'ftell',
            'atoi', 'atol', 'atof', 'strtol', 'strtod',
            'exit', 'abort', 'system', 'getenv',
            'abs', 'labs', 'rand', 'srand',
            'time', 'clock', 'difftime', 'mktime',
            'isalpha', 'isdigit', 'isalnum', 'isspace', 'toupper', 'tolower',
            'assert',
        }
        
    def analyze_project(self, project_dir: str) -> List[AnalysisResult]:
        """Main entry point - starts from main() and traces all reachable calls."""
        # Reset undeclared functions list
        self.undeclared_functions = []
        
        # Phase 1: Load everything
        self._load_all(project_dir)
        
        # Phase 2: Find main() function and build reachability
        main_func = self._find_main_function()
        if main_func is None:
            # No main() found - return empty or warn
            return []
        
        # Phase 3: Build reachable functions from main() (includes validation)
        reachable_funcs = self._get_reachable_functions(main_func)
        
        # Phase 3.5: Report undeclared functions as warnings
        if self.undeclared_functions:
            print("\n⚠️  UNDECLARED FUNCTION WARNINGS:")
            print("=" * 60)
            for func_name, called_from, file_path in self.undeclared_functions:
                print(f"  ❌ '{func_name}' called from '{called_from}()' in {os.path.basename(file_path)}")
                print(f"     → Not defined in any .c file")
                print(f"     → Not declared in any .h file")
            print("=" * 60)
            print()
        
        # Phase 4: Find mpf_mfs_open calls only in reachable functions
        mfs_calls = self.ast_parser.find_function_calls("mpf_mfs_open")
        
        # Filter to only reachable calls
        reachable_calls = [
            call for call in mfs_calls 
            if call.containing_function in reachable_funcs
        ]
        
        # Phase 5: Resolve each call
        results = []
        for call in reachable_calls:
            result = self._analyze_call(call)
            results.append(result)
        
        return results
    
    def _find_main_function(self) -> Optional[str]:
        """Find the main() function in the project."""
        # Check if 'main' is in our function bodies
        if 'main' in self.function_bodies:
            return 'main'
        
        # Also search for variations like 'int main', 'void main'
        for func_name in self.function_bodies.keys():
            if func_name == 'main':
                return func_name
        
        return None
    
    def _get_reachable_functions(self, start_func: str) -> Set[str]:
        """Get all functions reachable from start_func using BFS."""
        reachable = set()
        queue = [start_func]
        checked_calls = set()  # Track which calls we've validated
        
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            
            reachable.add(current)
            
            # Add all functions called by current
            if current in self.function_calls:
                for called_func, _ in self.function_calls[current]:
                    # Skip if already checked
                    call_key = (current, called_func)
                    if call_key in checked_calls:
                        continue
                    checked_calls.add(call_key)
                    
                    # Validate the called function
                    is_valid = self._validate_function_call(called_func, current)
                    
                    # Only add to queue if it has a body (defined in project)
                    if called_func not in reachable and called_func in self.function_bodies:
                        queue.append(called_func)
        
        return reachable
    
    def _validate_function_call(self, func_name: str, called_from: str) -> bool:
        """
        Validate that a called function is properly declared.
        Returns True if valid, False if undeclared.
        
        A function is valid if:
        1. It's defined in a .c file (has body)
        2. It's declared in a .h file (prototype)
        3. It's a standard library function
        4. It's a macro (will be resolved elsewhere)
        """
        # Check if it's a standard library function
        if func_name in self.stdlib_functions:
            return True
        
        # Check if it's a macro (defined in macro_resolver)
        if func_name in self.macro_resolver.macros:
            return True
        
        # Skip if it looks like a macro (all uppercase or common macros)
        if func_name.isupper() or func_name in ['NULL', 'TRUE', 'FALSE']:
            return True
        
        # Check if it's defined in a .c file
        if func_name in self.function_bodies:
            return True
        
        # Check if it's declared in a .h file
        if func_name in self.function_declarations:
            return True
        
        # Not found anywhere - this is an error!
        caller_file = self.function_files.get(called_from, "unknown")
        self.undeclared_functions.append((func_name, called_from, caller_file))
        return False
    
    def get_undeclared_functions(self) -> List[Tuple[str, str, str]]:
        """Get list of undeclared functions found during analysis."""
        return self.undeclared_functions
        
        return reachable
    
    def _load_all(self, project_dir: str):
        """Load all files, macros, and function info."""
        # Directories to scan
        dirs = [project_dir] + [d for d in self.include_paths if os.path.isdir(d)]
        
        for scan_dir in dirs:
            for root, _, files in os.walk(scan_dir):
                for f in files:
                    if f.endswith(('.c', '.h')):
                        path = os.path.join(root, f)
                        self._load_file(path)
        
        # Parse C files for AST
        for root, _, files in os.walk(project_dir):
            for f in files:
                if f.endswith('.c'):
                    self.ast_parser.parse_file(os.path.join(root, f))
    
    def _load_file(self, filepath: str):
        """Load a single file and extract all info."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return
        
        self.file_contents[filepath] = content
        
        # Extract macros
        self.macro_resolver.extract_from_content(content)
        
        # Extract function declarations from header files
        if filepath.endswith('.h'):
            self._extract_declarations(filepath, content)
        
        # Extract functions (only from .c files)
        if filepath.endswith('.c'):
            self._extract_functions(filepath, content)
    
    def _extract_declarations(self, filepath: str, content: str):
        """Extract function prototypes/declarations from header files."""
        # Pattern for function prototypes (declaration without body, ends with ;)
        # This handles complex cases like function pointers in parameters
        # Examples: int mpf_mfs_open(MPF_MFS_FCB *fcb, ...);
        #           void pmf_startproc(int argc, char **argv, void *data);
        #           int pmf_addevent(int eventno, void (*handler)(PMF_EVNHEAD*, void*), int size);
        
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # Skip if not a potential function declaration
            if not line or line.startswith('//') or line.startswith('/*') or line.startswith('#'):
                continue
            
            # Must end with );
            if not line.endswith(');'):
                continue
            
            # Skip typedefs
            if line.startswith('typedef'):
                continue
            
            # Pattern: return_type func_name(...);
            # Match: optional qualifiers, return type, function name, then parentheses
            match = re.match(r'(?:extern\s+)?(?:static\s+)?(?:inline\s+)?(\w+(?:\s*\*)?)\s+(\w+)\s*\(', line)
            if match:
                return_type = match.group(1)
                func_name = match.group(2)
                
                # Skip if it looks like a variable declaration or keyword
                if func_name in ['if', 'while', 'for', 'switch', 'return', 'sizeof']:
                    continue
                
                # Skip type casts
                if func_name in ['int', 'char', 'void', 'long', 'short', 'unsigned', 'float', 'double']:
                    continue
                
                # Store the declaration
                if func_name not in self.function_declarations:
                    self.function_declarations[func_name] = filepath
    
    def _extract_functions(self, filepath: str, content: str):
        """Extract all functions from a file."""
        # Pattern for function definitions
        func_pattern = r'(?:static\s+)?(?:int|void|char|long|short|unsigned|float|double)\s+(\w+)\s*\(([^)]*)\)\s*\{'
        
        for match in re.finditer(func_pattern, content):
            func_name = match.group(1)
            params_str = match.group(2)
            
            # Parse parameters
            params = []
            for p in params_str.split(','):
                p = p.strip()
                if p and p != 'void':
                    # Get last word (the parameter name)
                    parts = re.sub(r'[*]', ' ', p).split()
                    if parts:
                        params.append(parts[-1])
            
            self.function_params[func_name] = params
            self.function_files[func_name] = filepath
            
            # Extract body
            start = match.end() - 1
            brace = 1
            end = start + 1
            while end < len(content) and brace > 0:
                if content[end] == '{':
                    brace += 1
                elif content[end] == '}':
                    brace -= 1
                end += 1
            
            self.function_bodies[func_name] = content[start:end]
            
            # Extract function calls made by this function
            self._extract_calls(func_name, self.function_bodies[func_name])
    
    def _extract_calls(self, caller: str, body: str):
        """Extract all function calls and function pointers from a function body."""
        calls = []
        
        # First, extract all local variables declared in this function
        local_vars = set()
        var_pattern = r'(?:int|char|void|long|short|unsigned|float|double|MPF_\w+)\s+[*]*\s*(\w+)\s*[;=\[]'
        for match in re.finditer(var_pattern, body):
            local_vars.add(match.group(1))
        
        # Also add common parameter names
        common_params = {'argc', 'argv', 'fcb', 'ret', 'fileno', 'type', 'data', 'buf', 'len', 'size', 'ptr', 'p', 'i', 'j', 'n'}
        
        # Pattern: func(args)
        pattern = r'(\w+)\s*\(([^)]*)\)'
        
        for match in re.finditer(pattern, body):
            func = match.group(1)
            args_str = match.group(2)
            
            # Skip keywords
            if func in ['if', 'while', 'for', 'switch', 'return', 'sizeof']:
                continue
            
            # Skip if it's a known local variable (casting like (int)x)
            if func in local_vars:
                continue
            
            # Skip common type casts
            if func in ['int', 'char', 'void', 'long', 'short', 'unsigned', 'float', 'double']:
                continue
            
            # Parse arguments
            args = [a.strip() for a in args_str.split(',') if a.strip()]
            calls.append((func, args))
            
            # Check if any argument is a function pointer (callback)
            # Only add if it looks like a real function name and is defined in our project
            for arg in args:
                # Clean the argument (remove casts, &, etc.)
                clean_arg = re.sub(r'[&*()]', '', arg).strip()
                
                # Skip if it's a local variable
                if clean_arg in local_vars or clean_arg in common_params:
                    continue
                
                # Skip if it's a macro (all uppercase)
                if clean_arg.isupper() or clean_arg == 'NULL':
                    continue
                
                # Skip if it's a number or string
                if clean_arg.isdigit() or clean_arg.startswith('"'):
                    continue
                
                # Only add if it matches our known function definitions
                if clean_arg in self.function_bodies:
                    calls.append((clean_arg, []))
        
        self.function_calls[caller] = calls
    
    def _analyze_call(self, call: FunctionCall) -> AnalysisResult:
        """Analyze a single mpf_mfs_open call."""
        chain = []
        
        if len(call.arguments) < 3:
            return AnalysisResult(
                file_path=call.file_path,
                line_number=call.line_number,
                containing_function=call.containing_function,
                raw_argument="UNKNOWN",
                resolved_value=None,
                confidence=0.0,
                resolution_chain=["Less than 3 arguments"]
            )
        
        raw_arg = call.arguments[2].strip()
        chain.append(f"Found: {raw_arg} at line {call.line_number}")
        
        # Context: parameter values we know
        context = {}
        
        # If the argument is a function parameter, trace callers
        if call.containing_function in self.function_params:
            params = self.function_params[call.containing_function]
            if raw_arg in params:
                param_idx = params.index(raw_arg)
                chain.append(f"{raw_arg} is parameter #{param_idx} of {call.containing_function}")
                
                # Trace back through callers
                value = self._trace_param_from_callers(
                    call.containing_function, param_idx, chain, set()
                )
                if value is not None:
                    return AnalysisResult(
                        file_path=call.file_path,
                        line_number=call.line_number,
                        containing_function=call.containing_function,
                        raw_argument=raw_arg,
                        resolved_value=value,
                        confidence=0.9,
                        resolution_chain=chain
                    )
        
        # Try to resolve directly
        resolved, conf = self._resolve(
            raw_arg, call.file_path, call.containing_function, 
            call.line_number, chain, context, set()
        )
        
        return AnalysisResult(
            file_path=call.file_path,
            line_number=call.line_number,
            containing_function=call.containing_function,
            raw_argument=raw_arg,
            resolved_value=resolved,
            confidence=conf,
            resolution_chain=chain
        )
    
    def _resolve(self, value: str, file_path: str, func: str, line: int,
                 chain: List[str], context: Dict[str, int], visited: Set[str],
                 depth: int = 0) -> Tuple[Optional[int], float]:
        """Resolve any value to numeric."""
        if depth > 30:
            return None, 0.0
        
        value = str(value).strip()
        while value.startswith('(') and value.endswith(')'):
            value = value[1:-1].strip()
        
        # 1. Direct number
        try:
            return int(value), 1.0
        except:
            pass
        
        # 2. In context
        if value in context:
            chain.append(f"From context: {value} = {context[value]}")
            return context[value], 0.95
        
        # 3. Macro
        if value in self.macro_resolver.macros:
            result, trace = self.macro_resolver.resolve(value)
            chain.append(f"Macro: {' → '.join(trace)}")
            if result is not None:
                return result, 0.95
        
        # 4. Expression
        if re.search(r'[-+*/]', value):
            result, trace = self.macro_resolver.resolve(value)
            if result is not None:
                chain.append(f"Expression: {' → '.join(trace)}")
                return result, 0.9
        
        # 5. Variable - trace assignment
        var_result = self._trace_variable(value, file_path, func, line, chain, context, visited, depth)
        if var_result is not None:
            return var_result, 0.9
        
        return None, 0.0
    
    def _trace_variable(self, var: str, file_path: str, func: str, before_line: int,
                        chain: List[str], context: Dict[str, int], visited: Set[str],
                        depth: int) -> Optional[int]:
        """Trace a variable to find its value."""
        content = self.file_contents.get(file_path, "")
        if not content:
            return None
        
        lines = content.split('\n')
        
        # Search backwards
        for i in range(min(before_line - 1, len(lines) - 1), -1, -1):
            line = lines[i]
            
            # Skip mpf_mfs_open line
            if 'mpf_mfs_open' in line:
                continue
            
            # Pattern: var = func(args);
            m = re.search(rf'\b{re.escape(var)}\s*=\s*(\w+)\s*\(([^)]*)\)\s*;', line)
            if m:
                func_name, args_str = m.groups()
                
                if func_name in visited:
                    continue
                
                chain.append(f"Variable {var} = {func_name}({args_str})")
                
                # Resolve arguments
                args = self._resolve_args(args_str, file_path, i + 1, context)
                chain.append(f"Resolved args: {args}")
                
                # Analyze the function
                new_visited = visited | {func_name}
                result = self._analyze_function(func_name, args, chain, new_visited, depth + 1)
                if result is not None:
                    return result
            
            # Pattern: var = value;
            m = re.search(rf'\b{re.escape(var)}\s*=\s*([^;(]+);', line)
            if m:
                assigned = m.group(1).strip()
                
                if '"' in assigned:
                    continue
                
                chain.append(f"Variable {var} = {assigned}")
                
                # Direct number
                try:
                    return int(assigned)
                except:
                    pass
                
                # Macro or expression
                result, _ = self.macro_resolver.resolve(assigned)
                if result is not None:
                    chain.append(f"Resolved to: {result}")
                    return result
        
        return None
    
    def _resolve_args(self, args_str: str, file_path: str, before_line: int,
                      context: Dict[str, int]) -> List[Any]:
        """Resolve function arguments."""
        args = []
        content = self.file_contents.get(file_path, "")
        lines = content.split('\n') if content else []
        
        for arg in args_str.split(','):
            arg = arg.strip()
            if not arg:
                continue
            
            # Direct number
            try:
                args.append(int(arg))
                continue
            except:
                pass
            
            # Context
            if arg in context:
                args.append(context[arg])
                continue
            
            # Macro
            result, _ = self.macro_resolver.resolve(arg)
            if result is not None:
                args.append(result)
                continue
            
            # Variable - search backwards
            for i in range(min(before_line - 1, len(lines) - 1), -1, -1):
                m = re.search(rf'\b{re.escape(arg)}\s*=\s*([^;(]+);', lines[i])
                if m:
                    val = m.group(1).strip()
                    try:
                        args.append(int(val))
                        break
                    except:
                        result, _ = self.macro_resolver.resolve(val)
                        if result is not None:
                            args.append(result)
                            break
            else:
                args.append(None)
        
        return args
    
    def _trace_param_from_callers(self, func_name: str, param_idx: int,
                                   chain: List[str], visited: Set[str]) -> Optional[int]:
        """Trace parameter value by finding callers."""
        if func_name in visited:
            return None
        visited = visited | {func_name}
        
        # Search all functions for calls to func_name
        for caller, calls in self.function_calls.items():
            for called, args in calls:
                if called == func_name and param_idx < len(args):
                    arg = args[param_idx]
                    chain.append(f"Called by {caller} with arg[{param_idx}] = {arg}")
                    
                    # Resolve arg
                    try:
                        return int(arg)
                    except:
                        pass
                    
                    result, _ = self.macro_resolver.resolve(arg)
                    if result is not None:
                        chain.append(f"Resolved: {result}")
                        return result
                    
                    # Arg is a variable in caller - trace it with full context
                    if caller in self.function_bodies:
                        # First, get caller's parameter values by tracing its callers
                        caller_context = {}
                        if caller in self.function_params:
                            caller_params = self.function_params[caller]
                            for pi, p in enumerate(caller_params):
                                val = self._trace_param_from_callers(caller, pi, [], visited.copy())
                                if val is not None:
                                    caller_context[p] = val
                        
                        # Now trace the variable in caller with context
                        var_result = self._trace_var_in_func_with_context(arg, caller, caller_context, chain, visited)
                        if var_result is not None:
                            return var_result
                    
                    # Arg might be a parameter of caller - recurse
                    if caller in self.function_params:
                        caller_params = self.function_params[caller]
                        if arg in caller_params:
                            caller_param_idx = caller_params.index(arg)
                            chain.append(f"{arg} is parameter of {caller}")
                            return self._trace_param_from_callers(caller, caller_param_idx, chain, visited)
        
        return None
    
    def _trace_var_in_func_with_context(self, var: str, func_name: str, context: Dict[str, int],
                                         chain: List[str], visited: Set[str]) -> Optional[int]:
        """Trace a variable within a function with known parameter context."""
        if func_name not in self.function_bodies:
            return None
        
        body = self.function_bodies[func_name]
        params = self.function_params.get(func_name, [])
        
        # First check conditionals with context
        result = self._analyze_ifelse_for_var(body, var, context, chain)
        if result is not None:
            return result
        
        result = self._analyze_switch_for_var(body, var, context, chain)
        if result is not None:
            return result
        
        # Check direct assignment
        m = re.search(rf'\b{re.escape(var)}\s*=\s*([^;(]+);', body)
        if m:
            val = m.group(1).strip()
            chain.append(f"In {func_name}: {var} = {val}")
            
            try:
                return int(val)
            except:
                pass
            
            result, _ = self.macro_resolver.resolve(val)
            if result is not None:
                chain.append(f"Resolved: {result}")
                return result
        
        return None
    
    def _analyze_function(self, func_name: str, args: List[Any], chain: List[str],
                          visited: Set[str], depth: int) -> Optional[int]:
        """Analyze a function with given arguments."""
        if depth > 20 or func_name not in self.function_bodies:
            return None
        
        body = self.function_bodies[func_name]
        params = self.function_params.get(func_name, [])
        
        # Build context from args
        context = {}
        for i, p in enumerate(params):
            if i < len(args) and args[i] is not None:
                context[p] = args[i]
        
        chain.append(f"Analyzing {func_name} with context: {context}")
        
        # Try switch-case
        result = self._analyze_switch(body, context, chain)
        if result is not None:
            return result
        
        # Try if-else
        result = self._analyze_ifelse(body, context, chain)
        if result is not None:
            return result
        
        # Try direct assignment/return
        result = self._analyze_direct(body, context, chain)
        if result is not None:
            return result
        
        # Try nested calls
        result = self._analyze_nested(body, context, chain, visited, depth)
        if result is not None:
            return result
        
        return None
    
    def _analyze_conditional_in_body(self, body: str, target_var: str, params: List[str],
                                      context: Dict[str, int], chain: List[str],
                                      visited: Set[str]) -> Optional[int]:
        """Analyze conditionals to find assignment to target_var."""
        # Try switch first
        result = self._analyze_switch_for_var(body, target_var, context, chain)
        if result is not None:
            return result
        
        # Try if-else
        result = self._analyze_ifelse_for_var(body, target_var, context, chain)
        if result is not None:
            return result
        
        return None
    
    def _analyze_switch(self, body: str, context: Dict[str, int], chain: List[str]) -> Optional[int]:
        """Analyze switch-case."""
        m = re.search(r'switch\s*\(\s*(\w+)\s*\)', body)
        if not m:
            return None
        
        switch_var = m.group(1)
        if switch_var not in context:
            return None
        
        switch_val = context[switch_var]
        chain.append(f"Switch on {switch_var} = {switch_val}")
        
        # Find the case
        # Pattern: case N: followed by code until break/case/default/}
        case_pattern = rf'case\s+{switch_val}\s*:(.*?)(?:break\s*;|case\s+\d|default\s*:|\}})'
        m = re.search(case_pattern, body, re.DOTALL)
        
        if m:
            case_body = m.group(1)
            chain.append(f"Matched case {switch_val}")
            
            # Find assignment
            for am in re.finditer(r'(\w+)\s*=\s*([A-Z_]\w*|\d+)\s*;', case_body):
                val = am.group(2)
                chain.append(f"Assigns {am.group(1)} = {val}")
                
                result, _ = self.macro_resolver.resolve(val)
                if result is not None:
                    chain.append(f"Resolved: {result}")
                    return result
        
        # Default case
        dm = re.search(r'default\s*:(.*?)(?:break\s*;|\})', body, re.DOTALL)
        if dm and not m:
            chain.append("Using default case")
            for am in re.finditer(r'(\w+)\s*=\s*([A-Z_]\w*|\d+)\s*;', dm.group(1)):
                result, _ = self.macro_resolver.resolve(am.group(2))
                if result is not None:
                    return result
        
        return None
    
    def _analyze_switch_for_var(self, body: str, target_var: str, context: Dict[str, int],
                                 chain: List[str]) -> Optional[int]:
        """Analyze switch to find assignment to target_var."""
        m = re.search(r'switch\s*\(\s*(\w+)\s*\)', body)
        if not m:
            return None
        
        switch_var = m.group(1)
        if switch_var not in context:
            return None
        
        switch_val = context[switch_var]
        chain.append(f"Switch({switch_var}={switch_val})")
        
        # Find case
        case_pattern = rf'case\s+{switch_val}\s*:(.*?)(?:break|case\s+\d|default|\}})'
        cm = re.search(case_pattern, body, re.DOTALL)
        
        if cm:
            case_body = cm.group(1)
            chain.append(f"Case {switch_val}")
            
            # Find assignment to target_var
            am = re.search(rf'\b{re.escape(target_var)}\s*=\s*([A-Z_]\w*|\d+)\s*;', case_body)
            if am:
                val = am.group(1)
                chain.append(f"{target_var} = {val}")
                result, _ = self.macro_resolver.resolve(val)
                if result is not None:
                    chain.append(f"= {result}")
                    return result
        
        return None
    
    def _analyze_ifelse(self, body: str, context: Dict[str, int], chain: List[str]) -> Optional[int]:
        """Analyze if-else."""
        pattern = r'if\s*\(\s*(\w+)\s*(==|!=|<|>|<=|>=)\s*(\d+)\s*\)\s*\{([^}]+)\}'
        
        for m in re.finditer(pattern, body, re.DOTALL):
            cond_var, op, cond_val, if_body = m.groups()
            cond_val = int(cond_val)
            
            if cond_var not in context:
                continue
            
            actual = context[cond_var]
            
            # Evaluate condition
            is_true = {
                '==': actual == cond_val,
                '!=': actual != cond_val,
                '<': actual < cond_val,
                '>': actual > cond_val,
                '<=': actual <= cond_val,
                '>=': actual >= cond_val,
            }.get(op, False)
            
            if is_true:
                chain.append(f"If {cond_var}{op}{cond_val} TRUE (actual={actual})")
                
                # Find assignment
                for am in re.finditer(r'(\w+)\s*=\s*([^;]+);', if_body):
                    val = am.group(2).strip()
                    if '"' in val:
                        continue
                    
                    chain.append(f"Branch: {am.group(1)} = {val}")
                    result, _ = self.macro_resolver.resolve(val)
                    if result is not None:
                        chain.append(f"= {result}")
                        return result
        
        return None
    
    def _analyze_ifelse_for_var(self, body: str, target_var: str, context: Dict[str, int],
                                 chain: List[str]) -> Optional[int]:
        """Analyze if-else for specific variable."""
        pattern = r'if\s*\(\s*(\w+)\s*(==|!=|<|>|<=|>=)\s*(\d+)\s*\)\s*\{([^}]+)\}'
        
        for m in re.finditer(pattern, body, re.DOTALL):
            cond_var, op, cond_val, if_body = m.groups()
            cond_val = int(cond_val)
            
            if cond_var not in context:
                continue
            
            actual = context[cond_var]
            is_true = {
                '==': actual == cond_val,
                '!=': actual != cond_val,
                '<': actual < cond_val,
                '>': actual > cond_val,
                '<=': actual <= cond_val,
                '>=': actual >= cond_val,
            }.get(op, False)
            
            if is_true:
                chain.append(f"If({cond_var}{op}{cond_val}) TRUE")
                
                # Find assignment to target_var
                am = re.search(rf'\b{re.escape(target_var)}\s*=\s*([^;]+);', if_body)
                if am:
                    val = am.group(1).strip()
                    chain.append(f"{target_var} = {val}")
                    result, _ = self.macro_resolver.resolve(val)
                    if result is not None:
                        chain.append(f"= {result}")
                        return result
        
        return None
    
    def _analyze_direct(self, body: str, context: Dict[str, int], chain: List[str]) -> Optional[int]:
        """Analyze direct return."""
        for m in re.finditer(r'return\s+([^;(]+);', body):
            val = m.group(1).strip()
            
            if val in context:
                chain.append(f"Return {val} = {context[val]}")
                return context[val]
            
            try:
                return int(val)
            except:
                pass
            
            result, _ = self.macro_resolver.resolve(val)
            if result is not None:
                chain.append(f"Return {val} = {result}")
                return result
        
        return None
    
    def _analyze_nested(self, body: str, context: Dict[str, int], chain: List[str],
                        visited: Set[str], depth: int) -> Optional[int]:
        """Analyze nested function calls."""
        for m in re.finditer(r'(?:return\s+)?(\w+)\s*\(([^)]*)\)\s*;', body):
            func, args_str = m.groups()
            
            if func in ['if', 'while', 'for', 'switch', 'printf', 'mpf_mfs_close']:
                continue
            
            if func in visited or func not in self.function_bodies:
                continue
            
            # Resolve args with context
            args = []
            for a in args_str.split(','):
                a = a.strip()
                if not a:
                    continue
                if a in context:
                    args.append(context[a])
                else:
                    try:
                        args.append(int(a))
                    except:
                        result, _ = self.macro_resolver.resolve(a)
                        args.append(result)
            
            chain.append(f"Nested: {func}({args})")
            
            result = self._analyze_function(func, args, chain, visited | {func}, depth + 1)
            if result is not None:
                return result
        
        return None


# Alias for compatibility
SmartDataFlowAnalyzer = UltimateDataFlowAnalyzer
RobustDataFlowAnalyzer = UltimateDataFlowAnalyzer
DataFlowAnalyzer = UltimateDataFlowAnalyzer


def test():
    """Test the analyzer on available test cases."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import TEST_CASES_DIR, INCLUDE_DIR
    
    print("\n" + "="*70)
    print("  Pattern-Based Data Flow Analyzer v3.0 - Results")
    print("="*70)
    
    cases = sorted([d for d in os.listdir(TEST_CASES_DIR) 
                    if os.path.isdir(os.path.join(TEST_CASES_DIR, d))])
    
    for case in cases:
        case_dir = os.path.join(TEST_CASES_DIR, case)
        analyzer = UltimateDataFlowAnalyzer(include_paths=[INCLUDE_DIR])
        results = analyzer.analyze_project(case_dir)
        
        print(f"\n{case}:")
        if results:
            for r in results:
                icon = "✅" if r.resolved_value is not None else "❌"
                print(f"  {icon} {r.raw_argument} → {r.resolved_value}")
        else:
            print("  ⚠️  No mpf_mfs_open calls found")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    test()

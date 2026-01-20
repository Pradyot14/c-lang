"""
Ultra-Robust AI FileNo Analyzer v11.0
=====================================
Handles large real-world codebases by:
1. Smart context selection (only relevant files)
2. Token limit management
3. Better error handling with detailed messages
4. Fallback strategies for large codebases
"""
import os
import re
import time
import traceback
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Load environment variables
def load_env():
    paths = [Path(__file__).resolve().parent.parent / ".env", Path.cwd() / ".env"]
    for env_path in paths:
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
                break
            except:
                continue

try:
    from dotenv import load_dotenv
    for p in [Path(__file__).resolve().parent.parent / ".env", Path.cwd() / ".env"]:
        if p.exists():
            load_dotenv(p)
            break
except ImportError:
    load_env()

try:
    from openai import AzureOpenAI, OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class AnalysisResult:
    file_path: str
    line_number: int
    containing_function: str
    raw_argument: str
    resolved_value: Optional[int]
    confidence: float
    resolution_chain: List[str]
    all_possible_values: List[int] = field(default_factory=list)
    analysis_method: str = "AI"
    error: Optional[str] = None
    reasoning_trace: Optional[str] = None


class AIDataFlowAnalyzer:
    """
    Robust AI analyzer for real-world large codebases.
    """
    
    # Approximate token limits (conservative)
    MAX_CONTEXT_CHARS = 100000  # ~25k tokens
    MAX_FILE_CHARS = 20000     # Per file limit
    
    def __init__(self, include_paths: List[str] = None, verbose: bool = False, debug: bool = False):
        self.include_paths = include_paths or []
        self.verbose = verbose
        self.debug = debug  # Extra debug output
        self.max_retries = 3
        
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        # Azure config
        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
        
        if "your-resource" in azure_endpoint:
            azure_endpoint = ""
        if "your-api-key" in azure_api_key:
            azure_api_key = ""
        
        # OpenAI config
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o")
        
        # Initialize client
        if all([azure_endpoint, azure_api_key, azure_deployment]):
            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version="2024-02-15-preview"
            )
            self.model = azure_deployment
            self.provider = "Azure"
            if verbose:
                print("  🔷 Using Azure OpenAI")
        elif openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
            self.model = openai_model
            self.provider = "OpenAI"
            if verbose:
                print(f"  🟢 Using OpenAI ({self.model})")
        else:
            raise ValueError("No AI credentials configured. Set OPENAI_API_KEY in .env file")
        
        # Storage
        self.all_code: Dict[str, str] = {}
        self.all_macros: Dict[str, Tuple[str, str]] = {}
        self.functions: Dict[str, Dict] = {}  # function_name -> {file, body, calls}
        self.call_graph: Dict[str, Set[str]] = {}
    
    def _call_llm(self, prompt: str, system: str) -> str:
        """Call LLM with detailed error handling."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                if self.debug:
                    print(f"  [DEBUG] LLM call attempt {attempt + 1}, prompt size: {len(prompt)} chars")
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=4000,
                    timeout=120  # 2 minute timeout
                )
                
                result = response.choices[0].message.content.strip()
                
                if self.debug:
                    print(f"  [DEBUG] LLM response size: {len(result)} chars")
                
                return result
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                if self.debug:
                    print(f"  [DEBUG] LLM error (attempt {attempt + 1}): {e}")
                
                # Check for specific errors
                if "rate_limit" in error_str or "429" in error_str:
                    wait_time = 10 * (attempt + 1)
                    if self.verbose:
                        print(f"  ⏳ Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif "context_length" in error_str or "maximum context" in error_str or "too many tokens" in error_str:
                    raise Exception(f"Context too large for API. Reduce codebase size or use focused analysis.")
                elif "timeout" in error_str:
                    wait_time = 5 * (attempt + 1)
                    if self.verbose:
                        print(f"  ⏳ Timeout, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    time.sleep(2 * (attempt + 1))
        
        raise Exception(f"LLM call failed after {self.max_retries} attempts: {last_error}")
    
    def analyze_project(self, project_dir: str) -> List[AnalysisResult]:
        """Main entry point."""
        if self.verbose:
            print(f"  Loading: {project_dir}")
        
        self._load_all_code(project_dir)
        
        if self.verbose:
            print(f"  Files loaded: {len(self.all_code)}")
            print(f"  Macros found: {len(self.all_macros)}")
            total_chars = sum(len(c) for c in self.all_code.values())
            print(f"  Total code size: {total_chars:,} chars")
        
        calls = self._find_all_calls()
        
        if self.verbose:
            print(f"  Found {len(calls)} mpf_mfs_open calls")
        
        results = []
        for i, call in enumerate(calls):
            if self.verbose and len(calls) > 1:
                print(f"  Analyzing call {i+1}/{len(calls)}...")
            result = self._analyze_call(call)
            results.append(result)
            time.sleep(0.5)  # Rate limiting
        
        return results
    
    def _load_all_code(self, project_dir: str):
        """Load all code files and extract structure."""
        self.all_code = {}
        self.all_macros = {}
        self.functions = {}
        self.call_graph = {}
        
        dirs = [project_dir] + [d for d in self.include_paths if os.path.isdir(d)]
        
        for scan_dir in dirs:
            for root, _, files in os.walk(scan_dir):
                for f in files:
                    if f.endswith(('.c', '.h', '.inc')):
                        filepath = os.path.join(root, f)
                        try:
                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                                content = file.read()
                                self.all_code[filepath] = content
                                self._extract_macros(filepath, content)
                                self._extract_functions(filepath, content)
                        except Exception as e:
                            if self.debug:
                                print(f"  [DEBUG] Failed to load {filepath}: {e}")
    
    def _extract_macros(self, filepath: str, content: str):
        """Extract macros with better patterns - handles hex, comments, expressions."""
        content_clean = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content_clean = re.sub(r'//.*$', '', content_clean, flags=re.MULTILINE)
        content_clean = re.sub(r'\\\n', ' ', content_clean)
        
        # Find #define statements
        for line in content_clean.split('\n'):
            line = line.strip()
            if not line.startswith('#'):
                continue
            
            # Match #define NAME VALUE
            match = re.match(r'#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.+)$', line)
            if not match:
                continue
            
            name = match.group(1)
            value = match.group(2).strip()
            
            # Skip if already have it
            if name in self.all_macros:
                continue
            
            # Skip function-like macros #define NAME(x, y)
            if re.match(r'^\s*\(.*,.*\)', value):
                continue
            
            # Skip include guards and empty values
            if not value or value.startswith('#'):
                continue
            
            # Skip if value starts with typedef, struct, etc
            if re.match(r'^(typedef|struct|enum|union|extern|static)\b', value):
                continue
            
            self.all_macros[name] = (value, os.path.basename(filepath))
    
    def _extract_functions(self, filepath: str, content: str):
        """Extract function definitions and build call graph."""
        func_pattern = r'(?:static\s+)?(?:inline\s+)?(\w+(?:\s*\*)?)\s+(\w+)\s*\(([^)]*)\)\s*\{'
        
        for match in re.finditer(func_pattern, content):
            func_name = match.group(2)
            
            # Extract function body
            start = match.end() - 1
            brace = 1
            end = start + 1
            while end < len(content) and brace > 0:
                if content[end] == '{':
                    brace += 1
                elif content[end] == '}':
                    brace -= 1
                end += 1
            
            body = content[start:end]
            
            self.functions[func_name] = {
                'file': filepath,
                'body': body,
                'signature': match.group(0).rstrip('{').strip()
            }
            
            # Build call graph
            self.call_graph[func_name] = set()
            for call_match in re.finditer(r'\b(\w+)\s*\(', body):
                called = call_match.group(1)
                if called not in ['if', 'while', 'for', 'switch', 'return', 'sizeof', 'printf']:
                    self.call_graph[func_name].add(called)
    
    def _find_all_calls(self) -> List[Dict]:
        """Find all mpf_mfs_open calls."""
        calls = []
        
        for filepath, content in self.all_code.items():
            if not filepath.endswith('.c'):
                continue
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'mpf_mfs_open' in line:
                    full_call = line
                    j = i
                    while full_call.count('(') > full_call.count(')') and j < len(lines) - 1:
                        j += 1
                        full_call += ' ' + lines[j].strip()
                    
                    match = re.search(r'mpf_mfs_open\s*\(([^)]+)\)', full_call)
                    if match:
                        args = self._parse_args(match.group(1))
                        if len(args) >= 3:
                            containing_func = self._find_containing_function(filepath, i)
                            calls.append({
                                'file': filepath,
                                'line': i + 1,
                                'func': containing_func,
                                'full_call': match.group(0),
                                'args': args,
                                'raw_arg': args[2],
                                'context_before': '\n'.join(lines[max(0, i-30):i]),
                                'context_after': '\n'.join(lines[i:min(len(lines), i+10)]),
                            })
        
        return calls
    
    def _parse_args(self, args_str: str) -> List[str]:
        """Parse function arguments."""
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
    
    def _find_containing_function(self, filepath: str, line_idx: int) -> str:
        """Find function containing a line."""
        content = self.all_code.get(filepath, '')
        lines = content.split('\n')
        func_pattern = r'(?:static\s+)?(?:inline\s+)?(?:\w+(?:\s*\*)?)\s+(\w+)\s*\([^)]*\)\s*\{'
        
        current_func = "unknown"
        brace_depth = 0
        in_function = False
        
        for i, line in enumerate(lines[:line_idx + 1]):
            match = re.search(func_pattern, line)
            if match:
                current_func = match.group(1)
                in_function = True
                brace_depth = 0
            
            if in_function:
                brace_depth += line.count('{') - line.count('}')
                if brace_depth <= 0:
                    in_function = False
        
        return current_func
    
    def _analyze_call(self, call: Dict) -> AnalysisResult:
        """Analyze a single call with smart context selection."""
        raw_arg = call['raw_arg']
        
        # Strategy 1: Direct numeric
        if raw_arg.lstrip('-').isdigit():
            return AnalysisResult(
                file_path=call['file'],
                line_number=call['line'],
                containing_function=call['func'],
                raw_argument=raw_arg,
                resolved_value=int(raw_arg),
                confidence=1.0,
                resolution_chain=[f"Direct numeric: {raw_arg}"],
                analysis_method="Direct"
            )
        
        # Strategy 2: Simple macro lookup
        if raw_arg in self.all_macros:
            resolved = self._try_resolve_macro(raw_arg)
            if resolved is not None:
                return AnalysisResult(
                    file_path=call['file'],
                    line_number=call['line'],
                    containing_function=call['func'],
                    raw_argument=raw_arg,
                    resolved_value=resolved,
                    confidence=0.98,
                    resolution_chain=[f"Macro: {raw_arg} = {self.all_macros[raw_arg][0]} = {resolved}"],
                    analysis_method="Macro"
                )
        
        # Strategy 2.5: CRITICAL - Check for undefined macros before calling AI
        # This prevents AI hallucination from comments
        missing_macros = self._find_missing_macros(call)
        if missing_macros:
            return AnalysisResult(
                file_path=call['file'],
                line_number=call['line'],
                containing_function=call['func'],
                raw_argument=raw_arg,
                resolved_value=None,
                confidence=0.0,
                resolution_chain=[f"UNDEFINED: Missing macros: {', '.join(missing_macros)}"],
                analysis_method="Validation",
                error=f"Undefined macros: {', '.join(missing_macros)}"
            )
        
        # Strategy 3: AI analysis with SMART context selection
        return self._ai_analyze_smart(call)
    
    def _find_missing_macros(self, call: Dict) -> List[str]:
        """Find macros that are referenced but not defined."""
        missing = []
        
        # Get function body
        func_body = ""
        if call['func'] in self.functions:
            func_body = self.functions[call['func']]['body']
        
        # Also check called functions
        for called in self.call_graph.get(call['func'], []):
            if called in self.functions:
                func_body += "\n" + self.functions[called]['body']
        
        # Add context
        func_body += "\n" + call['context_before'] + "\n" + call['context_after']
        
        # Find macro-like identifiers (UPPER_CASE names that look like macros)
        # Pattern: starts with letter, all uppercase, may contain numbers and underscores
        macro_pattern = r'\b([A-Z][A-Z0-9_]{2,})\b'
        potential_macros = set(re.findall(macro_pattern, func_body))
        
        # Common keywords and types to exclude
        exclude = {
            'NULL', 'TRUE', 'FALSE', 'EOF', 'STDIN', 'STDOUT', 'STDERR',
            'INT', 'CHAR', 'VOID', 'LONG', 'SHORT', 'FLOAT', 'DOUBLE',
            'SIZE_T', 'UINT', 'ULONG', 'USHORT', 'UCHAR',
            'MPF_MFS_FCB', 'MPF_MFS_READLOCK', 'MPF_MFS_WRITELOCK',  # Known system macros
        }
        
        # Check assignment to the raw_arg variable
        raw_arg = call['raw_arg']
        assign_pattern = rf'\b{re.escape(raw_arg)}\s*=\s*([^;]+);'
        assign_match = re.search(assign_pattern, func_body)
        
        if assign_match:
            assigned_value = assign_match.group(1).strip()
            # Find macros in the assigned expression
            macros_in_expr = re.findall(macro_pattern, assigned_value)
            
            for macro in macros_in_expr:
                if macro not in self.all_macros and macro not in exclude:
                    # This macro is used but not defined - critical!
                    missing.append(macro)
        
        # Also check if raw_arg itself is an undefined macro
        if re.match(r'^[A-Z][A-Z0-9_]{2,}$', raw_arg):
            if raw_arg not in self.all_macros and raw_arg not in exclude:
                missing.append(raw_arg)
        
        return missing
    
    def _try_resolve_macro(self, name: str, depth: int = 0) -> Optional[int]:
        """Try to resolve macro to numeric value."""
        if depth > 20 or name not in self.all_macros:
            return None
        
        value, _ = self.all_macros[name]
        value = value.strip()
        
        # Remove parentheses
        if value.startswith('(') and value.endswith(')'):
            value = value[1:-1].strip()
        
        # Direct number
        try:
            return int(value)
        except ValueError:
            pass
        
        # Hex number
        if value.startswith('0x') or value.startswith('0X'):
            try:
                return int(value, 16)
            except ValueError:
                pass
        
        # Another macro
        if value in self.all_macros:
            return self._try_resolve_macro(value, depth + 1)
        
        # Expression
        expr = value
        for m in self.all_macros:
            if re.search(rf'\b{m}\b', expr):
                resolved = self._try_resolve_macro(m, depth + 1)
                if resolved is not None:
                    expr = re.sub(rf'\b{m}\b', str(resolved), expr)
        
        try:
            expr_clean = expr.replace(' ', '')
            if expr_clean and re.match(r'^[\d+\-*/()x0-9a-fA-F]+$', expr_clean):
                result = eval(expr_clean)
                if isinstance(result, (int, float)):
                    return int(result)
        except:
            pass
        
        return None
    
    def _get_relevant_functions(self, call: Dict) -> Dict[str, str]:
        """Get only functions relevant to the call (following call chain)."""
        relevant = {}
        containing_func = call['func']
        
        # Start with the containing function
        if containing_func in self.functions:
            relevant[containing_func] = self.functions[containing_func]['body']
        
        # BFS to find called functions
        visited = {containing_func}
        queue = [containing_func]
        depth = 0
        max_depth = 5  # Limit depth
        
        while queue and depth < max_depth:
            next_queue = []
            for func in queue:
                for called in self.call_graph.get(func, []):
                    if called not in visited and called in self.functions:
                        visited.add(called)
                        next_queue.append(called)
                        relevant[called] = self.functions[called]['body']
            queue = next_queue
            depth += 1
        
        return relevant
    
    def _get_relevant_macros(self, call: Dict, relevant_funcs: Dict[str, str]) -> Dict[str, str]:
        """Get only macros that appear in relevant code."""
        relevant = {}
        
        # Combine all relevant code
        all_relevant_code = call['context_before'] + call['context_after']
        for body in relevant_funcs.values():
            all_relevant_code += body
        
        # Find macros that appear in the code
        for macro_name, (macro_value, source) in self.all_macros.items():
            if re.search(rf'\b{macro_name}\b', all_relevant_code):
                relevant[macro_name] = f"{macro_value}  // from {source}"
                
                # Also include macros referenced in this macro's value
                for other_macro in self.all_macros:
                    if re.search(rf'\b{other_macro}\b', macro_value):
                        other_value, other_source = self.all_macros[other_macro]
                        relevant[other_macro] = f"{other_value}  // from {other_source}"
        
        return relevant
    
    def _ai_analyze_smart(self, call: Dict) -> AnalysisResult:
        """AI analysis with smart context selection for large codebases."""
        
        try:
            # Get only relevant functions (following call chain)
            relevant_funcs = self._get_relevant_functions(call)
            
            if self.debug:
                print(f"  [DEBUG] Relevant functions: {list(relevant_funcs.keys())}")
            
            # Get only relevant macros
            relevant_macros = self._get_relevant_macros(call, relevant_funcs)
            
            if self.debug:
                print(f"  [DEBUG] Relevant macros: {list(relevant_macros.keys())}")
            
            # Build focused prompt
            macros_text = "\n".join(f"#define {k} {v}" for k, v in sorted(relevant_macros.items()))
            if not macros_text:
                macros_text = "NO RELEVANT MACROS FOUND"
            
            funcs_text = ""
            for fname, body in relevant_funcs.items():
                # Truncate very long functions
                if len(body) > self.MAX_FILE_CHARS:
                    body = body[:self.MAX_FILE_CHARS] + "\n// ... (truncated)"
                funcs_text += f"\n// === Function: {fname} ===\n{body}\n"
            
            # Check total size
            total_size = len(macros_text) + len(funcs_text) + len(call['context_before']) + len(call['context_after'])
            if total_size > self.MAX_CONTEXT_CHARS:
                if self.verbose:
                    print(f"  ⚠️ Context size {total_size:,} chars, truncating...")
                # Truncate functions to fit
                max_func_size = (self.MAX_CONTEXT_CHARS - len(macros_text) - 5000) // max(1, len(relevant_funcs))
                funcs_text = ""
                for fname, body in relevant_funcs.items():
                    if len(body) > max_func_size:
                        body = body[:max_func_size] + "\n// ... (truncated)"
                    funcs_text += f"\n// === Function: {fname} ===\n{body}\n"
            
            system_prompt = """You are an expert C code static analyzer. Determine the EXACT numeric value of the 3rd argument to mpf_mfs_open().

## CRITICAL RULES - READ CAREFULLY:
1. You can ONLY use macro values that are EXPLICITLY listed in "Macro Definitions" section below
2. If a macro like APL_FILENO_XXX is used but NOT in the macro list, you MUST answer UNDEFINED
3. NEVER guess values from:
   - Comments like "Expected: 2003"
   - Variable names or file names
   - Assumed conventions
4. Comments are TEST ANNOTATIONS, not source of truth
5. If ANY required macro is missing from the list, answer UNDEFINED

## METHODOLOGY:
1. Check if the required macro EXISTS in the provided "Macro Definitions"
2. If macro is MISSING → immediately answer UNDEFINED
3. If macro EXISTS → trace the value step by step
4. Evaluate arithmetic expressions using ONLY defined values

## OUTPUT FORMAT:
```trace
STEP 1: [description]
STEP 2: [description]
...
FINAL: [expression] = [numeric value OR UNDEFINED if macro missing]
```

RESULT: <integer>
OR
RESULT: UNDEFINED

## EXAMPLE OF CORRECT BEHAVIOR:
If code uses `APL_FILENO_DATA3` but Macro Definitions shows:
```
#define MPF_MFS_READLOCK 1
#define MPF_MFS_WRITELOCK 2
```
(No APL_FILENO_DATA3 defined)

Then you MUST answer:
```trace
STEP 1: Variable fileno is assigned APL_FILENO_DATA3
STEP 2: Looking for APL_FILENO_DATA3 in macro definitions
STEP 3: APL_FILENO_DATA3 is NOT in the provided macro list
FINAL: Cannot resolve - macro APL_FILENO_DATA3 is not defined
```
RESULT: UNDEFINED"""

            user_prompt = f"""# ANALYSIS REQUEST

## Macro Definitions:
```c
{macros_text}
```

## Relevant Functions:
{funcs_text}

## Target Call:
- File: {os.path.basename(call['file'])}
- Line: {call['line']}
- Containing function: {call['func']}
- Call: {call['full_call']}
- 3rd argument to resolve: `{call['raw_arg']}`

## Context around the call:
```c
{call['context_before'][-2000:]}
>>> {call['context_after'].split(chr(10))[0]}  // <-- TARGET CALL
{chr(10).join(call['context_after'].split(chr(10))[1:])[:500]}
```

Trace the value of `{call['raw_arg']}` step by step."""

            if self.debug:
                print(f"  [DEBUG] Total prompt size: {len(system_prompt) + len(user_prompt):,} chars")
            
            response = self._call_llm(user_prompt, system_prompt)
            
            # Parse response
            result = self._parse_ai_response(response, call)
            result.reasoning_trace = response
            return result
            
        except Exception as e:
            error_msg = str(e)
            if self.debug:
                print(f"  [DEBUG] Full error:\n{traceback.format_exc()}")
            
            return AnalysisResult(
                file_path=call['file'],
                line_number=call['line'],
                containing_function=call['func'],
                raw_argument=call['raw_arg'],
                resolved_value=None,
                confidence=0.0,
                resolution_chain=[f"Error: {error_msg[:200]}"],
                analysis_method="AI-Error",
                error=error_msg
            )
    
    def _parse_ai_response(self, response: str, call: Dict) -> AnalysisResult:
        """Parse AI response."""
        
        # Check for UNDEFINED
        if re.search(r'RESULT:\s*UNDEFINED', response, re.IGNORECASE):
            trace_match = re.search(r'```trace\s*(.*?)```', response, re.DOTALL)
            reason = "Missing definition"
            if trace_match:
                for line in trace_match.group(1).split('\n'):
                    if 'NOT' in line.upper() or 'MISSING' in line.upper() or 'UNDEFINED' in line.upper():
                        reason = line.strip()[:100]
                        break
            
            return AnalysisResult(
                file_path=call['file'],
                line_number=call['line'],
                containing_function=call['func'],
                raw_argument=call['raw_arg'],
                resolved_value=None,
                confidence=0.0,
                resolution_chain=[f"UNDEFINED: {reason}"],
                analysis_method="AI-Trace",
                error=reason
            )
        
        # Look for RESULT: <number>
        result_match = re.search(r'RESULT:\s*(\d+)', response, re.IGNORECASE)
        if result_match:
            value = int(result_match.group(1))
            
            trace_steps = []
            trace_match = re.search(r'```trace\s*(.*?)```', response, re.DOTALL)
            if trace_match:
                for line in trace_match.group(1).split('\n'):
                    line = line.strip()
                    if line.startswith('STEP') or line.startswith('FINAL'):
                        trace_steps.append(line[:100])
            
            if not trace_steps:
                trace_steps = [f"AI: {call['raw_arg']} = {value}"]
            
            return AnalysisResult(
                file_path=call['file'],
                line_number=call['line'],
                containing_function=call['func'],
                raw_argument=call['raw_arg'],
                resolved_value=value,
                confidence=0.95,
                resolution_chain=trace_steps[:5],
                analysis_method="AI-Trace"
            )
        
        # Fallback: find 4-digit numbers
        numbers = re.findall(r'\b(\d{4,})\b', response)
        if numbers:
            value = int(numbers[-1])
            return AnalysisResult(
                file_path=call['file'],
                line_number=call['line'],
                containing_function=call['func'],
                raw_argument=call['raw_arg'],
                resolved_value=value,
                confidence=0.6,
                resolution_chain=["Extracted from AI response"],
                analysis_method="AI-Fallback"
            )
        
        # Could not determine
        return AnalysisResult(
            file_path=call['file'],
            line_number=call['line'],
            containing_function=call['func'],
            raw_argument=call['raw_arg'],
            resolved_value=None,
            confidence=0.0,
            resolution_chain=["Could not parse AI response"],
            analysis_method="AI-Failed",
            error="No result in AI response"
        )

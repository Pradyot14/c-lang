"""
Enhanced Data Flow Analyzer - Handles complex cases including conditionals
"""
import os
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path

from macro_extractor import MacroExtractor
from ast_parser import ASTParser, FunctionCall, VariableAssignment


@dataclass
class DataFlowResult:
    """Result of data flow analysis for a single mpf_mfs_open call."""
    file_path: str
    line_number: int
    raw_argument: str
    resolved_value: Optional[int]
    resolution_chain: List[str]
    confidence: float
    containing_function: str = ""
    all_possible_values: List[int] = field(default_factory=list)
    
    def __repr__(self):
        return f"DataFlowResult(raw={self.raw_argument}, resolved={self.resolved_value}, confidence={self.confidence})"


class EnhancedDataFlowAnalyzer:
    """
    Enhanced analyzer that handles complex data flow including:
    - Conditionals (if-else)
    - Arithmetic expressions
    - Cross-function calls
    - Nested macros
    """
    
    def __init__(self, include_paths: List[str] = None):
        self.include_paths = include_paths or []
        self.macro_extractor = MacroExtractor()
        self.ast_parser = ASTParser(include_paths)
        self.analyzed_files: Set[str] = set()
        self.file_contents: Dict[str, str] = {}
        
    def analyze_project(self, project_dir: str) -> List[DataFlowResult]:
        """Analyze all C files in a project directory."""
        results = []
        
        # Step 1: Load all file contents and extract macros
        self._load_project_files(project_dir)
        self._extract_all_macros(project_dir)
        
        # Step 2: Parse all C files
        self._parse_all_c_files(project_dir)
        
        # Step 3: Find all mpf_mfs_open calls
        mfs_calls = self.ast_parser.find_function_calls("mpf_mfs_open")
        
        if not mfs_calls:
            return []
        
        # Step 4: Resolve each call
        for call in mfs_calls:
            result = self._resolve_call(call, project_dir)
            results.append(result)
        
        return results
    
    def _load_project_files(self, project_dir: str):
        """Load all source file contents into memory."""
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith(('.c', '.h')):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            self.file_contents[filepath] = f.read()
                    except Exception:
                        pass
        
        # Also load from include paths
        for inc_path in self.include_paths:
            if os.path.exists(inc_path):
                for root, _, files in os.walk(inc_path):
                    for file in files:
                        if file.endswith(('.c', '.h')):
                            filepath = os.path.join(root, file)
                            try:
                                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                    self.file_contents[filepath] = f.read()
                            except Exception:
                                pass
        
    def _extract_all_macros(self, project_dir: str):
        """Extract macros from all header and source files."""
        self.macro_extractor.extract_from_directory(project_dir)
        for inc_path in self.include_paths:
            if os.path.exists(inc_path):
                self.macro_extractor.extract_from_directory(inc_path)
    
    def _parse_all_c_files(self, project_dir: str):
        """Parse all .c files in the project."""
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.endswith('.c'):
                    filepath = os.path.join(root, file)
                    self.ast_parser.parse_file(filepath)
                    self.analyzed_files.add(filepath)
    
    def _resolve_call(self, call: FunctionCall, project_dir: str) -> DataFlowResult:
        """Resolve the 3rd argument of an mpf_mfs_open call."""
        resolution_chain = []
        
        # Get the 3rd argument (index 2)
        if len(call.arguments) < 3:
            return DataFlowResult(
                file_path=call.file_path,
                line_number=call.line_number,
                raw_argument="UNKNOWN",
                resolved_value=None,
                resolution_chain=["Error: Less than 3 arguments"],
                confidence=0.0,
                containing_function=call.containing_function,
            )
        
        raw_arg = call.arguments[2].strip()
        resolution_chain.append(f"Found: {raw_arg} at {call.file_path}:{call.line_number}")
        
        # Try to resolve
        resolved_value, confidence, all_values = self._trace_value_enhanced(
            raw_arg, call.file_path, call.containing_function, call.line_number,
            project_dir, resolution_chain
        )
        
        return DataFlowResult(
            file_path=call.file_path,
            line_number=call.line_number,
            raw_argument=raw_arg,
            resolved_value=resolved_value,
            resolution_chain=resolution_chain,
            confidence=confidence,
            containing_function=call.containing_function,
            all_possible_values=all_values,
        )
    
    def _trace_value_enhanced(self, value: str, file_path: str, function: str,
                               call_line: int, project_dir: str, chain: List[str],
                               depth: int = 0) -> Tuple[Optional[int], float, List[int]]:
        """
        Enhanced value tracing that handles complex cases.
        Returns (resolved_value, confidence, all_possible_values)
        """
        if depth > 10:
            chain.append("Max depth reached")
            return None, 0.0, []
        
        value = value.strip()
        
        # Case 1: Direct numeric literal
        if self._is_numeric(value):
            numeric_val = int(value.strip('()'))
            chain.append(f"Direct numeric: {numeric_val}")
            return numeric_val, 1.0, [numeric_val]
        
        # Case 2: Macro
        if value in self.macro_extractor.macros:
            resolved, is_numeric = self.macro_extractor.resolve_macro(value)
            chain.append(f"Macro {value} -> {resolved}")
            if is_numeric:
                return int(resolved), 0.95, [int(resolved)]
            else:
                return self._trace_value_enhanced(
                    resolved, file_path, function, call_line, project_dir, chain, depth + 1
                )
        
        # Case 3: Arithmetic expression with macros
        if self._contains_operator(value):
            result, success = self.macro_extractor.evaluate_expression(value)
            if success:
                chain.append(f"Expression {value} -> {result}")
                return result, 0.9, [result]
        
        # Case 4: Variable - enhanced tracing
        # Look for assignment in the same function, before the call line
        assigned_value = self._find_variable_value_in_function(
            value, file_path, function, call_line
        )
        
        if assigned_value:
            chain.append(f"Variable {value} assigned via function logic")
            return self._trace_value_enhanced(
                assigned_value, file_path, function, call_line, project_dir, chain, depth + 1
            )
        
        # Case 5: Check if variable comes from a function call
        func_result = self._check_function_return_value(
            value, file_path, function, call_line, project_dir, chain, depth
        )
        if func_result[0] is not None:
            return func_result
        
        # Case 6: Direct search in file for assignment pattern
        direct_value = self._search_variable_value_directly(value, file_path)
        if direct_value:
            chain.append(f"Found {value} = {direct_value}")
            return self._trace_value_enhanced(
                direct_value, file_path, function, call_line, project_dir, chain, depth + 1
            )
        
        chain.append(f"Could not resolve: {value}")
        return None, 0.0, []
    
    def _find_variable_value_in_function(self, var_name: str, file_path: str,
                                          function: str, before_line: int) -> Optional[str]:
        """
        Find the value assigned to a variable within a function.
        Handles conditional assignments and function calls.
        """
        content = self.file_contents.get(file_path, "")
        if not content:
            return None
        
        lines = content.split('\n')
        
        # Find the function that contains this variable assignment
        # Check if var_name is assigned from a function call
        for i, line in enumerate(lines):
            if i >= before_line:
                break
            
            # Pattern: var = function_name(args);
            func_call_pattern = rf'{var_name}\s*=\s*(\w+)\s*\([^)]*\)\s*;'
            match = re.search(func_call_pattern, line)
            if match:
                called_func = match.group(1)
                # Analyze the called function
                return self._analyze_function_for_return_value(called_func, file_path)
        
        # Look for direct assignment
        for i, line in enumerate(lines):
            if i >= before_line:
                break
            
            # Pattern: var = value;
            assign_pattern = rf'{var_name}\s*=\s*([^;]+);'
            match = re.search(assign_pattern, line)
            if match:
                value = match.group(1).strip()
                # Skip if it's part of printf or similar
                if '%' in value or '"' in value:
                    continue
                return value
        
        return None
    
    def _analyze_function_for_return_value(self, func_name: str, current_file: str) -> Optional[str]:
        """
        Analyze a function to determine what value it returns.
        Handles conditionals by analyzing all branches.
        """
        # Search all files for the function definition
        for file_path, content in self.file_contents.items():
            # Find function definition
            func_pattern = rf'(?:int|void|char|float|double)\s+{func_name}\s*\([^)]*\)\s*\{{'
            if not re.search(func_pattern, content):
                continue
            
            # Extract function body
            func_body = self._extract_function_body(content, func_name)
            if not func_body:
                continue
            
            # Analyze the function body for return values
            return_values = self._analyze_function_returns(func_body)
            
            if return_values:
                # If there are multiple possible return values, try to determine
                # which one is most likely based on the call context
                # For now, return the first one
                return return_values[0]
        
        return None
    
    def _extract_function_body(self, content: str, func_name: str) -> Optional[str]:
        """Extract the body of a function."""
        # Find function start
        func_pattern = rf'(?:int|void|char|float|double|static)\s+{func_name}\s*\([^)]*\)\s*\{{'
        match = re.search(func_pattern, content)
        if not match:
            return None
        
        start = match.end() - 1  # Include the opening brace
        brace_count = 1
        end = start + 1
        
        while end < len(content) and brace_count > 0:
            if content[end] == '{':
                brace_count += 1
            elif content[end] == '}':
                brace_count -= 1
            end += 1
        
        return content[start:end]
    
    def _analyze_function_returns(self, func_body: str) -> List[str]:
        """
        Analyze function body to find all possible return values.
        Handles assignments and returns.
        """
        return_values = []
        
        # Find all assignments to local variables that are returned
        # Pattern: variable = MACRO - N or variable = MACRO
        assign_pattern = r'(\w+)\s*=\s*([A-Z_][A-Z0-9_]*(?:\s*[-+*/]\s*\d+)?)\s*;'
        assignments = {}
        
        for match in re.finditer(assign_pattern, func_body):
            var_name = match.group(1)
            value = match.group(2)
            assignments[var_name] = value
        
        # Find return statements
        return_pattern = r'return\s*\(?([^;)]+)\)?;'
        for match in re.finditer(return_pattern, func_body):
            ret_val = match.group(1).strip()
            
            # Check if it's a variable that was assigned
            if ret_val in assignments:
                return_values.append(assignments[ret_val])
            else:
                return_values.append(ret_val)
        
        # Also look for conditional assignments
        # Pattern: if (condition) { var = value; }
        cond_pattern = r'if\s*\([^)]*\)\s*\{\s*(\w+)\s*=\s*([^;]+);'
        for match in re.finditer(cond_pattern, func_body):
            var_name = match.group(1)
            value = match.group(2).strip()
            if var_name in assignments or any(var_name in rv for rv in return_values):
                return_values.append(value)
        
        return return_values
    
    def _check_function_return_value(self, var_name: str, file_path: str,
                                      function: str, call_line: int,
                                      project_dir: str, chain: List[str],
                                      depth: int) -> Tuple[Optional[int], float, List[int]]:
        """Check if a variable is assigned from a function return value."""
        content = self.file_contents.get(file_path, "")
        if not content:
            return None, 0.0, []
        
        # Look for: var = some_function(args);
        pattern = rf'{var_name}\s*=\s*(\w+)\s*\([^)]*\)\s*;'
        
        for line in content.split('\n'):
            match = re.search(pattern, line)
            if match:
                called_func = match.group(1)
                chain.append(f"Variable {var_name} from function {called_func}()")
                
                # Analyze the called function
                return_value = self._analyze_function_for_return_value(called_func, file_path)
                if return_value:
                    return self._trace_value_enhanced(
                        return_value, file_path, called_func, 0,
                        project_dir, chain, depth + 1
                    )
        
        return None, 0.0, []
    
    def _search_variable_value_directly(self, var_name: str, file_path: str) -> Optional[str]:
        """Direct search for variable value in file."""
        content = self.file_contents.get(file_path, "")
        if not content:
            return None
        
        # Pattern: var = MACRO; or var = number;
        pattern = rf'{var_name}\s*=\s*([A-Z_][A-Z0-9_]*|\d+)\s*;'
        match = re.search(pattern, content)
        if match:
            return match.group(1)
        
        return None
    
    def _is_numeric(self, value: str) -> bool:
        """Check if value is a numeric literal."""
        try:
            int(value.strip('()'))
            return True
        except ValueError:
            return False
    
    def _contains_operator(self, value: str) -> bool:
        """Check if value contains arithmetic operators."""
        return any(op in value for op in ['+', '-', '*', '/'])


# Try to import the best analyzer available
try:
    from robust_analyzer import UltimateDataFlowAnalyzer
    DataFlowAnalyzer = UltimateDataFlowAnalyzer
except ImportError:
    # Fallback to basic version
    DataFlowAnalyzer = EnhancedDataFlowAnalyzer


def test_analyzer():
    """Test the data flow analyzer on available test cases."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from config import TEST_CASES_DIR, INCLUDE_DIR
    
    # Find available test cases
    cases = sorted([d for d in os.listdir(TEST_CASES_DIR) 
                    if os.path.isdir(os.path.join(TEST_CASES_DIR, d))])
    
    if not cases:
        print("No test cases found")
        return
    
    print("\n" + "="*50)
    print("  Data Flow Analyzer - Test Results")
    print("="*50)
    
    for case in cases:
        case_dir = os.path.join(TEST_CASES_DIR, case)
        analyzer = EnhancedDataFlowAnalyzer(include_paths=[INCLUDE_DIR])
        results = analyzer.analyze_project(case_dir)
        
        print(f"\n{case}:")
        if results:
            for r in results:
                icon = "✅" if r.resolved_value is not None else "❌"
                print(f"  {icon} {r.raw_argument} → {r.resolved_value}")
        else:
            print("  ⚠️  No mpf_mfs_open calls found")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    test_analyzer()

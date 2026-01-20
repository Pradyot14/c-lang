"""
AST Parser using libclang - Parse C files and extract function calls
"""
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path

# Try to import clang bindings
try:
    from clang.cindex import Index, CursorKind, TokenKind, Config
    CLANG_AVAILABLE = True
except ImportError:
    CLANG_AVAILABLE = False
    print("Warning: clang module not available. Using regex fallback.")


@dataclass
class FunctionCall:
    """Represents a function call found in the code."""
    function_name: str
    file_path: str
    line_number: int
    column: int
    arguments: List[str] = field(default_factory=list)
    raw_arguments: List[str] = field(default_factory=list)
    containing_function: str = ""
    
    def __repr__(self):
        return f"FunctionCall({self.function_name}, args={self.arguments}, line={self.line_number})"


@dataclass
class VariableAssignment:
    """Represents a variable assignment."""
    variable_name: str
    assigned_value: str
    file_path: str
    line_number: int
    containing_function: str = ""


@dataclass
class FunctionDefinition:
    """Represents a function definition."""
    name: str
    file_path: str
    start_line: int
    end_line: int
    return_type: str = ""
    parameters: List[str] = field(default_factory=list)


class ASTParser:
    """Parse C source files using libclang or regex fallback."""
    
    def __init__(self, include_paths: List[str] = None):
        self.include_paths = include_paths or []
        self.function_calls: List[FunctionCall] = []
        self.variable_assignments: List[VariableAssignment] = []
        self.function_definitions: List[FunctionDefinition] = []
        
        if CLANG_AVAILABLE:
            self.index = Index.create()
        else:
            self.index = None
    
    def _get_clang_args(self) -> List[str]:
        """Get clang arguments for parsing."""
        args = ["-x", "c", "-ferror-limit=0"]
        for path in self.include_paths:
            args.extend(["-I", path])
        return args
    
    def parse_file(self, filepath: str) -> bool:
        """Parse a C file and extract information."""
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False
        
        if CLANG_AVAILABLE:
            return self._parse_with_clang(filepath)
        else:
            return self._parse_with_regex(filepath)
    
    def _parse_with_clang(self, filepath: str) -> bool:
        """Parse using libclang with regex fallback for missed calls."""
        try:
            tu = self.index.parse(filepath, self._get_clang_args())
            
            # Check for parse errors
            has_errors = any(d.severity >= 3 for d in tu.diagnostics)
            
            # Get clang results
            self._traverse_ast(tu.cursor, filepath)
            clang_calls = set(c.function_name for c in self.function_calls)
            
            # If there are errors OR if mpf_mfs_open wasn't found, supplement with regex
            if has_errors or 'mpf_mfs_open' not in clang_calls:
                # Store clang results
                clang_function_calls = self.function_calls.copy()
                clang_var_assignments = self.variable_assignments.copy()
                
                # Clear and run regex
                self.function_calls = []
                self.variable_assignments = []
                self._parse_with_regex(filepath)
                
                # Merge: add regex results that clang missed
                clang_call_keys = set((c.function_name, c.line_number) for c in clang_function_calls)
                for call in self.function_calls:
                    if (call.function_name, call.line_number) not in clang_call_keys:
                        clang_function_calls.append(call)
                
                clang_var_keys = set((v.variable_name, v.line_number) for v in clang_var_assignments)
                for var in self.variable_assignments:
                    if (var.variable_name, var.line_number) not in clang_var_keys:
                        clang_var_assignments.append(var)
                
                # Use merged results
                self.function_calls = clang_function_calls
                self.variable_assignments = clang_var_assignments
            
            return True
        except Exception as e:
            print(f"Clang parsing failed for {filepath}: {e}")
            return self._parse_with_regex(filepath)
    
    def _traverse_ast(self, cursor, filepath: str, parent_function: str = ""):
        """Traverse the AST and extract relevant nodes."""
        current_function = parent_function
        
        # Track function definitions
        if cursor.kind == CursorKind.FUNCTION_DECL and cursor.is_definition():
            func_def = FunctionDefinition(
                name=cursor.spelling,
                file_path=str(cursor.location.file) if cursor.location.file else filepath,
                start_line=cursor.extent.start.line,
                end_line=cursor.extent.end.line,
            )
            self.function_definitions.append(func_def)
            current_function = cursor.spelling
        
        # Track function calls
        if cursor.kind == CursorKind.CALL_EXPR:
            func_call = self._extract_function_call(cursor, filepath, current_function)
            if func_call:
                self.function_calls.append(func_call)
        
        # Track variable declarations with initialization
        if cursor.kind in [CursorKind.VAR_DECL, CursorKind.DECL_STMT]:
            var_assign = self._extract_variable_assignment(cursor, filepath, current_function)
            if var_assign:
                self.variable_assignments.append(var_assign)
        
        # Traverse children
        for child in cursor.get_children():
            self._traverse_ast(child, filepath, current_function)
    
    def _extract_function_call(self, cursor, filepath: str, containing_function: str) -> Optional[FunctionCall]:
        """Extract function call information from a CALL_EXPR cursor."""
        try:
            # Get function name
            func_name = cursor.spelling
            if not func_name:
                # Try to get from first child
                for child in cursor.get_children():
                    if child.kind == CursorKind.DECL_REF_EXPR:
                        func_name = child.spelling
                        break
            
            if not func_name:
                return None
            
            # Get location
            loc = cursor.location
            line = loc.line if loc.line else 0
            col = loc.column if loc.column else 0
            
            # Get arguments
            args = []
            for child in cursor.get_children():
                if child.kind != CursorKind.DECL_REF_EXPR:  # Skip function reference
                    # Get the text of the argument
                    tokens = list(child.get_tokens())
                    if tokens:
                        arg_text = " ".join(t.spelling for t in tokens)
                        args.append(arg_text)
            
            return FunctionCall(
                function_name=func_name,
                file_path=filepath,
                line_number=line,
                column=col,
                arguments=args,
                raw_arguments=args.copy(),
                containing_function=containing_function,
            )
        except Exception as e:
            print(f"Error extracting function call: {e}")
            return None
    
    def _extract_variable_assignment(self, cursor, filepath: str, containing_function: str) -> Optional[VariableAssignment]:
        """Extract variable assignment information."""
        try:
            var_name = cursor.spelling
            if not var_name:
                return None
            
            # Try to get assigned value
            children = list(cursor.get_children())
            if len(children) >= 2:
                # The second child is usually the initializer
                init_tokens = list(children[-1].get_tokens())
                if init_tokens:
                    value = " ".join(t.spelling for t in init_tokens)
                    return VariableAssignment(
                        variable_name=var_name,
                        assigned_value=value,
                        file_path=filepath,
                        line_number=cursor.location.line,
                        containing_function=containing_function,
                    )
            return None
        except Exception:
            return None
    
    def _parse_with_regex(self, filepath: str) -> bool:
        """Fallback parsing using regex (for when clang is not available)."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return False
        
        # Track current function
        current_function = ""
        brace_count = 0
        
        # Pattern for function definitions
        func_def_pattern = r'^\s*(?:int|void|char|float|double|long|short|unsigned|static|inline)\s+(\w+)\s*\([^)]*\)\s*\{'
        
        # Pattern for function calls: function_name(args)
        func_call_pattern = r'(\w+)\s*\(([^;]*)\)'
        
        # Pattern for variable assignments: type var = value; or var = value;
        var_assign_pattern = r'(\w+)\s*=\s*([^;]+);'
        
        for line_num, line in enumerate(lines, 1):
            # Track braces for function scope
            brace_count += line.count('{') - line.count('}')
            
            # Check for function definition
            func_match = re.match(func_def_pattern, line)
            if func_match:
                current_function = func_match.group(1)
            
            # Reset function when exiting
            if brace_count == 0:
                current_function = ""
            
            # Find function calls
            for match in re.finditer(func_call_pattern, line):
                func_name = match.group(1)
                args_str = match.group(2)
                
                # Skip common non-function patterns
                if func_name in ['if', 'while', 'for', 'switch', 'return']:
                    continue
                
                # Parse arguments
                args = self._parse_arguments(args_str)
                
                self.function_calls.append(FunctionCall(
                    function_name=func_name,
                    file_path=filepath,
                    line_number=line_num,
                    column=match.start(),
                    arguments=args,
                    raw_arguments=args.copy(),
                    containing_function=current_function,
                ))
            
            # Find variable assignments
            for match in re.finditer(var_assign_pattern, line):
                var_name = match.group(1)
                value = match.group(2).strip()
                
                self.variable_assignments.append(VariableAssignment(
                    variable_name=var_name,
                    assigned_value=value,
                    file_path=filepath,
                    line_number=line_num,
                    containing_function=current_function,
                ))
        
        return True
    
    def _parse_arguments(self, args_str: str) -> List[str]:
        """Parse a comma-separated argument string, handling nested parentheses."""
        args = []
        current_arg = ""
        paren_depth = 0
        
        for char in args_str:
            if char == '(':
                paren_depth += 1
                current_arg += char
            elif char == ')':
                paren_depth -= 1
                current_arg += char
            elif char == ',' and paren_depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char
        
        if current_arg.strip():
            args.append(current_arg.strip())
        
        return args
    
    def find_function_calls(self, function_name: str) -> List[FunctionCall]:
        """Find all calls to a specific function."""
        return [fc for fc in self.function_calls if fc.function_name == function_name]
    
    def find_variable_assignments(self, variable_name: str) -> List[VariableAssignment]:
        """Find all assignments to a specific variable."""
        return [va for va in self.variable_assignments if va.variable_name == variable_name]
    
    def get_function_definition(self, function_name: str) -> Optional[FunctionDefinition]:
        """Get the definition of a function by name."""
        for fd in self.function_definitions:
            if fd.name == function_name:
                return fd
        return None
    
    def clear(self):
        """Clear all parsed data."""
        self.function_calls.clear()
        self.variable_assignments.clear()
        self.function_definitions.clear()


def test_ast_parser():
    """Test the AST parser with a sample C file."""
    parser = ASTParser()
    
    test_code = '''
    #include <stdio.h>
    
    #define FILE_NUMBER (1234)
    
    int main(int argc, char *argv[]) {
        int ret;
        int fileno;
        
        fileno = FILE_NUMBER;
        
        ret = mpf_mfs_open(&fcb, NULL, fileno, 0, 0, 1);
        
        return 0;
    }
    '''
    
    # Write to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(test_code)
        temp_path = f.name
    
    # Parse
    parser.parse_file(temp_path)
    
    print("Function calls found:")
    for fc in parser.function_calls:
        print(f"  {fc}")
    
    print("\nVariable assignments found:")
    for va in parser.variable_assignments:
        print(f"  {va.variable_name} = {va.assigned_value}")
    
    # Cleanup
    os.unlink(temp_path)


if __name__ == "__main__":
    test_ast_parser()

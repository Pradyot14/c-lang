"""
Macro Extractor - Extract all #define macros from C files
"""
import re
import os
from typing import Dict, List, Tuple
from pathlib import Path


class MacroExtractor:
    """Extract and resolve macros from C source files."""
    
    def __init__(self):
        self.macros: Dict[str, str] = {}
        self.macro_sources: Dict[str, str] = {}  # macro -> file path
        
    def extract_from_file(self, filepath: str) -> Dict[str, str]:
        """Extract macros from a single file."""
        file_macros = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")
            return file_macros
        
        # Pattern for #define MACRO_NAME (value) or #define MACRO_NAME value
        patterns = [
            # #define NAME (value)
            r'#\s*define\s+(\w+)\s*\(([^)]+)\)',
            # #define NAME value
            r'#\s*define\s+(\w+)\s+([^\s\\]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for name, value in matches:
                # Clean up the value
                value = value.strip()
                file_macros[name] = value
                self.macros[name] = value
                self.macro_sources[name] = filepath
                
        return file_macros
    
    def extract_from_directory(self, dirpath: str, extensions: List[str] = None) -> Dict[str, str]:
        """Extract macros from all files in a directory."""
        if extensions is None:
            extensions = ['.h', '.c']
            
        for root, _, files in os.walk(dirpath):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    filepath = os.path.join(root, file)
                    self.extract_from_file(filepath)
                    
        return self.macros
    
    def resolve_macro(self, macro_name: str) -> Tuple[str, bool]:
        """
        Resolve a macro to its final value.
        Returns (value, is_numeric)
        """
        if macro_name not in self.macros:
            return macro_name, False
            
        value = self.macros[macro_name]
        
        # Try to resolve nested macros
        max_depth = 10
        depth = 0
        while depth < max_depth:
            # Check if value is another macro
            if value in self.macros:
                value = self.macros[value]
                depth += 1
            else:
                break
                
        # Try to evaluate as number
        try:
            # Handle expressions like (1234)
            clean_value = value.strip('()')
            numeric_value = int(clean_value)
            return str(numeric_value), True
        except ValueError:
            return value, False
    
    def evaluate_expression(self, expression: str) -> Tuple[int, bool]:
        """
        Evaluate an arithmetic expression that may contain macros.
        Returns (result, success)
        """
        # Replace macros with their values
        expr = expression
        
        for macro_name, macro_value in self.macros.items():
            if macro_name in expr:
                resolved, is_numeric = self.resolve_macro(macro_name)
                if is_numeric:
                    expr = expr.replace(macro_name, resolved)
        
        # Try to evaluate
        try:
            # Remove parentheses and clean up
            expr = expr.replace('(', '').replace(')', '')
            # Safe eval for arithmetic only
            result = eval(expr, {"__builtins__": {}}, {})
            return int(result), True
        except Exception:
            return 0, False
    
    def get_macro_value(self, name: str) -> int:
        """Get numeric value of a macro, returns -1 if not found/resolvable."""
        resolved, is_numeric = self.resolve_macro(name)
        if is_numeric:
            return int(resolved)
        return -1


def test_macro_extractor():
    """Test the macro extractor with sample macros."""
    extractor = MacroExtractor()
    
    # Test with sample content
    test_content = '''
    #define BASE_VALUE (1000)
    #define OFFSET_A (100)
    #define OFFSET_B (200)
    #define COMBINED (BASE_VALUE + OFFSET_A)
    '''
    
    # Write to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.h', delete=False) as f:
        f.write(test_content)
        temp_path = f.name
    
    # Extract
    macros = extractor.extract_from_file(temp_path)
    print(f"Extracted macros: {macros}")
    
    # Resolve
    for name in ['BASE_VALUE', 'OFFSET_A', 'OFFSET_B']:
        value, is_numeric = extractor.resolve_macro(name)
        print(f"{name} = {value} (numeric: {is_numeric})")
    
    # Test expression
    result, success = extractor.evaluate_expression("BASE_VALUE + OFFSET_B")
    print(f"BASE_VALUE + OFFSET_B = {result} (success: {success})")
    
    # Cleanup
    os.unlink(temp_path)


if __name__ == "__main__":
    test_macro_extractor()

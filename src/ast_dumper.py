"""
Complete AST Dumper using libclang
Generates detailed AST tree like clang's -ast-dump
"""
import os
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Try to import clang bindings
try:
    from clang.cindex import Index, CursorKind, TokenKind, Config
    CLANG_AVAILABLE = True
except ImportError:
    CLANG_AVAILABLE = False


@dataclass
class ASTNode:
    """Represents an AST node."""
    kind: str
    spelling: str
    file: str
    line: int
    column: int
    type_name: str
    children: List['ASTNode']
    
    def to_dict(self) -> Dict:
        return {
            'kind': self.kind,
            'spelling': self.spelling,
            'file': self.file,
            'line': self.line,
            'column': self.column,
            'type': self.type_name,
            'children': [c.to_dict() for c in self.children]
        }


class ASTDumper:
    """Dumps complete AST for C files."""
    
    def __init__(self, include_paths: List[str] = None):
        self.include_paths = include_paths or []
        
        if CLANG_AVAILABLE:
            self.index = Index.create()
        else:
            self.index = None
    
    def dump_file(self, filepath: str) -> str:
        """Dump complete AST for a file."""
        if CLANG_AVAILABLE:
            return self._dump_with_clang(filepath)
        else:
            return self._dump_with_regex(filepath)
    
    def dump_file_as_tree(self, filepath: str) -> Optional[ASTNode]:
        """Dump AST as tree structure."""
        if CLANG_AVAILABLE:
            return self._dump_tree_clang(filepath)
        else:
            return self._dump_tree_regex(filepath)
    
    def _dump_with_clang(self, filepath: str) -> str:
        """Dump using libclang."""
        try:
            args = ["-x", "c", "-ferror-limit=0"]
            for path in self.include_paths:
                args.extend(["-I", path])
            
            tu = self.index.parse(filepath, args)
            
            output = []
            self._traverse_and_dump(tu.cursor, filepath, output, 0)
            return '\n'.join(output)
        except Exception as e:
            return f"Clang parsing failed: {e}\n\n" + self._dump_with_regex(filepath)
    
    def _traverse_and_dump(self, cursor, filepath: str, output: List[str], depth: int):
        """Traverse AST and generate dump output."""
        # Filter to only nodes from our file
        if cursor.location.file and cursor.location.file.name != filepath:
            if depth > 0:  # Allow root level
                return
        
        # Build node info
        kind_name = cursor.kind.name if hasattr(cursor.kind, 'name') else str(cursor.kind)
        spelling = cursor.spelling or cursor.displayname or ""
        
        # Get location
        loc = cursor.location
        file_name = os.path.basename(loc.file.name) if loc.file else ""
        line = loc.line
        col = loc.column
        
        # Get type
        type_str = ""
        if cursor.type and cursor.type.spelling:
            type_str = cursor.type.spelling
        
        # Build indent
        indent = "  " * depth
        prefix = "├─ " if depth > 0 else ""
        
        # Format output
        if spelling:
            node_str = f"{indent}{prefix}{kind_name} '{spelling}'"
        else:
            node_str = f"{indent}{prefix}{kind_name}"
        
        if file_name and line:
            node_str += f" {file_name}:{line}:{col}"
        
        if type_str:
            node_str += f" type={type_str}"
        
        output.append(node_str)
        
        # Recurse children
        for child in cursor.get_children():
            self._traverse_and_dump(child, filepath, output, depth + 1)
    
    def _dump_tree_clang(self, filepath: str) -> Optional[ASTNode]:
        """Build AST tree using clang."""
        try:
            args = ["-x", "c", "-ferror-limit=0"]
            for path in self.include_paths:
                args.extend(["-I", path])
            
            tu = self.index.parse(filepath, args)
            return self._build_tree(tu.cursor, filepath)
        except:
            return None
    
    def _build_tree(self, cursor, filepath: str) -> Optional[ASTNode]:
        """Build tree node from cursor."""
        # Get location info
        loc = cursor.location
        file_name = os.path.basename(loc.file.name) if loc.file else ""
        
        # Get type
        type_str = cursor.type.spelling if cursor.type else ""
        
        # Build children
        children = []
        for child in cursor.get_children():
            child_node = self._build_tree(child, filepath)
            if child_node:
                children.append(child_node)
        
        return ASTNode(
            kind=cursor.kind.name if hasattr(cursor.kind, 'name') else str(cursor.kind),
            spelling=cursor.spelling or cursor.displayname or "",
            file=file_name,
            line=loc.line,
            column=loc.column,
            type_name=type_str,
            children=children
        )
    
    def _dump_with_regex(self, filepath: str) -> str:
        """Fallback: Parse with regex and generate pseudo-AST."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return "Failed to read file"
        
        output = []
        is_header = filepath.endswith('.h')
        output.append(f"--- Pseudo-AST for {os.path.basename(filepath)} (regex-based) ---")
        output.append("")
        
        lines = content.split('\n')
        filename = os.path.basename(filepath)
        
        # Track context
        current_function = None
        brace_depth = 0
        in_struct = False
        struct_name = None
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                continue
            
            # Macro definitions (important for header files)
            macro_match = re.match(r'#\s*define\s+(\w+)(?:\s+(.+))?', stripped)
            if macro_match:
                macro_name = macro_match.group(1)
                macro_value = macro_match.group(2) or ""
                macro_value = macro_value.strip()
                if macro_value:
                    output.append(f"MACRO_DEFINITION '{macro_name}' {filename}:{line_num} value={macro_value}")
                else:
                    output.append(f"MACRO_DEFINITION '{macro_name}' {filename}:{line_num}")
                continue
            
            # Include statements
            include_match = re.match(r'#\s*include\s*[<"]([^>"]+)[>"]', stripped)
            if include_match:
                include_file = include_match.group(1)
                output.append(f"INCLUSION_DIRECTIVE '{include_file}' {filename}:{line_num}")
                continue
            
            # Typedef
            typedef_match = re.match(r'typedef\s+(.+?)\s+(\w+)\s*;', stripped)
            if typedef_match:
                orig_type, new_type = typedef_match.groups()
                output.append(f"TYPEDEF_DECL '{new_type}' {filename}:{line_num} type={orig_type}")
                continue
            
            # Struct/Union definition
            struct_match = re.match(r'(typedef\s+)?(struct|union)\s+(\w+)?\s*\{?', stripped)
            if struct_match and '{' in stripped:
                typedef_prefix = struct_match.group(1)
                struct_type = struct_match.group(2)
                struct_name = struct_match.group(3) or "anonymous"
                in_struct = True
                output.append(f"STRUCT_DECL '{struct_name}' {filename}:{line_num} type={struct_type}")
                continue
            
            # Struct member
            if in_struct and brace_depth > 0:
                member_match = re.match(r'(\w+(?:\s*\*)?)\s+(\w+)(?:\[(\d+)\])?\s*;', stripped)
                if member_match:
                    member_type, member_name, array_size = member_match.groups()
                    if array_size:
                        output.append(f"  ├─ FIELD_DECL '{member_name}' {filename}:{line_num} type={member_type}[{array_size}]")
                    else:
                        output.append(f"  ├─ FIELD_DECL '{member_name}' {filename}:{line_num} type={member_type}")
            
            # Extern declaration
            extern_match = re.match(r'extern\s+(.+?)\s+(\w+)\s*;', stripped)
            if extern_match:
                extern_type, extern_name = extern_match.groups()
                output.append(f"VAR_DECL '{extern_name}' {filename}:{line_num} type=extern {extern_type}")
                continue
            
            # Function prototype (no body - common in headers)
            proto_match = re.match(r'(?:extern\s+)?(\w+(?:\s*\*)?)\s+(\w+)\s*\(([^)]*)\)\s*;', stripped)
            if proto_match:
                ret_type, func_name, params = proto_match.groups()
                output.append(f"FUNCTION_DECL '{func_name}' {filename}:{line_num} type={ret_type} (prototype)")
                for param in params.split(','):
                    param = param.strip()
                    if param and param != 'void':
                        parts = param.replace('*', ' * ').split()
                        if parts:
                            param_name = parts[-1].replace('*', '')
                            param_type = ' '.join(parts[:-1]) if len(parts) > 1 else 'int'
                            output.append(f"  ├─ PARM_DECL '{param_name}' {filename}:{line_num} type={param_type}")
                continue
            
            # Function definition (with body)
            func_match = re.match(r'(?:static\s+)?(?:inline\s+)?(\w+(?:\s*\*)?)\s+(\w+)\s*\(([^)]*)\)\s*\{?', stripped)
            if func_match and brace_depth == 0 and '{' in stripped:
                ret_type, func_name, params = func_match.groups()
                current_function = func_name
                output.append(f"FUNCTION_DECL '{func_name}' {filename}:{line_num} type={ret_type}")
                
                # Parse parameters
                for param in params.split(','):
                    param = param.strip()
                    if param and param != 'void':
                        parts = param.replace('*', ' * ').split()
                        if parts:
                            param_name = parts[-1].replace('*', '')
                            param_type = ' '.join(parts[:-1]) if len(parts) > 1 else 'int'
                            output.append(f"  ├─ PARM_DECL '{param_name}' {filename}:{line_num} type={param_type}")
            
            # Track brace depth
            brace_depth += stripped.count('{') - stripped.count('}')
            if brace_depth == 0:
                current_function = None
            
            if current_function:
                indent = "  │ "
                
                # Variable declaration
                var_match = re.match(r'(\w+(?:\s*\*)?)\s+(\w+)\s*(?:=|;)', stripped)
                if var_match and not func_match:
                    var_type, var_name = var_match.groups()
                    output.append(f"{indent}VAR_DECL '{var_name}' {filename}:{line_num} type={var_type}")
                
                # Function call
                call_matches = re.finditer(r'(\w+)\s*\(([^)]*)\)', stripped)
                for call_match in call_matches:
                    func_name, args = call_match.groups()
                    if func_name not in ['if', 'while', 'for', 'switch', 'return', 'sizeof']:
                        output.append(f"{indent}CALL_EXPR '{func_name}' {filename}:{line_num}")
                        # Parse arguments
                        for i, arg in enumerate(args.split(',')):
                            arg = arg.strip()
                            if arg:
                                if arg.isdigit():
                                    output.append(f"{indent}  ├─ INTEGER_LITERAL '{arg}' {filename}:{line_num}")
                                elif arg.startswith('&'):
                                    output.append(f"{indent}  ├─ UNARY_OPERATOR '&' {filename}:{line_num}")
                                    output.append(f"{indent}  │   └─ DECL_REF_EXPR '{arg[1:]}' {filename}:{line_num}")
                                elif re.match(r'^[A-Z_][A-Z0-9_]*$', arg):
                                    output.append(f"{indent}  ├─ MACRO_REF '{arg}' {filename}:{line_num}")
                                else:
                                    output.append(f"{indent}  ├─ DECL_REF_EXPR '{arg}' {filename}:{line_num}")
                
                # If statement
                if stripped.startswith('if'):
                    output.append(f"{indent}IF_STMT {filename}:{line_num}")
                
                # Return statement
                if stripped.startswith('return'):
                    output.append(f"{indent}RETURN_STMT {filename}:{line_num}")
                    ret_match = re.search(r'return\s*\(?\s*([^;)]+)', stripped)
                    if ret_match:
                        ret_val = ret_match.group(1).strip()
                        if ret_val.lstrip('-').isdigit():
                            output.append(f"{indent}  └─ INTEGER_LITERAL '{ret_val}' {filename}:{line_num}")
        
        return '\n'.join(output)
    
    def _dump_tree_regex(self, filepath: str) -> Optional[ASTNode]:
        """Build pseudo-AST tree using regex."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return None
        
        filename = os.path.basename(filepath)
        lines = content.split('\n')
        
        # Root node
        root = ASTNode(
            kind="TRANSLATION_UNIT",
            spelling=filename,
            file=filename,
            line=1,
            column=1,
            type_name="",
            children=[]
        )
        
        current_function_node = None
        brace_depth = 0
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                continue
            
            # Function definition
            func_match = re.match(r'(?:static\s+)?(?:inline\s+)?(\w+(?:\s*\*)?)\s+(\w+)\s*\(([^)]*)\)\s*\{?', stripped)
            if func_match and brace_depth == 0:
                ret_type, func_name, params = func_match.groups()
                
                func_node = ASTNode(
                    kind="FUNCTION_DECL",
                    spelling=func_name,
                    file=filename,
                    line=line_num,
                    column=1,
                    type_name=ret_type,
                    children=[]
                )
                
                # Add parameters
                for param in params.split(','):
                    param = param.strip()
                    if param and param != 'void':
                        parts = param.replace('*', ' * ').split()
                        if parts:
                            param_name = parts[-1].replace('*', '')
                            param_type = ' '.join(parts[:-1]) if len(parts) > 1 else 'int'
                            func_node.children.append(ASTNode(
                                kind="PARM_DECL",
                                spelling=param_name,
                                file=filename,
                                line=line_num,
                                column=1,
                                type_name=param_type,
                                children=[]
                            ))
                
                # Add compound statement for body
                body_node = ASTNode(
                    kind="COMPOUND_STMT",
                    spelling="",
                    file=filename,
                    line=line_num,
                    column=1,
                    type_name="",
                    children=[]
                )
                func_node.children.append(body_node)
                
                root.children.append(func_node)
                current_function_node = body_node
            
            brace_depth += stripped.count('{') - stripped.count('}')
            if brace_depth == 0:
                current_function_node = None
            
            if current_function_node:
                # Variable declaration
                var_match = re.match(r'(\w+(?:\s*\*)?)\s+(\w+)\s*(?:=|;)', stripped)
                if var_match and not func_match:
                    var_type, var_name = var_match.groups()
                    current_function_node.children.append(ASTNode(
                        kind="VAR_DECL",
                        spelling=var_name,
                        file=filename,
                        line=line_num,
                        column=1,
                        type_name=var_type,
                        children=[]
                    ))
                
                # Function calls
                for call_match in re.finditer(r'(\w+)\s*\(([^)]*)\)', stripped):
                    func_name, args = call_match.groups()
                    if func_name not in ['if', 'while', 'for', 'switch', 'return', 'sizeof']:
                        call_node = ASTNode(
                            kind="CALL_EXPR",
                            spelling=func_name,
                            file=filename,
                            line=line_num,
                            column=call_match.start() + 1,
                            type_name="",
                            children=[]
                        )
                        
                        # Add arguments
                        for arg in args.split(','):
                            arg = arg.strip()
                            if arg:
                                if arg.isdigit() or (arg.startswith('-') and arg[1:].isdigit()):
                                    call_node.children.append(ASTNode(
                                        kind="INTEGER_LITERAL",
                                        spelling=arg,
                                        file=filename,
                                        line=line_num,
                                        column=1,
                                        type_name="int",
                                        children=[]
                                    ))
                                elif arg.startswith('&'):
                                    unary = ASTNode(
                                        kind="UNARY_OPERATOR",
                                        spelling="&",
                                        file=filename,
                                        line=line_num,
                                        column=1,
                                        type_name="",
                                        children=[ASTNode(
                                            kind="DECL_REF_EXPR",
                                            spelling=arg[1:],
                                            file=filename,
                                            line=line_num,
                                            column=1,
                                            type_name="",
                                            children=[]
                                        )]
                                    )
                                    call_node.children.append(unary)
                                elif re.match(r'^[A-Z_][A-Z0-9_]*$', arg):
                                    call_node.children.append(ASTNode(
                                        kind="MACRO_INSTANTIATION",
                                        spelling=arg,
                                        file=filename,
                                        line=line_num,
                                        column=1,
                                        type_name="",
                                        children=[]
                                    ))
                                else:
                                    call_node.children.append(ASTNode(
                                        kind="DECL_REF_EXPR",
                                        spelling=arg,
                                        file=filename,
                                        line=line_num,
                                        column=1,
                                        type_name="",
                                        children=[]
                                    ))
                        
                        current_function_node.children.append(call_node)
                
                # If statement
                if stripped.startswith('if'):
                    current_function_node.children.append(ASTNode(
                        kind="IF_STMT",
                        spelling="",
                        file=filename,
                        line=line_num,
                        column=1,
                        type_name="",
                        children=[]
                    ))
                
                # Return statement
                if stripped.startswith('return'):
                    ret_node = ASTNode(
                        kind="RETURN_STMT",
                        spelling="",
                        file=filename,
                        line=line_num,
                        column=1,
                        type_name="",
                        children=[]
                    )
                    ret_match = re.search(r'return\s*\(?\s*([^;)]+)', stripped)
                    if ret_match:
                        ret_val = ret_match.group(1).strip()
                        if ret_val.lstrip('-').isdigit():
                            ret_node.children.append(ASTNode(
                                kind="INTEGER_LITERAL",
                                spelling=ret_val,
                                file=filename,
                                line=line_num,
                                column=1,
                                type_name="int",
                                children=[]
                            ))
                    current_function_node.children.append(ret_node)
        
        return root


def dump_project_ast(project_dir: str, include_paths: List[str] = None) -> Dict[str, str]:
    """Dump AST for all C files in a project."""
    dumper = ASTDumper(include_paths)
    results = {}
    
    for root, _, files in os.walk(project_dir):
        for f in files:
            if f.endswith('.c'):
                filepath = os.path.join(root, f)
                results[f] = dumper.dump_file(filepath)
    
    return results


def dump_project_ast_trees(project_dir: str, include_paths: List[str] = None) -> Dict[str, ASTNode]:
    """Get AST trees for all C files in a project."""
    dumper = ASTDumper(include_paths)
    results = {}
    
    for root, _, files in os.walk(project_dir):
        for f in files:
            if f.endswith('.c'):
                filepath = os.path.join(root, f)
                tree = dumper.dump_file_as_tree(filepath)
                if tree:
                    results[f] = tree
    
    return results

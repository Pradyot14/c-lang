#!/usr/bin/env python3
"""
C Call Graph Parser v8.1 - Nightmare Edition
=============================================
Handles extreme C patterns including:
- Cast transparency (function pointers through casts)
- Union member aliasing  
- Function call return value tracking (var = func() tracks return, not func)
- Linked list node collection
- Triple+ function pointer indirection

Author: Pradyot
"""

import os
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict

from clang.cindex import Index, CursorKind, TranslationUnit, TokenKind


class C:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


@dataclass
class MacroInfo:
    name: str
    value: str
    file_path: str
    line: int
    tokens: List[str] = field(default_factory=list)

@dataclass
class FunctionInfo:
    name: str
    file_path: str
    line: int
    calls: Set[str] = field(default_factory=set)

@dataclass
class CallSite:
    caller: str
    callee: str
    file_path: str
    line: int
    macro_name: Optional[str] = None
    macro_file: Optional[str] = None
    indirect_via: Optional[str] = None

@dataclass 
class CallPath:
    path: List[str]
    call_sites: List[CallSite] = field(default_factory=list)
    
    def __str__(self):
        return ' → '.join(self.path)


class CParser:
    """
    Ultimate robust C parser v8.1 - handles nightmare C patterns.
    """
    
    def __init__(self, include_paths: List[str] = None, verbose: bool = False):
        self.include_paths = include_paths or []
        self.verbose = verbose
        self.index = Index.create()
        
        self.registry = self._load_registry()
        
        # Core storage
        self.functions: Dict[str, FunctionInfo] = {}
        self.macros: Dict[str, MacroInfo] = {}
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)
        self.call_sites: List[CallSite] = []
        self.main_file: Optional[str] = None
        
        # All known function names
        self.all_functions: Set[str] = set()
        
        # Global variables
        self.global_vars: Set[str] = set()
        
        # UNIFIED data flow storage
        self.data_flow: Dict[str, Set[str]] = defaultdict(set)
        
        # Return value tracking
        self.return_values: Dict[str, Set[str]] = defaultdict(set)
        
        # Parameter name mapping
        self.param_names: Dict[Tuple[str, int], str] = {}
        
        # Track all members of each base variable (for union aliasing)
        self.base_members: Dict[str, Set[str]] = defaultdict(set)
        
        # v8.1: Track all .member assignments in each scope (for linked list collection)
        # Key: "scope:member_suffix" -> Set of possible values
        self.scope_member_values: Dict[str, Set[str]] = defaultdict(set)
        
        # v8.2: GLOBAL member values - track ALL .member assignments across entire codebase
        # Key: "member_name" -> Set of all functions ever assigned to any X.member_name
        self.global_member_values: Dict[str, Set[str]] = defaultdict(set)
        
        # v8.2: Track variables that are linked list chain variables (updated via .next)
        # Key: "scope:var_name" -> True if this var is updated via next-style patterns
        self.chain_variables: Set[str] = set()
        
        # v8.2: Track linked chains - which variables are linked together via .next
        # Key: "scope:var.next" -> set of "@ref:..." values pointing to next nodes
        self.next_chains: Dict[str, Set[str]] = defaultdict(set)
        
        # Store parsed TUs
        self.translation_units: Dict[str, Any] = {}
    
    def _load_registry(self) -> Dict:
        registry_path = Path(__file__).parent / "function_registry.json"
        if registry_path.exists():
            try:
                with open(registry_path) as f:
                    return json.load(f)
            except:
                pass
        return {"functions": {}}
    
    def _get_callback_indices(self, func_name: str) -> List[int]:
        funcs = self.registry.get("functions", {})
        if func_name in funcs:
            args = funcs[func_name].get("arguments", [])
            return [i for i, a in enumerate(args) if a.get("is_callback")]
        return []
    
    def _get_clang_args(self) -> List[str]:
        args = [
            '-std=c11', 
            '-Wno-everything', 
            '-ferror-limit=0', 
            '-fno-spell-checking',
            '-DNULL=0',
            '-Dsize_t=unsigned long',
            '-Dssize_t=long',
        ]
        for inc in self.include_paths:
            args.append(f'-I{inc}')
        return args
    
    # ==================== MAIN PARSING ====================
    
    def parse_directory(self, directory: str) -> None:
        """Parse all C files with multi-pass cross-file analysis."""
        c_files = list(Path(directory).glob("*.c"))
        
        if not c_files:
            if self.verbose:
                print(f"  {C.BRIGHT_YELLOW}⚠️  No .c files found{C.RESET}")
            return
        
        for c_file in c_files:
            self._parse_and_store(str(c_file))
        
        if self.verbose:
            print(f"  {C.DIM}📋 Pass 1: Collecting declarations...{C.RESET}")
        for file_path, tu in self.translation_units.items():
            self._pass1_collect_declarations(tu.cursor, file_path)
        
        if self.verbose:
            print(f"  {C.DIM}📋 Pass 2: Tracking data flow...{C.RESET}")
        for file_path, tu in self.translation_units.items():
            self._pass2_track_data_flow(tu.cursor, file_path, "global")
        
        if self.verbose:
            print(f"  {C.DIM}📋 Pass 3: Extracting calls...{C.RESET}")
        for file_path, tu in self.translation_units.items():
            self._pass3_extract_calls(tu.cursor, file_path)
        
        for func_name, func_info in self.functions.items():
            for callee in func_info.calls:
                self.call_graph[func_name].add(callee)
    
    def _parse_and_store(self, file_path: str) -> None:
        if self.verbose:
            print(f"  {C.DIM}📄 Parsing:{C.RESET} {C.WHITE}{os.path.basename(file_path)}{C.RESET}")
        try:
            tu = self.index.parse(file_path, args=self._get_clang_args(),
                options=TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
            self.translation_units[file_path] = tu
        except Exception as e:
            if self.verbose:
                print(f"    {C.BRIGHT_RED}Error: {e}{C.RESET}")
    
    # ==================== PASS 1: DECLARATIONS ====================
    
    def _pass1_collect_declarations(self, cursor, file_path: str) -> None:
        for child in cursor.get_children():
            if child.kind == CursorKind.FUNCTION_DECL:
                self.all_functions.add(child.spelling)
                
                if child.is_definition():
                    param_idx = 0
                    for param in child.get_children():
                        if param.kind == CursorKind.PARM_DECL:
                            self.param_names[(child.spelling, param_idx)] = param.spelling
                            param_idx += 1
            
            if child.kind == CursorKind.VAR_DECL:
                parent = child.semantic_parent
                if parent and parent.kind == CursorKind.TRANSLATION_UNIT:
                    self.global_vars.add(child.spelling)
            
            if child.kind == CursorKind.MACRO_DEFINITION:
                name = child.spelling
                if not name.startswith('__'):
                    loc = child.location
                    if loc.file:
                        tokens = [t.spelling for t in child.get_tokens()]
                        value = ""
                        if len(tokens) > 1:
                            if tokens[1] != '(':
                                value = tokens[1]
                            else:
                                in_body = False
                                for t in tokens[1:]:
                                    if in_body:
                                        value = t
                                        break
                                    if t == ')':
                                        in_body = True
                        
                        self.macros[name] = MacroInfo(
                            name=name,
                            value=value,
                            file_path=loc.file.name,
                            line=loc.line,
                            tokens=tokens[1:] if len(tokens) > 1 else []
                        )
            
            self._pass1_collect_declarations(child, file_path)
    
    # ==================== PASS 2: DATA FLOW ====================
    
    def _is_global(self, var_name: str) -> bool:
        return var_name in self.global_vars
    
    def _make_key(self, scope: str, location: str) -> str:
        return f"{scope}:{location}"
    
    def _unwrap_expr(self, cursor) -> Any:
        """Unwrap UNEXPOSED_EXPR and PAREN_EXPR to get the real expression."""
        while cursor.kind in (CursorKind.UNEXPOSED_EXPR, CursorKind.PAREN_EXPR):
            children = list(cursor.get_children())
            if children:
                cursor = children[0]
            else:
                break
        return cursor
    
    def _unwrap_cast(self, cursor) -> Any:
        """Unwrap cast expressions to get the underlying expression."""
        cursor = self._unwrap_expr(cursor)
        while cursor.kind == CursorKind.CSTYLE_CAST_EXPR:
            children = list(cursor.get_children())
            for child in children:
                if child.kind != CursorKind.TYPE_REF:
                    cursor = self._unwrap_expr(child)
                    break
            else:
                break
        return cursor
    
    def _extract_call_callee(self, cursor) -> Optional[str]:
        """
        v8.1: Extract the function name being called from a CALL_EXPR.
        Returns None if not a simple function call.
        """
        cursor = self._unwrap_cast(cursor)
        if cursor.kind != CursorKind.CALL_EXPR:
            return None
        
        children = list(cursor.get_children())
        if not children:
            return cursor.spelling if cursor.spelling else None
        
        target = self._unwrap_cast(children[0])
        
        if target.kind == CursorKind.DECL_REF_EXPR:
            return target.spelling
        
        return cursor.spelling if cursor.spelling else None
    
    def _extract_call_target_location(self, cursor) -> Optional[str]:
        """
        v8.1: Extract the location being called from a CALL_EXPR.
        Returns the location string for indirect calls (through variables, members, etc.)
        """
        cursor = self._unwrap_cast(cursor)
        if cursor.kind != CursorKind.CALL_EXPR:
            return None
        
        children = list(cursor.get_children())
        if not children:
            return None
        
        target = self._unwrap_cast(children[0])
        return self._extract_location(target)
    
    def _extract_all_functions_deep(self, cursor, depth: int = 0) -> Set[str]:
        """
        Extract ALL function references from ANY expression - DEEPLY.
        v8.1: Does NOT follow into CALL_EXPR - those need special handling for return values.
        """
        if depth > 100:
            return set()
        
        results = set()
        kind = cursor.kind
        
        # Direct function reference
        if kind == CursorKind.DECL_REF_EXPR:
            name = cursor.spelling
            if name in self.all_functions:
                results.add(name)
            return results
        
        # v8.1: CALL_EXPR - don't extract the function, it will be handled separately
        # Only recurse into arguments, not the callee
        if kind == CursorKind.CALL_EXPR:
            children = list(cursor.get_children())
            # Skip first child (callee), only check arguments
            for arg in children[1:]:
                results.update(self._extract_all_functions_deep(arg, depth + 1))
            return results
        
        # Cast: look through
        if kind == CursorKind.CSTYLE_CAST_EXPR:
            for child in cursor.get_children():
                if child.kind != CursorKind.TYPE_REF:
                    results.update(self._extract_all_functions_deep(child, depth + 1))
            return results
        
        # Ternary: get BOTH branches
        if kind == CursorKind.CONDITIONAL_OPERATOR:
            children = list(cursor.get_children())
            if len(children) >= 3:
                results.update(self._extract_all_functions_deep(children[1], depth + 1))
                results.update(self._extract_all_functions_deep(children[2], depth + 1))
            return results
        
        # For ALL other nodes, recurse into ALL children
        for child in cursor.get_children():
            results.update(self._extract_all_functions_deep(child, depth + 1))
        
        return results
    
    def _extract_initializer_value(self, cursor, scope: str) -> Set[str]:
        """
        v8.1: Extract what a variable initializer evaluates to.
        For CALL_EXPR with direct function: returns @ret:func_name marker
        For CALL_EXPR with indirect (variable): returns @fptr_ret:scope:var marker
        For direct function refs: returns the function name
        """
        results = set()
        cursor = self._unwrap_cast(cursor)
        kind = cursor.kind
        
        # CALL_EXPR: var = func() or var = fptr_var()
        if kind == CursorKind.CALL_EXPR:
            callee = self._extract_call_callee(cursor)
            if callee:
                if callee in self.all_functions:
                    # Direct function call: var = func()
                    results.add(f"@ret:{callee}")
                else:
                    # Indirect call through variable: var = fptr_var()
                    results.add(f"@fptr_ret:{scope}:{callee}")
            return results
        
        # Direct function reference
        if kind == CursorKind.DECL_REF_EXPR:
            name = cursor.spelling
            if name in self.all_functions:
                results.add(name)
            return results
        
        # Ternary: both branches
        if kind == CursorKind.CONDITIONAL_OPERATOR:
            children = list(cursor.get_children())
            if len(children) >= 3:
                results.update(self._extract_initializer_value(children[1], scope))
                results.update(self._extract_initializer_value(children[2], scope))
            return results
        
        # For other expressions, try extracting direct function refs
        funcs = self._extract_all_functions_deep(cursor)
        results.update(funcs)
        
        return results
    
    def _extract_nested_call_chain(self, cursor) -> Optional[str]:
        """Detect nested call pattern: get_factory()(args)"""
        cursor = self._unwrap_cast(cursor)
        
        if cursor.kind != CursorKind.CALL_EXPR:
            return None
        
        children = list(cursor.get_children())
        if not children:
            return None
        
        target = self._unwrap_cast(children[0])
        
        # If target is a CALL_EXPR, this is a nested call: f()()
        if target.kind == CursorKind.CALL_EXPR:
            inner_callee = self._extract_call_callee(target)
            if inner_callee:
                return f"@call_ret:{inner_callee}"
        
        return None
    
    def _extract_location(self, cursor) -> Optional[str]:
        """Extract a location string from any lvalue expression."""
        kind = cursor.kind
        
        if kind == CursorKind.DECL_REF_EXPR:
            return cursor.spelling
        
        if kind == CursorKind.MEMBER_REF_EXPR:
            member = cursor.spelling
            children = list(cursor.get_children())
            if children:
                base = self._extract_location(children[0])
                if base:
                    return f"{base}.{member}"
            return member
        
        if kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            children = list(cursor.get_children())
            if children:
                base = self._extract_location(children[0])
                if base:
                    return f"{base}[*]"
        
        if kind == CursorKind.UNARY_OPERATOR:
            children = list(cursor.get_children())
            if children:
                tokens = [t.spelling for t in cursor.get_tokens()]
                if tokens:
                    op = tokens[0]
                    inner = self._extract_location(children[0])
                    if inner:
                        if op == '*':
                            return f"*{inner}"
                        elif op == '&':
                            return f"&{inner}"
                        return inner
        
        if kind == CursorKind.PAREN_EXPR:
            children = list(cursor.get_children())
            if children:
                return self._extract_location(children[0])
        
        if kind == CursorKind.UNEXPOSED_EXPR:
            children = list(cursor.get_children())
            if children:
                return self._extract_location(children[0])
        
        if kind == CursorKind.CSTYLE_CAST_EXPR:
            children = list(cursor.get_children())
            for child in children:
                if child.kind != CursorKind.TYPE_REF:
                    return self._extract_location(child)
        
        if kind == CursorKind.BINARY_OPERATOR:
            children = list(cursor.get_children())
            if children:
                return self._extract_location(children[0])
        
        return None
    
    def _process_init_list(self, cursor, base_var: str, scope: str, member_prefix: str = "") -> None:
        """Process an initializer list recursively."""
        for i, child in enumerate(cursor.get_children()):
            member_name = None
            
            for subchild in child.get_children():
                if subchild.kind == CursorKind.MEMBER_REF:
                    member_name = subchild.spelling
                    break
            
            if member_name:
                if member_prefix:
                    full_path = f"{base_var}.{member_prefix}.{member_name}"
                else:
                    full_path = f"{base_var}.{member_name}"
            else:
                if member_prefix:
                    full_path = f"{base_var}.{member_prefix}[*]"
                else:
                    full_path = f"{base_var}[*]"
            
            nested_init = None
            for subchild in child.get_children():
                if subchild.kind == CursorKind.INIT_LIST_EXPR:
                    nested_init = subchild
                    break
            
            if nested_init:
                new_prefix = member_name if member_name else ""
                self._process_init_list(nested_init, base_var, scope, new_prefix)
            else:
                funcs = self._extract_all_functions_deep(child)
                if funcs:
                    key = self._make_key(scope, full_path)
                    self.data_flow[key].update(funcs)
                    if self.verbose:
                        print(f"    {C.DIM}📌 {key} = {funcs}{C.RESET}")
                    # v8.2: Track member values for linked list collection
                    self._track_member_value(scope, full_path, funcs)
    
    def _track_member_for_base(self, scope: str, location: str) -> None:
        """Track that a base variable has this member (for union aliasing)."""
        if '.' in location:
            parts = location.split('.', 1)
            base = parts[0].split('[')[0].lstrip('*&')
            member = parts[1]
            key = self._make_key(scope, base)
            self.base_members[key].add(member)
    
    def _track_member_value(self, scope: str, location: str, funcs: Set[str]) -> None:
        """v8.2: Track all values assigned to a particular member suffix - both scoped and global."""
        if '.' in location:
            parts = location.rsplit('.', 1)
            member_suffix = parts[1]  # e.g., "first", "handler", "callback"
            
            # Track in scope (original v8.1 behavior)
            key = self._make_key(scope, f"@member:{member_suffix}")
            self.scope_member_values[key].update(funcs)
            
            # v8.2: Also track GLOBALLY for cross-scope linked list resolution
            self.global_member_values[member_suffix].update(funcs)
    
    def _track_chain_variable(self, scope: str, var_name: str, next_ref: str) -> None:
        """v8.2: Track that a variable is part of a linked list chain."""
        # Track that this variable is updated via next-style pattern
        key = self._make_key(scope, var_name)
        self.chain_variables.add(key)
        
        # Track the next chain reference
        self.next_chains[key].add(next_ref)
    
    def _pass2_track_data_flow(self, cursor, file_path: str, scope: str) -> None:
        """Track ALL data flow - assignments, returns, initializers."""
        kind = cursor.kind
        
        if kind == CursorKind.FUNCTION_DECL and cursor.is_definition():
            scope = cursor.spelling
        
        # Variable declaration with initializer
        if kind == CursorKind.VAR_DECL:
            var_name = cursor.spelling
            var_scope = "global" if self._is_global(var_name) else scope
            key = self._make_key(var_scope, var_name)
            
            # Check each child for initializer
            for child in cursor.get_children():
                # Skip type refs
                if child.kind == CursorKind.TYPE_REF:
                    continue
                
                # v8.1: Check for nested call pattern first: var = outer()(args)
                nested_call_result = self._extract_nested_call_chain(child)
                if nested_call_result:
                    self.data_flow[key].add(nested_call_result)
                    if self.verbose:
                        print(f"    {C.DIM}📌 {key} = {nested_call_result} (nested call){C.RESET}")
                    continue
                
                # v8.1: Extract initializer value properly (handles CALL_EXPR -> @ret:func)
                values = self._extract_initializer_value(child, var_scope)
                if values:
                    self.data_flow[key].update(values)
                    if self.verbose:
                        print(f"    {C.DIM}📌 {key} = {values}{C.RESET}")
                
                # Process initializer lists
                if child.kind == CursorKind.INIT_LIST_EXPR:
                    self._process_init_list(child, var_name, var_scope)
                
                if child.kind in (CursorKind.UNEXPOSED_EXPR, CursorKind.COMPOUND_LITERAL_EXPR):
                    for subchild in child.get_children():
                        if subchild.kind == CursorKind.COMPOUND_LITERAL_EXPR:
                            for sub2 in subchild.get_children():
                                if sub2.kind == CursorKind.INIT_LIST_EXPR:
                                    self._process_init_list(sub2, var_name, var_scope)
                        if subchild.kind == CursorKind.INIT_LIST_EXPR:
                            self._process_init_list(subchild, var_name, var_scope)
                
                # Variable reference chain (only if not a function or @ret marker)
                loc = self._extract_location(child)
                if loc and loc not in self.all_functions:
                    unwrapped = self._unwrap_cast(child)
                    if unwrapped.kind != CursorKind.CALL_EXPR:
                        self.data_flow[key].add(f"@ref:{loc}")
        
        # Binary operator (assignment)
        if kind == CursorKind.BINARY_OPERATOR:
            children = list(cursor.get_children())
            if len(children) == 2:
                lhs, rhs = children
                lhs_loc = self._extract_location(lhs)
                
                if lhs_loc:
                    base_var = lhs_loc.split('.')[0].split('[')[0].lstrip('*&')
                    lhs_scope = "global" if self._is_global(base_var) else scope
                    key = self._make_key(lhs_scope, lhs_loc)
                    
                    # Track member for union aliasing
                    self._track_member_for_base(lhs_scope, lhs_loc)
                    
                    # v8.1: Handle call expression on RHS: var = func() or var = fptr_var()
                    rhs_unwrapped = self._unwrap_cast(rhs)
                    if rhs_unwrapped.kind == CursorKind.CALL_EXPR:
                        callee = self._extract_call_callee(rhs_unwrapped)
                        if callee:
                            if callee in self.all_functions:
                                # Direct function call
                                self.data_flow[key].add(f"@ret:{callee}")
                                if self.verbose:
                                    print(f"    {C.DIM}📌 {key} = @ret:{callee}{C.RESET}")
                            else:
                                # Indirect call through variable/member
                                self.data_flow[key].add(f"@fptr_ret:{scope}:{callee}")
                                if self.verbose:
                                    print(f"    {C.DIM}📌 {key} = @fptr_ret:{scope}:{callee}{C.RESET}")
                    else:
                        # Extract direct function refs
                        funcs = self._extract_all_functions_deep(rhs)
                        if funcs:
                            self.data_flow[key].update(funcs)
                            if self.verbose:
                                print(f"    {C.DIM}📌 {key} = {funcs}{C.RESET}")
                            # v8.1: Track member values for linked list collection
                            self._track_member_value(lhs_scope, lhs_loc, funcs)
                    
                    # Variable reference chain
                    rhs_loc = self._extract_location(rhs)
                    if rhs_loc and rhs_loc not in self.all_functions:
                        if rhs_unwrapped.kind != CursorKind.CALL_EXPR:
                            self.data_flow[key].add(f"@ref:{rhs_loc}")
                            if self.verbose:
                                print(f"    {C.DIM}📌 {key} -> {rhs_loc} (chain){C.RESET}")
                            
                            # v8.2: Detect chain variable pattern: var = var->next or var = something->next
                            if rhs_loc.endswith('.next') or rhs_loc.endswith('->next'):
                                # This variable is being iterated through a linked list
                                self._track_chain_variable(scope, lhs_loc, f"@ref:{rhs_loc}")
                            
                            # v8.2: Track .next assignments for linked list following
                            if lhs_loc.endswith('.next'):
                                next_key = self._make_key(lhs_scope, lhs_loc)
                                self.next_chains[next_key].add(f"@ref:{rhs_loc}")
        
        # Return statement
        if kind == CursorKind.RETURN_STMT:
            children = list(cursor.get_children())
            if children:
                ret_expr = children[0]
                ret_unwrapped = self._unwrap_cast(ret_expr)
                
                # v8.1: If returning a function call result, track @ret: marker
                if ret_unwrapped.kind == CursorKind.CALL_EXPR:
                    callee = self._extract_call_callee(ret_unwrapped)
                    if callee and callee in self.all_functions:
                        self.return_values[scope].add(f"@ret:{callee}")
                        if self.verbose:
                            print(f"    {C.DIM}📌 {scope} returns @ret:{callee}{C.RESET}")
                else:
                    funcs = self._extract_all_functions_deep(ret_expr)
                    if funcs:
                        self.return_values[scope].update(funcs)
                        if self.verbose:
                            print(f"    {C.DIM}📌 {scope} returns {funcs}{C.RESET}")
                
                loc = self._extract_location(ret_expr)
                if loc and loc not in self.all_functions:
                    if ret_unwrapped.kind != CursorKind.CALL_EXPR:
                        self.return_values[scope].add(f"@ref:{scope}:{loc}")
        
        # Function call - track callback arguments
        if kind == CursorKind.CALL_EXPR:
            callee = cursor.spelling
            children = list(cursor.get_children())
            
            for i, arg in enumerate(children[1:] if children else []):
                funcs = self._extract_all_functions_deep(arg)
                if funcs:
                    param_name = self.param_names.get((callee, i))
                    if param_name:
                        key = self._make_key(callee, param_name)
                        self.data_flow[key].update(funcs)
                        if self.verbose:
                            print(f"    {C.DIM}📌 {key} (arg) = {funcs}{C.RESET}")
        
        for child in cursor.get_children():
            self._pass2_track_data_flow(child, file_path, scope)
    
    # ==================== RESOLUTION ====================
    
    def _resolve(self, location: str, scope: str, visited: Set[str] = None, depth: int = 0) -> Set[str]:
        """
        Resolve a location to all possible function names.
        v8.1: Enhanced with @ret: handling for function return values.
        v8.3: Better cycle detection for self-referential patterns.
        """
        if visited is None:
            visited = set()
        
        if depth > 50:
            return set()
        
        # v8.3: Safeguard against unbounded location growth
        if len(location) > 500 or location.count('.') > 10:
            # Extract final member and use global lookup
            if '.' in location:
                final_member = location.rsplit('.', 1)[1]
                if '[' in final_member and not final_member.endswith('[*]'):
                    final_member = final_member.rsplit('[', 1)[0] + '[*]'
                if final_member in self.global_member_values:
                    return self.global_member_values[final_member].copy()
            return set()
        
        cache_key = f"{scope}:{location}"
        if cache_key in visited:
            return set()
        visited.add(cache_key)
        
        # v8.3: Detect self-referential member access cycles
        # Pattern: X.a.b.c... where X is repeatedly extended with its own members
        # This happens with linked list patterns like: current = current->next
        if '.' in location:
            parts = location.split('.')
            if len(parts) > 3:
                # Check for repeated member patterns indicating a cycle
                base = parts[0]
                seen_members = set()
                for part in parts[1:]:
                    if part in seen_members:
                        # Cycle detected - use global_member_values for final member
                        final_member = parts[-1]
                        if '[' in final_member and not final_member.endswith('[*]'):
                            final_member = final_member.rsplit('[', 1)[0] + '[*]'
                        if final_member in self.global_member_values:
                            return self.global_member_values[final_member].copy()
                        return set()
                    seen_members.add(part)
        
        results = set()
        
        # Direct function name
        if location in self.all_functions:
            results.add(location)
            return results
        
        # Handle @ref: prefix (variable reference)
        if location.startswith("@ref:"):
            ref_loc = location[5:]
            if ':' in ref_loc:
                ref_scope, ref_name = ref_loc.split(':', 1)
                results.update(self._resolve(ref_name, ref_scope, visited, depth + 1))
            else:
                results.update(self._resolve(ref_loc, scope, visited, depth + 1))
            return results
        
        # v8.1: Handle @ret: prefix - function return value
        if location.startswith("@ret:"):
            func_name = location[5:]
            results.update(self._resolve_return_value(func_name, set(), depth + 1))
            return results
        
        # v8.1: Handle @fptr_ret: prefix - indirect call through function pointer variable
        # Format: @fptr_ret:scope:varname - resolve varname to get function, then get its return value
        if location.startswith("@fptr_ret:"):
            rest = location[10:]  # scope:varname
            if ':' in rest:
                fptr_scope, var_name = rest.split(':', 1)
                # First resolve what the variable points to (the function)
                pointed_funcs = self._resolve(var_name, fptr_scope, visited.copy(), depth + 1)
                # Then get what those functions return
                for func in pointed_funcs:
                    if func in self.all_functions:
                        ret_vals = self._resolve_return_value(func, set(), depth + 1)
                        results.update(ret_vals)
            return results
        
        # Handle @call_ret: prefix - nested call pattern: var = get_factory()(args)
        if location.startswith("@call_ret:"):
            func_name = location[10:]
            # First get what func_name returns
            returned_funcs = self._resolve_return_value(func_name)
            # Then get what THOSE functions return (double chain)
            for returned_func in returned_funcs:
                deeper = self._resolve_return_value(returned_func)
                results.update(deeper)
                # Also include intermediate if it's a function
                if returned_func in self.all_functions:
                    results.add(returned_func)
            return results
        
        # Try scoped lookup
        key = self._make_key(scope, location)
        if key in self.data_flow:
            for value in self.data_flow[key]:
                if value in self.all_functions:
                    results.add(value)
                else:
                    results.update(self._resolve(value, scope, visited, depth + 1))
        
        # Try global lookup
        global_key = self._make_key("global", location)
        if global_key in self.data_flow and global_key != key:
            for value in self.data_flow[global_key]:
                if value in self.all_functions:
                    results.add(value)
                else:
                    results.update(self._resolve(value, "global", visited, depth + 1))
        
        # Handle wildcard array
        if '[' in location and not location.endswith('[*]'):
            wildcard = location.rsplit('[', 1)[0] + '[*]'
            results.update(self._resolve(wildcard, scope, visited, depth + 1))
        
        # Handle member with array index
        if '.' in location:
            parts = location.rsplit('.', 1)
            if len(parts) == 2 and '[' in parts[1] and not parts[1].endswith('[*]'):
                wildcard = f"{parts[0]}.{parts[1].rsplit('[', 1)[0]}[*]"
                results.update(self._resolve(wildcard, scope, visited, depth + 1))
        
        # Handle pointer dereference
        if location.startswith('*'):
            inner = location[1:]
            results.update(self._resolve(inner, scope, visited, depth + 1))
            results.update(self._resolve(f"{inner}[*]", scope, visited, depth + 1))
        
        # Handle address-of
        if location.startswith('&'):
            inner = location[1:]
            results.update(self._resolve(inner, scope, visited, depth + 1))
        
        # Handle struct copy propagation
        if '.' in location and not results:
            parts = location.split('.', 1)
            base, member = parts[0], parts[1]
            
            base_key = self._make_key(scope, base)
            if base_key in self.data_flow:
                for ref in self.data_flow[base_key]:
                    if ref.startswith('@ref:'):
                        ref_var = ref[5:]
                        if ':' in ref_var:
                            ref_scope, ref_name = ref_var.split(':', 1)
                            results.update(self._resolve(f"{ref_name}.{member}", ref_scope, visited, depth + 1))
                        else:
                            results.update(self._resolve(f"{ref_var}.{member}", scope, visited, depth + 1))
        
        # UNION ALIASING - if looking for base.memberX, also check base.memberY
        if '.' in location and not results:
            parts = location.split('.', 1)
            base, member = parts[0], parts[1]
            base_key = self._make_key(scope, base)
            
            if base_key in self.base_members:
                for other_member in self.base_members[base_key]:
                    if other_member != member:
                        other_loc = f"{base}.{other_member}"
                        other_key = self._make_key(scope, other_loc)
                        if other_key in self.data_flow:
                            for value in self.data_flow[other_key]:
                                if value in self.all_functions:
                                    results.add(value)
        
        # v8.2: LINKED LIST COLLECTION - Enhanced to follow chains across scopes
        # If looking for var.member where var is a chain variable (updated via .next pattern),
        # collect all .member values from the GLOBAL tracking
        if '.' in location and not results:
            parts = location.rsplit('.', 1)
            base_var = parts[0]
            member_suffix = parts[1]
            
            # Check if this is a chain variable (updated via .next pattern)
            base_key = self._make_key(scope, base_var)
            is_chain_var = base_key in self.chain_variables
            
            # Also check if base_var references something updated via .next
            if not is_chain_var and base_key in self.data_flow:
                for ref in self.data_flow[base_key]:
                    if ref.startswith('@ref:'):
                        ref_loc = ref[5:]
                        if ref_loc.endswith('.next') or '.next' in ref_loc:
                            is_chain_var = True
                            break
            
            if is_chain_var:
                # Use GLOBAL member values - collect from all scopes
                if member_suffix in self.global_member_values:
                    results.update(self.global_member_values[member_suffix])
            else:
                # Original v8.1 behavior - check scoped member values
                member_key = self._make_key(scope, f"@member:{member_suffix}")
                if member_key in self.scope_member_values:
                    results.update(self.scope_member_values[member_key])
        
        # v8.2: Also try following the linked list explicitly
        if '.' in location and not results:
            parts = location.rsplit('.', 1)
            base_var = parts[0]
            member_suffix = parts[1]
            
            # Try to follow the data flow from base_var and collect all .member values
            base_key = self._make_key(scope, base_var)
            results.update(self._follow_linked_list_members(base_key, member_suffix, set(), depth + 1))
        
        # v8.2: FINAL FALLBACK - For deeply nested member access (a.b.c.d[*]),
        # try looking up just the final member suffix in global_member_values
        # This handles cases like chain->middleware->pre_handlers[i] where
        # the intermediate path is through function parameters
        if not results and '.' in location:
            # Extract the final member suffix (handle array subscripts)
            final_part = location.rsplit('.', 1)[1]
            # Normalize array access: pre_handlers[i] -> pre_handlers[*]
            if '[' in final_part and not final_part.endswith('[*]'):
                final_part = final_part.rsplit('[', 1)[0] + '[*]'
            
            if final_part in self.global_member_values:
                results.update(self.global_member_values[final_part])
        
        return results
    
    def _follow_linked_list_members(self, start_key: str, member_name: str, visited: Set[str], depth: int = 0) -> Set[str]:
        """
        v8.2: Follow a linked list chain collecting all .member_name values.
        Follows @ref: chains through .next assignments.
        """
        if depth > 50 or start_key in visited:
            return set()
        visited.add(start_key)
        
        results = set()
        
        # Extract scope and base location from key
        if ':' not in start_key:
            return results
        
        scope, base_loc = start_key.split(':', 1)
        
        # Get the .member_name value at this node
        member_loc = f"{base_loc}.{member_name}"
        member_key = self._make_key(scope, member_loc)
        
        if member_key in self.data_flow:
            for value in self.data_flow[member_key]:
                if value in self.all_functions:
                    results.add(value)
                elif not value.startswith('@'):
                    # Might be another function reference
                    resolved = self._resolve(value, scope, set(), depth + 1)
                    results.update(resolved)
        
        # Follow the .next chain
        next_loc = f"{base_loc}.next"
        next_key = self._make_key(scope, next_loc)
        
        if next_key in self.data_flow:
            for next_ref in self.data_flow[next_key]:
                if next_ref.startswith('@ref:'):
                    ref_target = next_ref[5:]
                    # ref_target might be like "&h2" or "h2"
                    ref_target = ref_target.lstrip('&')
                    
                    # Determine scope for the ref target
                    if ':' in ref_target:
                        ref_scope, ref_name = ref_target.split(':', 1)
                    else:
                        ref_scope = scope
                        ref_name = ref_target
                    
                    # Recursively follow this chain
                    next_node_key = self._make_key(ref_scope, ref_name)
                    results.update(self._follow_linked_list_members(next_node_key, member_name, visited, depth + 1))
        
        return results
    
    def _resolve_return_value(self, func_name: str, visited: Set[str] = None, depth: int = 0) -> Set[str]:
        """Resolve what functions a function can return."""
        if visited is None:
            visited = set()
        
        if func_name in visited or depth > 20:
            return set()
        visited.add(func_name)
        
        results = set()
        
        for ret_val in self.return_values.get(func_name, []):
            if ret_val in self.all_functions:
                results.add(ret_val)
            elif ret_val.startswith("@ret:"):
                # v8.1: Chain return values: func1 returns @ret:func2 means func1 returns what func2 returns
                inner_func = ret_val[5:]
                results.update(self._resolve_return_value(inner_func, visited, depth + 1))
            elif ret_val.startswith("@ref:"):
                ref = ret_val[5:]
                if ':' in ref:
                    ref_scope, ref_loc = ref.split(':', 1)
                    results.update(self._resolve(ref_loc, ref_scope))
                else:
                    results.update(self._resolve(ref, func_name))
            else:
                results.update(self._resolve(ret_val, func_name))
        
        return results
    
    def _resolve_call_target(self, cursor, scope: str) -> Tuple[Set[str], Optional[str]]:
        """Resolve what functions a call expression could be calling."""
        results = set()
        indirect_via = None
        kind = cursor.kind
        
        # Direct function reference
        if kind == CursorKind.DECL_REF_EXPR:
            name = cursor.spelling
            if name in self.all_functions:
                results.add(name)
            else:
                resolved = self._resolve(name, scope)
                if resolved:
                    results.update(resolved)
                    indirect_via = name
            return results, indirect_via
        
        # Member reference
        if kind == CursorKind.MEMBER_REF_EXPR:
            loc = self._extract_location(cursor)
            if loc:
                resolved = self._resolve(loc, scope)
                if resolved:
                    results.update(resolved)
                    indirect_via = loc
            return results, indirect_via
        
        # Array subscript
        if kind == CursorKind.ARRAY_SUBSCRIPT_EXPR:
            loc = self._extract_location(cursor)
            if loc:
                resolved = self._resolve(loc, scope)
                if resolved:
                    results.update(resolved)
                    indirect_via = loc
            return results, indirect_via
        
        # Unary operator
        if kind == CursorKind.UNARY_OPERATOR:
            loc = self._extract_location(cursor)
            if loc:
                resolved = self._resolve(loc, scope)
                if resolved:
                    results.update(resolved)
                    indirect_via = loc
            return results, indirect_via
        
        # Call expression (function returning fptr)
        if kind == CursorKind.CALL_EXPR:
            callee = self._extract_call_callee(cursor)
            if callee:
                ret_funcs = self._resolve_return_value(callee)
                if ret_funcs:
                    results.update(ret_funcs)
                    indirect_via = f"{callee}()"
            return results, indirect_via
        
        # Cast expression - look through
        if kind == CursorKind.CSTYLE_CAST_EXPR:
            children = list(cursor.get_children())
            for child in children:
                if child.kind != CursorKind.TYPE_REF:
                    inner_results, inner_via = self._resolve_call_target(child, scope)
                    if inner_results:
                        results.update(inner_results)
                        indirect_via = inner_via
                    return results, indirect_via
        
        # Parenthesized or unexposed
        if kind in (CursorKind.PAREN_EXPR, CursorKind.UNEXPOSED_EXPR):
            children = list(cursor.get_children())
            if children:
                return self._resolve_call_target(children[0], scope)
        
        return results, indirect_via
    
    # ==================== PASS 3: CALL EXTRACTION ====================
    
    def _pass3_extract_calls(self, cursor, file_path: str) -> None:
        for child in cursor.get_children():
            if not child.location.file:
                continue
            if child.location.file.name != file_path:
                continue
            
            if child.kind == CursorKind.FUNCTION_DECL and child.is_definition():
                func_name = child.spelling
                
                if func_name == "main":
                    self.main_file = file_path
                
                calls = set()
                self._extract_calls_recursive(child, func_name, calls, file_path)
                
                self.functions[func_name] = FunctionInfo(
                    name=func_name,
                    file_path=file_path,
                    line=child.location.line,
                    calls=calls
                )
    
    def _extract_calls_recursive(self, cursor, caller: str, calls: Set[str], source_file: str) -> None:
        if cursor.kind == CursorKind.CALL_EXPR:
            line_num = cursor.location.line
            file_path = cursor.location.file.name if cursor.location.file else source_file
            
            children = list(cursor.get_children())
            callee_name = cursor.spelling
            
            macro_name = None
            macro_file = None
            tokens = [t.spelling for t in cursor.get_tokens() if t.kind == TokenKind.IDENTIFIER]
            
            if tokens and callee_name and tokens[0] != callee_name:
                if tokens[0] in self.macros:
                    macro_name = tokens[0]
                    macro_file = self.macros[tokens[0]].file_path
            
            resolved_funcs = set()
            indirect_via = None
            
            if children:
                target = children[0]
                target = self._unwrap_cast(target)
                resolved_funcs, indirect_via = self._resolve_call_target(target, caller)
            
            if not resolved_funcs and callee_name:
                if callee_name in self.all_functions:
                    resolved_funcs.add(callee_name)
                else:
                    resolved = self._resolve(callee_name, caller)
                    if resolved:
                        resolved_funcs.update(resolved)
                        indirect_via = callee_name
            
            for func in resolved_funcs:
                calls.add(func)
                self.call_sites.append(CallSite(
                    caller=caller,
                    callee=func,
                    file_path=file_path,
                    line=line_num,
                    macro_name=macro_name,
                    macro_file=macro_file,
                    indirect_via=indirect_via if func != callee_name else None
                ))
            
            if callee_name:
                cb_indices = self._get_callback_indices(callee_name)
                if cb_indices and children:
                    for idx in cb_indices:
                        if idx + 1 < len(children):
                            cb_funcs = self._extract_all_functions_deep(children[idx + 1])
                            for cb_func in cb_funcs:
                                if cb_func in self.all_functions:
                                    calls.add(cb_func)
                                    self.call_sites.append(CallSite(
                                        caller=caller,
                                        callee=cb_func,
                                        file_path=file_path,
                                        line=line_num,
                                        indirect_via="callback"
                                    ))
        
        for child in cursor.get_children():
            self._extract_calls_recursive(child, caller, calls, source_file)
    
    # ==================== PATH FINDING ====================
    
    def find_paths(self, target: str) -> List[CallPath]:
        if "main" not in self.functions:
            return []
        
        paths = []
        queue = [["main"]]
        visited = set()
        
        while queue:
            path = queue.pop(0)
            current = path[-1]
            
            if current == target:
                sites = []
                for i in range(len(path) - 1):
                    for site in self.call_sites:
                        if site.caller == path[i] and site.callee == path[i+1]:
                            sites.append(site)
                            break
                paths.append(CallPath(path=path, call_sites=sites))
                continue
            
            for callee in self.call_graph.get(current, []):
                if callee not in path:
                    new_path = path + [callee]
                    key = tuple(new_path)
                    if key not in visited:
                        visited.add(key)
                        queue.append(new_path)
        
        return paths
    
    def get_unreachable_calls(self, target: str) -> List[CallSite]:
        reachable = set()
        queue = ["main"]
        while queue:
            func = queue.pop(0)
            if func not in reachable:
                reachable.add(func)
                for callee in self.call_graph.get(func, []):
                    if callee in self.functions:
                        queue.append(callee)
        
        return [s for s in self.call_sites 
                if s.callee == target and s.caller not in reachable]


# ==================== OUTPUT ====================

def analyze(project_dir: str, target: str = "mpf_mfs_open",
            include_paths: List[str] = None, verbose: bool = True):
    if verbose:
        print(f"\n{C.BRIGHT_BLUE}{'═'*60}{C.RESET}")
        print(f"{C.BOLD}{C.BRIGHT_WHITE}🔬 Analyzing:{C.RESET} {C.CYAN}{project_dir}{C.RESET}")
        print(f"{C.BRIGHT_BLUE}{'═'*60}{C.RESET}")
    
    all_includes = [project_dir] + (include_paths or [])
    
    parser = CParser(include_paths=all_includes, verbose=verbose)
    parser.parse_directory(project_dir)
    
    paths = parser.find_paths(target)
    unreachable = parser.get_unreachable_calls(target)
    has_target = any(s.callee == target for s in parser.call_sites)
    
    if verbose:
        _print_results(parser, paths, unreachable, target, has_target)
    
    return {
        "paths": paths,
        "unreachable": unreachable,
        "has_main": "main" in parser.functions,
        "has_target": has_target,
        "main_file": parser.main_file,
        "parser": parser
    }


def _print_results(parser, paths, unreachable, target, has_target):
    print(f"\n{C.BRIGHT_CYAN}{'━'*60}{C.RESET}")
    print(f"  {C.BOLD}{C.BRIGHT_WHITE}🔍 CALL GRAPH{C.RESET}")
    print(f"  {C.DIM}Target:{C.RESET} {C.GREEN}{target}(){C.RESET}")
    print(f"{C.BRIGHT_CYAN}{'━'*60}{C.RESET}")
    
    if paths:
        print(f"\n  {C.BRIGHT_GREEN}✅ Found {len(paths)} path(s){C.RESET}\n")
        
        for i, path in enumerate(paths, 1):
            parts = []
            for p in path.path:
                if p == target:
                    parts.append(f"{C.BRIGHT_GREEN}{p}{C.RESET}")
                elif p == "main":
                    parts.append(f"{C.YELLOW}{p}{C.RESET}")
                else:
                    parts.append(f"{C.CYAN}{p}{C.RESET}")
            print(f"  {C.BOLD}{C.WHITE}PATH #{i}:{C.RESET} {f' {C.BRIGHT_MAGENTA}→{C.RESET} '.join(parts)}\n")
            
            for j, func in enumerate(path.path):
                func_info = parser.functions.get(func)
                call_site = path.call_sites[j-1] if j > 0 and j-1 < len(path.call_sites) else None
                
                if func_info:
                    file_name = os.path.basename(func_info.file_path)
                    line = func_info.line
                elif call_site:
                    file_name = os.path.basename(call_site.file_path)
                    line = call_site.line
                else:
                    file_name, line = "?", "?"
                
                if func == target:
                    color, icon = C.BRIGHT_GREEN, "🎯"
                elif func == "main":
                    color, icon = C.YELLOW, "🚀"
                else:
                    color, icon = C.CYAN, "📦"
                
                prefix = "  " if j == 0 else "  " + "    " * (j-1) + f"{C.DIM}└── {C.RESET}"
                print(f"{prefix}{icon} {color}{C.BOLD}{func}(){C.RESET} {C.DIM}[{file_name}:{line}]{C.RESET}")
                
                if call_site and call_site.indirect_via:
                    ptr_prefix = "  " + "    " * j + f"{C.DIM}    {C.RESET}"
                    print(f"{ptr_prefix}{C.BRIGHT_BLUE}↳ via:{C.RESET} {C.BRIGHT_CYAN}{call_site.indirect_via}{C.RESET}")
                
                if call_site and call_site.macro_name:
                    macro_prefix = "  " + "    " * j + f"{C.DIM}    {C.RESET}"
                    macro_file = os.path.basename(call_site.macro_file) if call_site.macro_file else "?"
                    print(f"{macro_prefix}{C.BRIGHT_MAGENTA}↳ macro:{C.RESET} {C.BRIGHT_YELLOW}{call_site.macro_name}{C.RESET} {C.DIM}[{macro_file}]{C.RESET}")
            
            print()
    
    elif has_target:
        print(f"\n  {C.BRIGHT_YELLOW}⚠️  {target}() exists but NOT reachable from main(){C.RESET}\n")
    else:
        print(f"\n  {C.BRIGHT_BLUE}ℹ️  No {target}() calls found{C.RESET}\n")
    
    if unreachable:
        print(f"  {C.BRIGHT_RED}⚠️  UNREACHABLE:{C.RESET}")
        for site in unreachable:
            print(f"    {C.RED}• {site.caller}(){C.RESET} {C.DIM}[{os.path.basename(site.file_path)}:{site.line}]{C.RESET}")
        print()
    
    print(f"{C.DIM}{'─'*60}{C.RESET}")


def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="C Call Graph Parser v8.1")
    ap.add_argument("project_dir", nargs="?", help="Project directory")
    ap.add_argument("--all", action="store_true", help="Run all test cases")
    ap.add_argument("-I", "--include", action="append", default=[], help="Include paths")
    ap.add_argument("-t", "--target", default="mpf_mfs_open", help="Target function")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    
    args = ap.parse_args()
    
    print(f"""
{C.BRIGHT_MAGENTA}╔═══════════════════════════════════════════════════════════════╗{C.RESET}
{C.BRIGHT_MAGENTA}║{C.RESET}  {C.BOLD}{C.BRIGHT_WHITE}🔬 C Call Graph Parser v8.1{C.RESET}                               {C.BRIGHT_MAGENTA}║{C.RESET}
{C.BRIGHT_MAGENTA}║{C.RESET}  {C.CYAN}Nightmare Edition - Return Value Tracking{C.RESET}                {C.BRIGHT_MAGENTA}║{C.RESET}
{C.BRIGHT_MAGENTA}╚═══════════════════════════════════════════════════════════════╝{C.RESET}
    """)
    
    script_dir = Path(__file__).parent
    test_dir = script_dir.parent / "test_cases"
    include_dir = test_dir / "include"
    
    includes = args.include.copy()
    if include_dir.exists():
        includes.append(str(include_dir))
    
    if args.all:
        cases = sorted([d for d in os.listdir(test_dir) 
                       if os.path.isdir(test_dir/d) and d.startswith("apl")])
        
        results = []
        for case in cases:
            result = analyze(str(test_dir/case), args.target, includes, not args.quiet)
            status = f"✅ {len(result['paths'])} path(s)" if result['paths'] else \
                     "⚠️  Unreachable" if result['has_target'] else "ℹ️  No calls"
            results.append((case, status, result))
        
        print(f"\n{'='*80}")
        print(f"📊 SUMMARY")
        print(f"{'='*80}\n")
        print(f"{'Case':<25} {'Status':<18} Paths")
        print("-"*80)
        for case, status, result in results:
            paths_str = "; ".join(str(p) for p in result['paths'][:2])
            if len(result['paths']) > 2:
                paths_str += f" (+{len(result['paths'])-2})"
            print(f"{case:<25} {status:<18} {paths_str[:45]}")
        print("="*80)
    
    elif args.project_dir:
        analyze(args.project_dir, args.target, includes, not args.quiet)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()

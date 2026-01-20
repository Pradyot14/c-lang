"""
Comprehensive Data Flow Tree Visualizer v2.0
Generates detailed visual representations with:
- All macro definitions
- Function call graph
- Complete data flow tracing
- Source code snippets
- Step-by-step resolution
"""
import os
import re
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum


class NodeType(Enum):
    """Types of nodes in the data flow tree."""
    ROOT = "root"
    FUNCTION_CALL = "function_call"
    VARIABLE = "variable"
    MACRO = "macro"
    LITERAL = "literal"
    EXPRESSION = "expression"
    CONDITION = "condition"
    ASSIGNMENT = "assignment"
    FILE_REF = "file_ref"
    RESOLVED = "resolved"


@dataclass
class TreeNode:
    """A node in the data flow tree."""
    id: str
    label: str
    node_type: NodeType
    value: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    children: List['TreeNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'TreeNode'):
        self.children.append(child)
        return child


class ComprehensiveAnalyzer:
    """Analyzes project and extracts all details for visualization."""
    
    def __init__(self, project_dir: str, include_paths: List[str] = None):
        self.project_dir = project_dir
        self.include_paths = include_paths or []
        
        # Storage
        self.files: Dict[str, str] = {}  # path -> content
        self.macros: Dict[str, Dict] = {}  # name -> {value, file, line, expanded}
        self.functions: Dict[str, Dict] = {}  # name -> {body, file, line, params, calls}
        self.call_graph: Dict[str, Set[str]] = {}  # caller -> set of callees
        self.global_vars: Dict[str, Dict] = {}  # name -> {value, file, line}
        
        self._load_project()
    
    def _load_project(self):
        """Load and parse all project files."""
        dirs = [self.project_dir] + [d for d in self.include_paths if os.path.isdir(d)]
        
        for scan_dir in dirs:
            for root, _, files in os.walk(scan_dir):
                for f in files:
                    if f.endswith(('.c', '.h', '.inc')):
                        self._load_file(os.path.join(root, f))
    
    def _load_file(self, filepath: str):
        """Load and parse a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return
        
        self.files[filepath] = content
        self._extract_macros(filepath, content)
        
        if filepath.endswith('.c'):
            self._extract_functions(filepath, content)
            self._extract_globals(filepath, content)
    
    def _extract_macros(self, filepath: str, content: str):
        """Extract all macro definitions with full details."""
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            # Match #define patterns
            patterns = [
                (r'#\s*define\s+(\w+)\s+\(([^)]+)\)', 'parenthesized'),
                (r'#\s*define\s+(\w+)\s+(\d+)', 'numeric'),
                (r'#\s*define\s+(\w+)\s+([A-Z_][A-Z0-9_]*(?:\s*[-+*/]\s*(?:\d+|[A-Z_][A-Z0-9_]*))*)', 'expression'),
                (r'#\s*define\s+(\w+)\s+"([^"]*)"', 'string'),
            ]
            
            for pattern, macro_type in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    name, value = match.groups()
                    if name not in self.macros:
                        self.macros[name] = {
                            'value': value.strip(),
                            'file': os.path.basename(filepath),
                            'file_path': filepath,
                            'line': i + 1,
                            'type': macro_type,
                            'expanded': None,  # Will be computed later
                            'raw_line': line.strip()
                        }
                    break
        
        # Expand all macros
        self._expand_all_macros()
    
    def _expand_all_macros(self):
        """Expand all macros to their final numeric values."""
        for name in self.macros:
            self.macros[name]['expanded'] = self._expand_macro(name, set())
    
    def _expand_macro(self, name: str, visited: Set[str]) -> Optional[int]:
        """Recursively expand a macro to numeric value."""
        if name in visited or name not in self.macros:
            return None
        
        visited.add(name)
        value = self.macros[name]['value'].strip('()')
        
        # Direct number
        try:
            return int(value)
        except:
            pass
        
        # Another macro
        if value in self.macros:
            return self._expand_macro(value, visited)
        
        # Expression
        if re.search(r'[-+*/]', value):
            expr = value
            for macro in re.findall(r'[A-Z_][A-Z0-9_]*', value):
                if macro in self.macros:
                    expanded = self._expand_macro(macro, visited.copy())
                    if expanded is not None:
                        expr = re.sub(r'\b' + macro + r'\b', str(expanded), expr)
            try:
                return int(eval(expr.replace('(', '').replace(')', ''), {"__builtins__": {}}, {}))
            except:
                pass
        
        return None
    
    def _extract_functions(self, filepath: str, content: str):
        """Extract all function definitions with details."""
        func_pattern = r'(?:static\s+)?(?:inline\s+)?(\w+(?:\s*\*)?)\s+(\w+)\s*\(([^)]*)\)\s*\{'
        
        for match in re.finditer(func_pattern, content):
            return_type = match.group(1)
            func_name = match.group(2)
            params_str = match.group(3)
            
            # Find line number
            line_num = content[:match.start()].count('\n') + 1
            
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
            
            body = content[start:end]
            
            # Parse parameters
            params = []
            for p in params_str.split(','):
                p = p.strip()
                if p and p != 'void':
                    parts = re.sub(r'[*]', ' ', p).split()
                    if parts:
                        params.append({
                            'name': parts[-1],
                            'type': ' '.join(parts[:-1]) if len(parts) > 1 else 'int'
                        })
            
            # Extract function calls made by this function
            calls = []
            for m in re.finditer(r'(\w+)\s*\(([^)]*)\)', body):
                called_func = m.group(1)
                args = m.group(2)
                if called_func not in ['if', 'while', 'for', 'switch', 'return', 'sizeof']:
                    calls.append({
                        'function': called_func,
                        'args': [a.strip() for a in args.split(',') if a.strip()],
                        'line': line_num + body[:m.start()].count('\n')
                    })
            
            self.functions[func_name] = {
                'name': func_name,
                'return_type': return_type,
                'params': params,
                'body': body,
                'file': os.path.basename(filepath),
                'file_path': filepath,
                'line': line_num,
                'calls': calls,
                'signature': f"{return_type} {func_name}({params_str})"
            }
            
            # Build call graph
            self.call_graph[func_name] = set()
            for call in calls:
                self.call_graph[func_name].add(call['function'])
    
    def _extract_globals(self, filepath: str, content: str):
        """Extract global variable definitions."""
        lines = content.split('\n')
        in_function = False
        brace_depth = 0
        
        for i, line in enumerate(lines):
            brace_depth += line.count('{') - line.count('}')
            
            # Check if we're outside functions
            if brace_depth == 0:
                match = re.match(r'(?:static\s+)?(?:const\s+)?(\w+)\s+(\w+)\s*=\s*([^;]+);', line.strip())
                if match:
                    var_type, var_name, var_value = match.groups()
                    self.global_vars[var_name] = {
                        'type': var_type,
                        'value': var_value.strip(),
                        'file': os.path.basename(filepath),
                        'line': i + 1
                    }
    
    def get_call_path(self, target_func: str) -> List[str]:
        """Get call path from main to target function."""
        if 'main' not in self.call_graph:
            return [target_func]
        
        # BFS
        queue = [(['main'], 'main')]
        visited = {'main'}
        
        while queue:
            path, current = queue.pop(0)
            
            if current == target_func:
                return path
            
            for called in self.call_graph.get(current, []):
                if called not in visited and called in self.functions:
                    visited.add(called)
                    queue.append((path + [called], called))
        
        return [target_func]
    
    def get_reachable_functions(self) -> Set[str]:
        """Get all functions reachable from main."""
        if 'main' not in self.functions:
            return set(self.functions.keys())
        
        reachable = set()
        queue = ['main']
        
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            
            # Add called functions
            for called in self.call_graph.get(current, []):
                if called not in reachable and called in self.functions:
                    queue.append(called)
            
            # Check for function pointers
            if current in self.functions:
                body = self.functions[current]['body']
                for func_name in self.functions:
                    if func_name != current and func_name in body:
                        if func_name not in reachable:
                            queue.append(func_name)
        
        return reachable


class DataFlowTreeBuilder:
    """Builds comprehensive tree structure."""
    
    def __init__(self):
        self.node_counter = 0
    
    def _next_id(self) -> str:
        self.node_counter += 1
        return f"node_{self.node_counter}"
    
    def build_from_chain(self, resolution_chain: List[str], 
                         raw_arg: str, resolved_value: Optional[int],
                         file_path: str, line_number: int) -> TreeNode:
        """Build a tree from a resolution chain."""
        self.node_counter = 0
        
        root = TreeNode(
            id=self._next_id(),
            label=f"mpf_mfs_open() @ {os.path.basename(file_path)}:{line_number}",
            node_type=NodeType.ROOT,
            file_path=file_path,
            line_number=line_number
        )
        
        arg_node = TreeNode(
            id=self._next_id(),
            label=f"3rd Argument: {raw_arg}",
            node_type=NodeType.VARIABLE if not raw_arg.isdigit() else NodeType.LITERAL,
            value=raw_arg
        )
        root.add_child(arg_node)
        
        current_node = arg_node
        
        for item in resolution_chain:
            item = item.strip()
            if item.startswith("Found:"):
                continue
            
            if "Macro:" in item or "→" in item:
                child = TreeNode(
                    id=self._next_id(),
                    label=item,
                    node_type=NodeType.MACRO,
                    value=item
                )
                current_node.add_child(child)
                current_node = child
            elif "Variable" in item:
                child = TreeNode(
                    id=self._next_id(),
                    label=item,
                    node_type=NodeType.VARIABLE,
                    value=item
                )
                current_node.add_child(child)
                current_node = child
            elif "=" in item:
                child = TreeNode(
                    id=self._next_id(),
                    label=item,
                    node_type=NodeType.EXPRESSION,
                    value=item
                )
                current_node.add_child(child)
                current_node = child
            else:
                child = TreeNode(
                    id=self._next_id(),
                    label=item,
                    node_type=NodeType.ASSIGNMENT,
                    value=item
                )
                current_node.add_child(child)
                current_node = child
        
        # Add resolved value node
        if resolved_value is not None:
            resolved_node = TreeNode(
                id=self._next_id(),
                label=f"✅ RESOLVED: {resolved_value}",
                node_type=NodeType.RESOLVED,
                value=str(resolved_value)
            )
            current_node.add_child(resolved_node)
        
        return root
    
    def to_dict(self, node: TreeNode) -> Dict:
        """Convert tree to dictionary for JSON."""
        return {
            'id': node.id,
            'label': node.label,
            'type': node.node_type.value,
            'value': node.value,
            'file': node.file_path,
            'line': node.line_number,
            'metadata': node.metadata,
            'children': [self.to_dict(c) for c in node.children]
        }


class TreeVisualizer:
    """Generates visualizations in various formats."""
    
    def to_ascii(self, node: TreeNode, prefix: str = "", is_last: bool = True) -> str:
        """Generate ASCII tree."""
        connector = "└── " if is_last else "├── "
        result = prefix + connector + node.label + "\n"
        
        prefix += "    " if is_last else "│   "
        
        for i, child in enumerate(node.children):
            result += self.to_ascii(child, prefix, i == len(node.children) - 1)
        
        return result
    
    def to_mermaid(self, node: TreeNode, builder: DataFlowTreeBuilder) -> str:
        """Generate Mermaid diagram."""
        lines = ["graph TD"]
        self._add_mermaid_nodes(node, lines, set())
        return "\n".join(lines)
    
    def _add_mermaid_nodes(self, node: TreeNode, lines: List[str], visited: Set[str]):
        if node.id in visited:
            return
        visited.add(node.id)
        
        label = node.label.replace('"', "'").replace('\n', ' ')[:50]
        shape = self._get_mermaid_shape(node.node_type)
        lines.append(f'    {node.id}{shape[0]}"{label}"{shape[1]}')
        
        for child in node.children:
            lines.append(f'    {node.id} --> {child.id}')
            self._add_mermaid_nodes(child, lines, visited)
    
    def _get_mermaid_shape(self, node_type: NodeType) -> tuple:
        shapes = {
            NodeType.ROOT: ('[', ']'),
            NodeType.FUNCTION_CALL: ('([', '])'),
            NodeType.VARIABLE: ('{{', '}}'),
            NodeType.MACRO: ('[[', ']]'),
            NodeType.LITERAL: ('((', '))'),
            NodeType.RESOLVED: ('[/', '/]'),
        }
        return shapes.get(node_type, ('[', ']'))


    def to_html_comprehensive(self, result, analyzer: ComprehensiveAnalyzer) -> str:
        """Generate comprehensive HTML with all project details."""
        
        # Get data
        macros = analyzer.macros
        functions = analyzer.functions
        call_graph = analyzer.call_graph
        files = analyzer.files
        
        # Build call path
        call_path = analyzer.get_call_path(result.containing_function)
        reachable = analyzer.get_reachable_functions()
        
        # Build tree
        builder = DataFlowTreeBuilder()
        tree = builder.build_from_chain(
            result.resolution_chain,
            result.raw_argument,
            result.resolved_value,
            result.file_path,
            result.line_number
        )
        tree_data = json.dumps(builder.to_dict(tree))
        
        # Format macros table
        macros_html = self._generate_macros_table(macros, result.raw_argument)
        
        # Format functions
        functions_html = self._generate_functions_section(functions, call_path, reachable, result.containing_function)
        
        # Format call graph
        call_graph_html = self._generate_call_graph(call_graph, call_path, reachable)
        
        # Format source files
        files_html = self._generate_files_section(files, result)
        
        # Format resolution steps
        resolution_html = self._generate_resolution_steps(result)
        
        # Generate complete AST dump
        ast_html = self._generate_ast_section(analyzer.project_dir, result)
        
        title = f"Data Flow Analysis: {result.raw_argument} → {result.resolved_value}"
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #1a1a2e;
            color: #eee;
            line-height: 1.6;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .header h1 {{ font-size: 2em; margin-bottom: 10px; }}
        .header .subtitle {{ opacity: 0.9; font-size: 1.1em; }}
        .result-badge {{
            display: inline-block;
            background: #00c853;
            color: white;
            padding: 10px 30px;
            border-radius: 30px;
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 15px;
            box-shadow: 0 4px 15px rgba(0,200,83,0.4);
        }}
        .container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
        .grid-full {{ grid-column: 1 / -1; }}
        .card {{
            background: #16213e;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}
        .card-header {{
            background: linear-gradient(135deg, #0f3460, #1a1a2e);
            padding: 15px 20px;
            border-bottom: 2px solid #667eea;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .card-header h2 {{ font-size: 1.2em; }}
        .card-header .icon {{ font-size: 1.5em; }}
        .card-body {{ padding: 20px; max-height: 500px; overflow-y: auto; }}
        .card-body.tall {{ max-height: 700px; }}
        
        /* Tables */
        table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #2a2a4a; }}
        th {{ background: #0f3460; color: #667eea; font-weight: 600; position: sticky; top: 0; }}
        tr:hover {{ background: #1e3a5f; }}
        .highlight {{ background: #2e5090 !important; }}
        .highlight-strong {{ background: #4a7c59 !important; border-left: 4px solid #00c853; }}
        
        /* Code */
        pre, code {{
            font-family: 'Fira Code', 'Consolas', monospace;
            background: #0d1b2a;
            border-radius: 8px;
        }}
        pre {{ padding: 15px; overflow-x: auto; font-size: 0.85em; line-height: 1.5; }}
        code {{ padding: 2px 6px; color: #7dd3fc; }}
        .code-line {{ display: block; }}
        .code-line:hover {{ background: #1e3a5f; }}
        .code-line.highlight {{ background: #2e5090; border-left: 3px solid #667eea; padding-left: 12px; }}
        .line-num {{ 
            display: inline-block; 
            width: 40px; 
            color: #666; 
            text-align: right; 
            padding-right: 15px;
            border-right: 1px solid #333;
            margin-right: 15px;
            user-select: none;
        }}
        
        /* Tree */
        .tree ul {{ list-style: none; padding-left: 25px; }}
        .tree > ul {{ padding-left: 0; }}
        .tree li {{ position: relative; padding: 8px 0; }}
        .tree li::before {{
            content: "";
            position: absolute;
            left: -15px;
            top: 0;
            border-left: 2px solid #667eea;
            height: 100%;
        }}
        .tree li::after {{
            content: "";
            position: absolute;
            left: -15px;
            top: 20px;
            border-top: 2px solid #667eea;
            width: 15px;
        }}
        .tree li:last-child::before {{ height: 20px; }}
        .tree > ul > li::before, .tree > ul > li::after {{ display: none; }}
        
        .node {{
            display: inline-flex;
            align-items: center;
            padding: 8px 15px;
            border-radius: 8px;
            font-size: 0.9em;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .node:hover {{ transform: translateX(5px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }}
        .node .icon {{ margin-right: 8px; font-size: 1.1em; }}
        .node.root {{ background: linear-gradient(135deg, #1e3a5f, #0f3460); border: 2px solid #667eea; }}
        .node.function_call {{ background: linear-gradient(135deg, #4a3f35, #3d2914); border: 2px solid #ff9800; }}
        .node.variable {{ background: linear-gradient(135deg, #3d2952, #2d1b40); border: 2px solid #9c27b0; }}
        .node.macro {{ background: linear-gradient(135deg, #1b4332, #0d2818); border: 2px solid #4caf50; }}
        .node.literal {{ background: linear-gradient(135deg, #4a1942, #2d0f28); border: 2px solid #e91e63; }}
        .node.expression {{ background: linear-gradient(135deg, #3d3d14, #2a2a0f); border: 2px solid #ffeb3b; }}
        .node.resolved {{ 
            background: linear-gradient(135deg, #1b5e20, #0d3d12); 
            border: 2px solid #00c853;
            font-weight: bold;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ box-shadow: 0 0 0 0 rgba(0,200,83,0.4); }}
            50% {{ box-shadow: 0 0 0 15px rgba(0,200,83,0); }}
        }}
        
        /* Call Graph */
        .call-graph {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }}
        .call-node {{
            padding: 8px 15px;
            border-radius: 8px;
            background: #0f3460;
            border: 2px solid #333;
        }}
        .call-node.active {{ border-color: #667eea; background: #1e3a5f; }}
        .call-node.target {{ border-color: #00c853; background: #1b5e20; }}
        .call-arrow {{ color: #667eea; font-size: 1.5em; }}
        
        /* Steps */
        .step {{
            display: flex;
            gap: 15px;
            padding: 15px;
            background: #0d1b2a;
            border-radius: 10px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }}
        .step-num {{
            width: 30px;
            height: 30px;
            background: #667eea;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            flex-shrink: 0;
        }}
        .step-content {{ flex: 1; }}
        
        /* Tabs */
        .tabs {{ display: flex; gap: 5px; margin-bottom: 15px; }}
        .tab {{
            padding: 10px 20px;
            background: #0f3460;
            border: none;
            color: #888;
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            transition: all 0.2s;
        }}
        .tab:hover {{ background: #1e3a5f; color: #fff; }}
        .tab.active {{ background: #667eea; color: white; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
        
        /* Legend */
        .legend {{ display: flex; flex-wrap: wrap; gap: 10px; padding: 15px; background: #0d1b2a; border-radius: 10px; margin-top: 15px; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 0.85em; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 4px; }}
        
        /* Scrollbar */
        ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
        ::-webkit-scrollbar-track {{ background: #0d1b2a; }}
        ::-webkit-scrollbar-thumb {{ background: #667eea; border-radius: 4px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: #8b9ff5; }}
        
        /* Collapsible */
        .collapsible {{ cursor: pointer; }}
        .collapsible::before {{ content: "▼ "; font-size: 0.8em; }}
        .collapsible.collapsed::before {{ content: "▶ "; }}
        .collapsible-content {{ transition: max-height 0.3s; overflow: hidden; }}
        .collapsible.collapsed + .collapsible-content {{ max-height: 0 !important; padding: 0 !important; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Data Flow Analysis</h1>
        <div class="subtitle">Tracing the 3rd argument of mpf_mfs_open()</div>
        <div class="result-badge">✅ {result.raw_argument} → {result.resolved_value}</div>
    </div>
    
    <div class="container">
        <!-- Summary -->
        <div class="card" style="margin-bottom: 20px;">
            <div class="card-header">
                <span class="icon">📋</span>
                <h2>Analysis Summary</h2>
            </div>
            <div class="card-body">
                <table>
                    <tr><td><strong>Target File</strong></td><td>{os.path.basename(result.file_path)}</td></tr>
                    <tr><td><strong>Line Number</strong></td><td>{result.line_number}</td></tr>
                    <tr><td><strong>Containing Function</strong></td><td><code>{result.containing_function}()</code></td></tr>
                    <tr><td><strong>Raw Argument</strong></td><td><code>{result.raw_argument}</code></td></tr>
                    <tr><td><strong>Resolved Value</strong></td><td><strong style="color: #00c853; font-size: 1.2em;">{result.resolved_value}</strong></td></tr>
                    <tr><td><strong>Confidence</strong></td><td>{result.confidence:.0%}</td></tr>
                    <tr><td><strong>Analysis Method</strong></td><td>{getattr(result, 'analysis_method', 'Pattern')}</td></tr>
                </table>
            </div>
        </div>
        
        <div class="grid">
            <!-- Resolution Steps -->
            <div class="card">
                <div class="card-header">
                    <span class="icon">📝</span>
                    <h2>Resolution Steps</h2>
                </div>
                <div class="card-body">
                    {resolution_html}
                </div>
            </div>
            
            <!-- Data Flow Tree -->
            <div class="card">
                <div class="card-header">
                    <span class="icon">🌳</span>
                    <h2>Data Flow Tree</h2>
                </div>
                <div class="card-body">
                    <div class="tree" id="tree"></div>
                    <div class="legend">
                        <div class="legend-item"><div class="legend-color" style="background:#1e3a5f;border:2px solid #667eea;"></div>Root</div>
                        <div class="legend-item"><div class="legend-color" style="background:#4a3f35;border:2px solid #ff9800;"></div>Function</div>
                        <div class="legend-item"><div class="legend-color" style="background:#3d2952;border:2px solid #9c27b0;"></div>Variable</div>
                        <div class="legend-item"><div class="legend-color" style="background:#1b4332;border:2px solid #4caf50;"></div>Macro</div>
                        <div class="legend-item"><div class="legend-color" style="background:#4a1942;border:2px solid #e91e63;"></div>Literal</div>
                        <div class="legend-item"><div class="legend-color" style="background:#1b5e20;border:2px solid #00c853;"></div>Resolved</div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Call Graph -->
        <div class="card" style="margin-bottom: 20px;">
            <div class="card-header">
                <span class="icon">🔗</span>
                <h2>Function Call Path (main → target)</h2>
            </div>
            <div class="card-body">
                {call_graph_html}
            </div>
        </div>
        
        <div class="grid">
            <!-- Macros -->
            <div class="card">
                <div class="card-header">
                    <span class="icon">🔧</span>
                    <h2>Macro Definitions ({len(macros)} total)</h2>
                </div>
                <div class="card-body tall">
                    {macros_html}
                </div>
            </div>
            
            <!-- Functions -->
            <div class="card">
                <div class="card-header">
                    <span class="icon">📞</span>
                    <h2>Functions ({len(functions)} total)</h2>
                </div>
                <div class="card-body tall">
                    {functions_html}
                </div>
            </div>
        </div>
        
        <!-- Source Files -->
        <div class="card grid-full" style="margin-top: 20px;">
            <div class="card-header">
                <span class="icon">📄</span>
                <h2>Source Files</h2>
            </div>
            <div class="card-body tall">
                {files_html}
            </div>
        </div>
        
        <!-- Complete AST -->
        <div class="card grid-full" style="margin-top: 20px;">
            <div class="card-header">
                <span class="icon">🌲</span>
                <h2>Complete AST (Abstract Syntax Tree)</h2>
            </div>
            <div class="card-body tall">
                {ast_html}
            </div>
        </div>
    </div>
    
    <script>
        const treeData = {tree_data};
        
        const icons = {{
            'root': '🎯',
            'function_call': '📞',
            'variable': '📦',
            'macro': '🔧',
            'literal': '🔢',
            'expression': '🔄',
            'condition': '❓',
            'assignment': '📝',
            'file_ref': '📄',
            'resolved': '✅'
        }};
        
        function renderNode(node) {{
            const icon = icons[node.type] || '•';
            const li = document.createElement('li');
            
            const nodeDiv = document.createElement('div');
            nodeDiv.className = `node ${{node.type}}`;
            nodeDiv.innerHTML = `<span class="icon">${{icon}}</span>${{node.label}}`;
            nodeDiv.onclick = (e) => {{
                e.stopPropagation();
                alert(`Type: ${{node.type}}\\nLabel: ${{node.label}}\\nValue: ${{node.value || 'N/A'}}`);
            }};
            
            li.appendChild(nodeDiv);
            
            if (node.children && node.children.length > 0) {{
                const ul = document.createElement('ul');
                node.children.forEach(child => {{
                    ul.appendChild(renderNode(child));
                }});
                li.appendChild(ul);
            }}
            
            return li;
        }}
        
        const treeContainer = document.getElementById('tree');
        const ul = document.createElement('ul');
        ul.appendChild(renderNode(treeData));
        treeContainer.appendChild(ul);
        
        // Tab functionality
        document.querySelectorAll('.tab').forEach(tab => {{
            tab.onclick = () => {{
                const tabGroup = tab.parentElement;
                const contentGroup = tabGroup.nextElementSibling;
                
                tabGroup.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                contentGroup.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                contentGroup.querySelector(`#${{tab.dataset.tab}}`).classList.add('active');
            }};
        }});
        
        // Collapsible functionality
        document.querySelectorAll('.collapsible').forEach(el => {{
            el.onclick = () => el.classList.toggle('collapsed');
        }});
    </script>
</body>
</html>'''
        
        return html

    
    def _generate_macros_table(self, macros: Dict, highlight_arg: str) -> str:
        """Generate HTML table for macros."""
        if not macros:
            return "<p>No macros found</p>"
        
        rows = []
        for name, info in sorted(macros.items()):
            highlight_class = ""
            if name == highlight_arg:
                highlight_class = "highlight-strong"
            elif highlight_arg in str(info.get('value', '')):
                highlight_class = "highlight"
            
            expanded = info.get('expanded')
            expanded_str = f"<strong style='color:#00c853;'>{expanded}</strong>" if expanded is not None else "<em style='color:#888;'>N/A</em>"
            
            rows.append(f'''
                <tr class="{highlight_class}">
                    <td><code>{name}</code></td>
                    <td><code>{info.get('value', '?')}</code></td>
                    <td>{expanded_str}</td>
                    <td>{info.get('file', '?')}:{info.get('line', '?')}</td>
                </tr>
            ''')
        
        return f'''
            <table>
                <thead>
                    <tr>
                        <th>Macro Name</th>
                        <th>Definition</th>
                        <th>Expanded Value</th>
                        <th>Location</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        '''
    
    def _generate_functions_section(self, functions: Dict, call_path: List[str], 
                                    reachable: Set[str], target_func: str) -> str:
        """Generate HTML for functions."""
        if not functions:
            return "<p>No functions found</p>"
        
        html_parts = []
        
        # Sort: call path first, then reachable, then others
        sorted_funcs = []
        for f in call_path:
            if f in functions:
                sorted_funcs.append((f, functions[f], 'path'))
        for f in sorted(reachable - set(call_path)):
            if f in functions:
                sorted_funcs.append((f, functions[f], 'reachable'))
        for f in sorted(set(functions.keys()) - reachable):
            sorted_funcs.append((f, functions[f], 'unreachable'))
        
        for func_name, func_info, category in sorted_funcs:
            badge = ""
            if func_name == 'main':
                badge = '<span style="background:#667eea;color:white;padding:2px 8px;border-radius:4px;margin-left:10px;font-size:0.8em;">ENTRY</span>'
            elif func_name == target_func:
                badge = '<span style="background:#00c853;color:white;padding:2px 8px;border-radius:4px;margin-left:10px;font-size:0.8em;">TARGET</span>'
            elif category == 'path':
                badge = '<span style="background:#ff9800;color:white;padding:2px 8px;border-radius:4px;margin-left:10px;font-size:0.8em;">IN PATH</span>'
            elif category == 'unreachable':
                badge = '<span style="background:#666;color:white;padding:2px 8px;border-radius:4px;margin-left:10px;font-size:0.8em;">UNREACHABLE</span>'
            
            # Prepare code with line numbers
            body = func_info.get('body', '')
            lines = body.split('\n')
            start_line = func_info.get('line', 1)
            code_lines = []
            for i, line in enumerate(lines[:50]):  # Limit to 50 lines
                line_num = start_line + i
                escaped = line.replace('<', '&lt;').replace('>', '&gt;')
                code_lines.append(f'<span class="code-line"><span class="line-num">{line_num}</span>{escaped}</span>')
            
            if len(lines) > 50:
                code_lines.append(f'<span class="code-line" style="color:#888;">... ({len(lines) - 50} more lines)</span>')
            
            # Calls made by this function
            calls = func_info.get('calls', [])
            calls_html = ""
            if calls:
                calls_list = ', '.join([f"<code>{c['function']}()</code>" for c in calls[:10]])
                if len(calls) > 10:
                    calls_list += f" ... and {len(calls) - 10} more"
                calls_html = f'<div style="margin-top:10px;color:#888;font-size:0.85em;">Calls: {calls_list}</div>'
            
            html_parts.append(f'''
                <div style="margin-bottom: 15px; border: 1px solid #333; border-radius: 8px; overflow: hidden;">
                    <div class="collapsible" style="padding: 10px 15px; background: #0f3460; display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <strong>{func_info.get('signature', func_name + '()')}</strong>
                            {badge}
                        </div>
                        <span style="color: #888; font-size: 0.85em;">{func_info.get('file', '?')}:{func_info.get('line', '?')}</span>
                    </div>
                    <div class="collapsible-content">
                        <pre>{''.join(code_lines)}</pre>
                        {calls_html}
                    </div>
                </div>
            ''')
        
        return ''.join(html_parts)
    
    def _generate_call_graph(self, call_graph: Dict, call_path: List[str], reachable: Set[str]) -> str:
        """Generate call graph visualization."""
        if not call_path:
            return "<p>No call path found</p>"
        
        nodes = []
        for i, func in enumerate(call_path):
            cls = "call-node"
            if func == 'main':
                cls += " active"
            elif i == len(call_path) - 1:
                cls += " target"
            else:
                cls += " active"
            
            nodes.append(f'<div class="{cls}"><code>{func}()</code></div>')
            if i < len(call_path) - 1:
                nodes.append('<span class="call-arrow">→</span>')
        
        # Also show other reachable functions
        other_reachable = reachable - set(call_path)
        if other_reachable:
            other_html = '<div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #333;">'
            other_html += '<span style="color: #888; margin-right: 10px;">Other reachable functions:</span>'
            for func in sorted(list(other_reachable)[:15]):
                other_html += f'<span class="call-node" style="margin: 3px; font-size: 0.85em;"><code>{func}()</code></span>'
            if len(other_reachable) > 15:
                other_html += f'<span style="color: #888;"> ... and {len(other_reachable) - 15} more</span>'
            other_html += '</div>'
        else:
            other_html = ""
        
        return f'''
            <div class="call-graph">
                {''.join(nodes)}
            </div>
            {other_html}
        '''
    
    def _generate_files_section(self, files: Dict, result) -> str:
        """Generate source files section with tabs."""
        if not files:
            return "<p>No source files found</p>"
        
        tabs = []
        contents = []
        
        # Sort: target file first, then .c files, then .h files
        sorted_files = []
        for path in files:
            if path == result.file_path:
                sorted_files.insert(0, path)
            elif path.endswith('.c'):
                sorted_files.append(path)
        for path in files:
            if path.endswith('.h') and path not in sorted_files:
                sorted_files.append(path)
        
        for i, filepath in enumerate(sorted_files):
            filename = os.path.basename(filepath)
            tab_id = f"file_{i}"
            active = "active" if i == 0 else ""
            
            badge = ""
            if filepath == result.file_path:
                badge = ' <span style="background:#00c853;color:white;padding:1px 6px;border-radius:3px;font-size:0.75em;">TARGET</span>'
            
            tabs.append(f'<button class="tab {active}" data-tab="{tab_id}">{filename}{badge}</button>')
            
            # Format file content with line numbers and highlighting
            content = files[filepath]
            lines = content.split('\n')
            code_lines = []
            
            for line_num, line in enumerate(lines, 1):
                highlight = ""
                if filepath == result.file_path and line_num == result.line_number:
                    highlight = " highlight"
                
                escaped = line.replace('<', '&lt;').replace('>', '&gt;')
                code_lines.append(f'<span class="code-line{highlight}"><span class="line-num">{line_num}</span>{escaped}</span>')
            
            contents.append(f'''
                <div id="{tab_id}" class="tab-content {active}">
                    <pre>{''.join(code_lines)}</pre>
                </div>
            ''')
        
        return f'''
            <div class="tabs">
                {''.join(tabs)}
            </div>
            <div class="tab-contents">
                {''.join(contents)}
            </div>
        '''
    
    def _generate_resolution_steps(self, result) -> str:
        """Generate resolution steps HTML."""
        steps = result.resolution_chain
        if not steps:
            return "<p>No resolution steps available</p>"
        
        html_parts = []
        for i, step in enumerate(steps, 1):
            html_parts.append(f'''
                <div class="step">
                    <div class="step-num">{i}</div>
                    <div class="step-content">{step}</div>
                </div>
            ''')
        
        return ''.join(html_parts)
    
    def _generate_ast_section(self, project_dir: str, result) -> str:
        """Generate complete AST dump section."""
        try:
            from ast_dumper import ASTDumper
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from config import INCLUDE_DIR
            
            dumper = ASTDumper([INCLUDE_DIR])
            
            tabs = []
            contents = []
            
            # Get all C and H files
            all_files = []
            for root, _, files in os.walk(project_dir):
                for f in files:
                    if f.endswith(('.c', '.h')):
                        all_files.append(os.path.join(root, f))
            
            # Also check include directory for header files
            if INCLUDE_DIR and os.path.isdir(INCLUDE_DIR):
                for root, _, files in os.walk(INCLUDE_DIR):
                    for f in files:
                        if f.endswith('.h'):
                            filepath = os.path.join(root, f)
                            if filepath not in all_files:
                                all_files.append(filepath)
            
            # Sort: target file first, then .c files, then .h files
            def sort_key(x):
                if x == result.file_path:
                    return (0, x)
                elif x.endswith('.c'):
                    return (1, x)
                else:
                    return (2, x)
            
            all_files.sort(key=sort_key)
            
            for i, filepath in enumerate(all_files):
                filename = os.path.basename(filepath)
                tab_id = f"ast_{i}"
                active = "active" if i == 0 else ""
                
                badge = ""
                if filepath == result.file_path:
                    badge = ' <span style="background:#00c853;color:white;padding:1px 6px;border-radius:3px;font-size:0.75em;">TARGET</span>'
                
                tabs.append(f'<button class="tab {active}" data-tab="{tab_id}">{filename}{badge}</button>')
                
                # Get AST dump
                ast_dump = dumper.dump_file(filepath)
                
                # Format AST with colors
                ast_lines = []
                for line in ast_dump.split('\n'):
                    escaped = line.replace('<', '&lt;').replace('>', '&gt;')
                    
                    # Color different node types
                    if 'FUNCTION_DECL' in escaped:
                        escaped = f'<span style="color:#ff9800;font-weight:bold;">{escaped}</span>'
                    elif 'CALL_EXPR' in escaped:
                        escaped = f'<span style="color:#4caf50;">{escaped}</span>'
                    elif 'VAR_DECL' in escaped or 'PARM_DECL' in escaped or 'FIELD_DECL' in escaped:
                        escaped = f'<span style="color:#9c27b0;">{escaped}</span>'
                    elif 'INTEGER_LITERAL' in escaped:
                        escaped = f'<span style="color:#e91e63;">{escaped}</span>'
                    elif 'RETURN_STMT' in escaped:
                        escaped = f'<span style="color:#2196f3;">{escaped}</span>'
                    elif 'IF_STMT' in escaped or 'COMPOUND_STMT' in escaped:
                        escaped = f'<span style="color:#ff5722;">{escaped}</span>'
                    elif 'MACRO_DEFINITION' in escaped:
                        escaped = f'<span style="color:#8bc34a;font-weight:bold;">{escaped}</span>'
                    elif 'MACRO' in escaped:
                        escaped = f'<span style="color:#8bc34a;">{escaped}</span>'
                    elif 'INCLUSION_DIRECTIVE' in escaped:
                        escaped = f'<span style="color:#03a9f4;">{escaped}</span>'
                    elif 'TYPEDEF_DECL' in escaped:
                        escaped = f'<span style="color:#ff4081;">{escaped}</span>'
                    elif 'STRUCT_DECL' in escaped:
                        escaped = f'<span style="color:#7c4dff;font-weight:bold;">{escaped}</span>'
                    elif 'DECL_REF_EXPR' in escaped:
                        escaped = f'<span style="color:#00bcd4;">{escaped}</span>'
                    elif 'UNARY_OPERATOR' in escaped or 'BINARY_OPERATOR' in escaped:
                        escaped = f'<span style="color:#ffc107;">{escaped}</span>'
                    
                    # Highlight mpf_mfs_open
                    if 'mpf_mfs_open' in escaped:
                        escaped = escaped.replace('mpf_mfs_open', '<span style="background:#1b5e20;padding:0 4px;border-radius:3px;">mpf_mfs_open</span>')
                    
                    ast_lines.append(escaped)
                
                contents.append(f'''
                    <div id="{tab_id}" class="tab-content {active}">
                        <pre style="font-size:0.85em;line-height:1.4;">{chr(10).join(ast_lines)}</pre>
                    </div>
                ''')
            
            # Add legend
            legend = '''
                <div style="margin-bottom:15px;padding:10px;background:#0d1b2a;border-radius:8px;display:flex;flex-wrap:wrap;gap:15px;font-size:0.85em;">
                    <span><span style="color:#ff9800;">■</span> FUNCTION_DECL</span>
                    <span><span style="color:#4caf50;">■</span> CALL_EXPR</span>
                    <span><span style="color:#9c27b0;">■</span> VAR/PARM/FIELD_DECL</span>
                    <span><span style="color:#e91e63;">■</span> INTEGER_LITERAL</span>
                    <span><span style="color:#2196f3;">■</span> RETURN_STMT</span>
                    <span><span style="color:#ff5722;">■</span> IF_STMT/COMPOUND</span>
                    <span><span style="color:#8bc34a;">■</span> MACRO_DEFINITION</span>
                    <span><span style="color:#03a9f4;">■</span> INCLUDE</span>
                    <span><span style="color:#ff4081;">■</span> TYPEDEF</span>
                    <span><span style="color:#7c4dff;">■</span> STRUCT</span>
                    <span><span style="color:#00bcd4;">■</span> DECL_REF_EXPR</span>
                    <span><span style="color:#ffc107;">■</span> OPERATORS</span>
                </div>
            '''
            
            return f'''
                {legend}
                <div class="tabs">
                    {''.join(tabs)}
                </div>
                <div class="tab-contents">
                    {''.join(contents)}
                </div>
            '''
        except Exception as e:
            return f'<p style="color:#ff5722;">AST generation failed: {e}</p><p>Install libclang for full AST support.</p>'


def generate_tree_for_result(result, output_format: str = "ascii", project_dir: str = None) -> str:
    """Generate tree visualization for a result."""
    builder = DataFlowTreeBuilder()
    tree = builder.build_from_chain(
        result.resolution_chain,
        result.raw_argument,
        result.resolved_value,
        result.file_path,
        result.line_number
    )
    
    visualizer = TreeVisualizer()
    
    if output_format == "ascii":
        return visualizer.to_ascii(tree)
    elif output_format == "mermaid":
        return visualizer.to_mermaid(tree, builder)
    elif output_format == "html":
        # Use comprehensive HTML if project_dir available
        if project_dir:
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from config import INCLUDE_DIR
                analyzer = ComprehensiveAnalyzer(project_dir, [INCLUDE_DIR])
                return visualizer.to_html_comprehensive(result, analyzer)
            except:
                pass
        
        # Fallback: create analyzer from result file path
        try:
            project_dir = os.path.dirname(result.file_path)
            import sys
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from config import INCLUDE_DIR
            analyzer = ComprehensiveAnalyzer(project_dir, [INCLUDE_DIR])
            return visualizer.to_html_comprehensive(result, analyzer)
        except Exception as e:
            # Ultimate fallback: simple HTML
            tree_data = json.dumps(builder.to_dict(tree))
            return f'''<!DOCTYPE html>
<html><head><title>Data Flow Tree</title></head>
<body><pre>{visualizer.to_ascii(tree)}</pre>
<p>Error loading comprehensive view: {e}</p>
</body></html>'''
    elif output_format == "graphviz":
        return visualizer.to_mermaid(tree, builder).replace("graph TD", "digraph G {{\n    rankdir=TD;").replace("-->", "->") + "\n}"
    
    return visualizer.to_ascii(tree)

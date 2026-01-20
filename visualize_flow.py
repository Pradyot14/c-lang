#!/usr/bin/env python3
"""
Data Flow Visualizer for FileNo Extractor
Shows step-by-step how the fileno value is computed
"""
import os
import sys
import re
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from ai_analyzer import AIDataFlowAnalyzer


class DataFlowVisualizer:
    """Visualize data flow in terminal with ASCII art"""
    
    # Box drawing characters
    BOX_H = "─"
    BOX_V = "│"
    BOX_TL = "┌"
    BOX_TR = "┐"
    BOX_BL = "└"
    BOX_BR = "┘"
    BOX_T = "┬"
    BOX_B = "┴"
    BOX_L = "├"
    BOX_R = "┤"
    BOX_X = "┼"
    ARROW_DOWN = "▼"
    ARROW_RIGHT = "►"
    
    # Colors (ANSI)
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RED = "\033[91m"
    DIM = "\033[2m"
    
    def __init__(self, use_color: bool = True):
        self.use_color = use_color
        if not use_color:
            self.RESET = self.BOLD = self.GREEN = self.BLUE = ""
            self.YELLOW = self.CYAN = self.MAGENTA = self.RED = self.DIM = ""
    
    def color(self, text: str, color: str) -> str:
        """Apply color to text"""
        if self.use_color:
            return f"{color}{text}{self.RESET}"
        return text
    
    def draw_box(self, text: str, width: int = 50, style: str = "normal") -> List[str]:
        """Draw a box around text"""
        lines = text.split('\n')
        max_len = max(len(line) for line in lines)
        width = max(width, max_len + 4)
        
        if style == "start":
            color = self.GREEN
        elif style == "end":
            color = self.CYAN
        elif style == "macro":
            color = self.MAGENTA
        elif style == "func":
            color = self.YELLOW
        elif style == "cond":
            color = self.BLUE
        elif style == "assign":
            color = self.DIM
        else:
            color = ""
        
        result = []
        result.append(self.color(f"{self.BOX_TL}{self.BOX_H * (width-2)}{self.BOX_TR}", color))
        for line in lines:
            padded = line.center(width - 4)
            result.append(self.color(f"{self.BOX_V} {padded} {self.BOX_V}", color))
        result.append(self.color(f"{self.BOX_BL}{self.BOX_H * (width-2)}{self.BOX_BR}", color))
        return result
    
    def draw_arrow(self, label: str = "", length: int = 1) -> List[str]:
        """Draw a downward arrow with optional label"""
        result = []
        for _ in range(length):
            if label:
                result.append(f"        {self.BOX_V}  {self.DIM}{label}{self.RESET}")
                label = ""
            else:
                result.append(f"        {self.BOX_V}")
        result.append(f"        {self.ARROW_DOWN}")
        return result
    
    def parse_trace(self, trace: str) -> List[Dict]:
        """Parse the AI reasoning trace into structured steps"""
        steps = []
        
        # Look for STEP patterns
        step_pattern = r'STEP\s*\d+[:\s]*(.+?)(?=STEP\s*\d+|FINAL|RESULT|$)'
        matches = re.findall(step_pattern, trace, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            text = match.strip()
            if not text:
                continue
                
            step = {'text': text, 'type': 'step'}  # Full text, no truncation
            
            # Classify step type
            text_lower = text.lower()
            if 'main()' in text_lower or 'start' in text_lower:
                step['type'] = 'start'
            elif 'macro' in text_lower or '#define' in text_lower:
                step['type'] = 'macro'
            elif 'call' in text_lower or 'function' in text_lower or '()' in text:
                step['type'] = 'func'
            elif 'if' in text_lower or 'condition' in text_lower or 'branch' in text_lower or 'switch' in text_lower:
                step['type'] = 'cond'
            elif '=' in text and ('result' in text_lower or 'fileno' in text_lower or 'value' in text_lower):
                step['type'] = 'assign'
            
            steps.append(step)
        
        # Look for FINAL
        final_match = re.search(r'FINAL[:\s]*(.+?)(?=RESULT|$)', trace, re.DOTALL | re.IGNORECASE)
        if final_match:
            steps.append({'text': final_match.group(1).strip(), 'type': 'end'})
        
        return steps
    
    def visualize(self, result, project_name: str = ""):
        """Generate and print the data flow visualization"""
        
        print()
        print(self.color("=" * 70, self.BOLD))
        print(self.color(f"  DATA FLOW ANALYSIS: {project_name}", self.BOLD + self.CYAN))
        print(self.color("=" * 70, self.BOLD))
        print()
        
        # File info
        print(f"  {self.DIM}File:{self.RESET} {os.path.basename(result.file_path)}")
        print(f"  {self.DIM}Function:{self.RESET} {result.containing_function}()")
        print(f"  {self.DIM}Line:{self.RESET} {result.line_number}")
        print(f"  {self.DIM}Raw Argument:{self.RESET} {result.raw_argument}")
        print()
        
        # Draw the flow diagram
        print(self.color("  DATA FLOW:", self.BOLD))
        print()
        
        # Start node
        for line in self.draw_box("START\nmain()", 40, "start"):
            print(f"    {line}")
        
        # Parse and display trace steps
        if result.reasoning_trace:
            steps = self.parse_trace(result.reasoning_trace)
            
            for i, step in enumerate(steps):
                # Arrow
                for line in self.draw_arrow():
                    print(f"    {line}")
                
                # Step box
                step_text = step['text']
                # Truncate and format
                if len(step_text) > 60:
                    step_text = step_text[:57] + "..."
                
                # Add step number
                step_label = f"Step {i+1}"
                box_text = f"{step_label}\n{step_text}"
                
                for line in self.draw_box(box_text, 50, step['type']):
                    print(f"    {line}")
        else:
            # Use resolution chain if no trace
            for i, chain_step in enumerate(result.resolution_chain):
                for line in self.draw_arrow():
                    print(f"    {line}")
                
                for line in self.draw_box(f"Step {i+1}\n{chain_step[:60]}", 50, "step"):
                    print(f"    {line}")
        
        # Final arrow
        for line in self.draw_arrow():
            print(f"    {line}")
        
        # Result box
        if result.resolved_value is not None:
            result_text = f"RESULT\nfileno = {self.BOLD}{result.resolved_value}{self.RESET}"
            for line in self.draw_box(f"RESULT\nfileno = {result.resolved_value}", 40, "end"):
                print(f"    {line}")
            print()
            print(f"    {self.GREEN}✓ Successfully resolved: {self.BOLD}{result.resolved_value}{self.RESET}")
        else:
            for line in self.draw_box(f"RESULT\nUNDEFINED", 40, "end"):
                print(f"    {line}")
            print()
            print(f"    {self.RED}✗ Could not resolve: {result.error}{self.RESET}")
        
        print()
        print(self.color("=" * 70, self.BOLD))
        print()
    
    def visualize_detailed(self, result, project_name: str = ""):
        """Generate detailed data flow with more information"""
        
        print()
        print(self.color("╔" + "═" * 68 + "╗", self.BOLD + self.CYAN))
        print(self.color("║" + f"  DATA FLOW ANALYSIS: {project_name}".ljust(68) + "║", self.BOLD + self.CYAN))
        print(self.color("╚" + "═" * 68 + "╝", self.BOLD + self.CYAN))
        print()
        
        # Summary section
        print(f"  {self.BOLD}📁 Source:{self.RESET} {os.path.basename(result.file_path)}:{result.line_number}")
        print(f"  {self.BOLD}📍 Function:{self.RESET} {result.containing_function}()")
        print(f"  {self.BOLD}🎯 Target:{self.RESET} 3rd arg of mpf_mfs_open() = `{result.raw_argument}`")
        print(f"  {self.BOLD}📊 Method:{self.RESET} {result.analysis_method}")
        print(f"  {self.BOLD}🎲 Confidence:{self.RESET} {result.confidence:.0%}")
        print()
        
        # Flow diagram header
        print(f"  {self.BOLD}{'─' * 66}{self.RESET}")
        print(f"  {self.BOLD}  EXECUTION FLOW{self.RESET}")
        print(f"  {self.BOLD}{'─' * 66}{self.RESET}")
        print()
        
        # Parse trace for detailed view
        if result.reasoning_trace:
            self._print_detailed_trace(result.reasoning_trace)
        else:
            # Fallback to resolution chain
            print(f"    {self.YELLOW}┌─ Entry Point{self.RESET}")
            print(f"    {self.BOX_V}")
            for i, step in enumerate(result.resolution_chain):
                prefix = "└─" if i == len(result.resolution_chain) - 1 else "├─"
                print(f"    {prefix} {step}")
        
        print()
        
        # Final result
        print(f"  {self.BOLD}{'─' * 66}{self.RESET}")
        if result.resolved_value is not None:
            print(f"  {self.GREEN}{self.BOLD}  ✅ FINAL VALUE: {result.resolved_value}{self.RESET}")
        else:
            print(f"  {self.RED}{self.BOLD}  ❌ UNDEFINED: {result.error}{self.RESET}")
        print(f"  {self.BOLD}{'─' * 66}{self.RESET}")
        print()
    
    def _print_detailed_trace(self, trace: str):
        """Print detailed trace with tree structure"""
        
        # Extract steps
        lines = trace.split('\n')
        in_trace = False
        step_num = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if '```trace' in line.lower():
                in_trace = True
                continue
            if '```' in line and in_trace:
                in_trace = False
                continue
            
            if in_trace or line.upper().startswith('STEP') or line.upper().startswith('FINAL'):
                step_num += 1
                
                # Determine icon based on content
                line_lower = line.lower()
                if 'main' in line_lower:
                    icon = "🚀"
                    color = self.GREEN
                elif 'macro' in line_lower or '#define' in line_lower:
                    icon = "📝"
                    color = self.MAGENTA
                elif 'call' in line_lower or '()' in line:
                    icon = "📞"
                    color = self.YELLOW
                elif 'if' in line_lower or 'condition' in line_lower or 'switch' in line_lower:
                    icon = "🔀"
                    color = self.BLUE
                elif 'return' in line_lower:
                    icon = "↩️ "
                    color = self.CYAN
                elif '=' in line and any(x in line_lower for x in ['result', 'fileno', 'value', 'offset']):
                    icon = "💾"
                    color = self.DIM
                elif 'final' in line_lower:
                    icon = "🎯"
                    color = self.GREEN + self.BOLD
                else:
                    icon = "  "
                    color = ""
                
                # Format the line - NO TRUNCATION, show full text
                # Wrap long lines instead
                max_width = 70
                prefix = f"    {self.BOX_V}"
                print(f"{prefix}")
                
                if len(line) > max_width:
                    # Word wrap
                    words = line.split()
                    current_line = ""
                    first = True
                    for word in words:
                        if len(current_line) + len(word) + 1 > max_width:
                            if first:
                                print(f"    {self.BOX_L}─{icon} {color}{current_line}{self.RESET}")
                                first = False
                            else:
                                print(f"    {self.BOX_V}     {color}{current_line}{self.RESET}")
                            current_line = word
                        else:
                            current_line = current_line + " " + word if current_line else word
                    if current_line:
                        if first:
                            print(f"    {self.BOX_L}─{icon} {color}{current_line}{self.RESET}")
                        else:
                            print(f"    {self.BOX_V}     {color}{current_line}{self.RESET}")
                else:
                    print(f"    {self.BOX_L}─{icon} {color}{line}{self.RESET}")


def analyze_with_visualization(project_path: str, include_paths: List[str] = None, detailed: bool = True):
    """Analyze a project and show data flow visualization"""
    
    if include_paths is None:
        include_paths = []
    
    # Add default include path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_include = os.path.join(base_dir, 'test_cases', 'include')
    if os.path.isdir(default_include) and default_include not in include_paths:
        include_paths.append(default_include)
    
    visualizer = DataFlowVisualizer(use_color=True)
    
    print(f"\n{visualizer.DIM}Analyzing: {project_path}{visualizer.RESET}")
    
    analyzer = AIDataFlowAnalyzer(include_paths=include_paths, verbose=False)
    results = analyzer.analyze_project(project_path)
    
    project_name = os.path.basename(project_path)
    
    for result in results:
        if detailed:
            visualizer.visualize_detailed(result, project_name)
        else:
            visualizer.visualize(result, project_name)
    
    if not results:
        print(f"\n{visualizer.YELLOW}No mpf_mfs_open() calls found in {project_name}{visualizer.RESET}\n")
    
    return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze C code and visualize data flow')
    parser.add_argument('project', nargs='?', help='Path to project directory')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all test cases')
    parser.add_argument('--simple', '-s', action='store_true', help='Use simple visualization')
    parser.add_argument('--include', '-I', action='append', default=[], help='Include paths')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_cases_dir = os.path.join(base_dir, 'test_cases')
    
    if args.all:
        # Analyze all test cases
        for name in sorted(os.listdir(test_cases_dir)):
            path = os.path.join(test_cases_dir, name)
            if os.path.isdir(path) and name.startswith('apl'):
                analyze_with_visualization(path, args.include, not args.simple)
    elif args.project:
        # Analyze specific project
        if os.path.isdir(args.project):
            analyze_with_visualization(args.project, args.include, not args.simple)
        else:
            # Try as test case name
            path = os.path.join(test_cases_dir, args.project)
            if os.path.isdir(path):
                analyze_with_visualization(path, args.include, not args.simple)
            else:
                print(f"Error: Project not found: {args.project}")
                sys.exit(1)
    else:
        # Interactive mode - show menu
        print("\nAvailable test cases:")
        cases = sorted([d for d in os.listdir(test_cases_dir) if d.startswith('apl') and os.path.isdir(os.path.join(test_cases_dir, d))])
        for i, case in enumerate(cases, 1):
            print(f"  {i}. {case}")
        print(f"  a. All cases")
        print()
        
        choice = input("Enter choice (number, name, or 'a' for all): ").strip()
        
        if choice.lower() == 'a':
            for case in cases:
                analyze_with_visualization(os.path.join(test_cases_dir, case), args.include, True)
        elif choice.isdigit() and 1 <= int(choice) <= len(cases):
            analyze_with_visualization(os.path.join(test_cases_dir, cases[int(choice)-1]), args.include, True)
        elif choice in cases:
            analyze_with_visualization(os.path.join(test_cases_dir, choice), args.include, True)
        else:
            print(f"Invalid choice: {choice}")


if __name__ == "__main__":
    main()

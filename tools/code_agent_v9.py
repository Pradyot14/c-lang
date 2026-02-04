#!/usr/bin/env python3
"""
C Code Analysis Agent v9 - Optimized with FULL Verbose Output
==============================================================
Shows everything: system prompt, user prompt, code context, LLM responses
"""

import re
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / ".env")


# ===================== COLORS =====================
class C:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'
    WHITE = '\033[97m'
    BG_GREEN = '\033[42m'
    BG_RED = '\033[41m'
    BG_BLUE = '\033[44m'
    BG_YELLOW = '\033[43m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'


# ===================== CONFIG =====================
@dataclass
class AgentConfig:
    model: str = "gpt-4.1"
    arg_position: int = 3
    max_iterations: int = 10
    verbose: bool = True


# ===================== TOOLS =====================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate arithmetic. Macros are pre-resolved - just use the numbers directly. You can call this multiple times in parallel for different expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression (e.g., '32768 + 1024 + 256 - 48')"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit final answer. Call this as soon as you have the value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "integer",
                        "description": "Final numeric value"
                    },
                    "trace": {
                        "type": "string",
                        "description": "Brief trace: 'main→func1→func2: calculation = answer'"
                    }
                },
                "required": ["value", "trace"]
            }
        }
    }
]


# ===================== AGENT =====================
class CodeAgent:
    """Optimized agent with full verbose output."""
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.client = OpenAI()
        self.macros: Dict[str, int] = {}
        self.current_answer: Optional[int] = None
        self.answer_trace: str = ""
    
    def _parse_macros_from_content(self, content: str):
        """Extract macros from content."""
        for match in re.finditer(r'#define\s+([A-Z_][A-Z0-9_]*)\s+(-?\d+)', content):
            name, value = match.groups()
            self.macros[name] = int(value)
        for match in re.finditer(r'#define\s+([A-Z_][A-Z0-9_]*)\s+(0x[0-9a-fA-F]+)', content):
            name, value = match.groups()
            self.macros[name] = int(value, 16)
    
    def tool_calculate(self, expression: str) -> dict:
        """Evaluate arithmetic expression."""
        expr = expression.strip()
        original = expr
        
        for name, value in self.macros.items():
            expr = re.sub(rf'\b{name}\b', str(value), expr)
        
        expr = re.sub(r'\(int\)|\(long\)|\(unsigned\)', '', expr)
        
        try:
            result = int(eval(expr))
            return {"expression": original, "evaluated": expr, "result": result}
        except Exception as e:
            return {"expression": original, "error": str(e)}
    
    def tool_submit_answer(self, value: int, trace: str) -> dict:
        """Submit final answer."""
        self.current_answer = value
        self.answer_trace = trace
        return {"submitted": True, "value": value, "trace": trace}
    
    def execute_tool(self, name: str, args: dict) -> dict:
        """Route tool call."""
        if name == 'calculate':
            return self.tool_calculate(args.get('expression', ''))
        elif name == 'submit_answer':
            return self.tool_submit_answer(args.get('value', 0), args.get('trace', ''))
        return {'error': f'Unknown tool: {name}'}
    
    def _build_system_prompt(self, target_function: str) -> str:
        ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(
            self.config.arg_position, 
            f"{self.config.arg_position}th"
        )
        
        return f"""You are a C code tracer. Find the {ordinal} argument to {target_function}().

EFFICIENCY RULES:
1. ALL macros are pre-resolved as numbers in the code - use them directly
2. SKIP validation/error checking branches - they don't affect the answer
3. Only trace the HAPPY PATH that reaches {target_function}()
4. Call calculate() with FULL expressions (e.g., "32768 + 1024 + 256 - 48")
5. You can make MULTIPLE tool calls in ONE response - batch your calculations
6. Submit answer as soon as you have it

PROCESS:
1. Read the code, identify the values at each step
2. Substitute macro values (already shown as numbers)
3. Calculate the final value in 1-2 calculate() calls
4. Submit immediately

BE FAST. 2-3 iterations max."""
    
    def _build_user_prompt(self, code_context: str, target_function: str) -> str:
        return f"""Find the {self.config.arg_position}{'st' if self.config.arg_position==1 else 'nd' if self.config.arg_position==2 else 'rd' if self.config.arg_position==3 else 'th'} argument to {target_function}():

{code_context}

Trace the execution path quickly. Use the pre-resolved macro values. Submit your answer."""
    
    def _llm_call(self, messages: list) -> object:
        """Make LLM call."""
        for attempt in range(3):
            try:
                return self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0,
                    parallel_tool_calls=True
                )
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise Exception("Max retries exceeded")
    
    def _print_box(self, title: str, content: str, color=C.CYAN, max_lines: int = None):
        """Print content in a box."""
        lines = content.split('\n')
        if max_lines and len(lines) > max_lines:
            display_lines = lines[:max_lines]
            truncated = len(lines) - max_lines
        else:
            display_lines = lines
            truncated = 0
        
        print(f"\n{color}┌{'─' * 70}┐{C.RESET}")
        print(f"{color}│ {C.BOLD}{title:<68}{C.RESET}{color} │{C.RESET}")
        print(f"{color}├{'─' * 70}┤{C.RESET}")
        
        for line in display_lines:
            # Truncate long lines
            if len(line) > 68:
                line = line[:65] + "..."
            print(f"{color}│{C.RESET} {line:<68} {color}│{C.RESET}")
        
        if truncated:
            print(f"{color}│{C.RESET} {C.DIM}... ({truncated} more lines){C.RESET}{' ' * (68 - 20 - len(str(truncated)))} {color}│{C.RESET}")
        
        print(f"{color}└{'─' * 70}┘{C.RESET}")
    
    def _print_header(self, text, bg_color=C.BG_BLUE):
        print(f"\n{bg_color}{C.WHITE}{C.BOLD} {text} {C.RESET}")
    
    def solve(self, code_context: str, target_function: str = "mpf_mfs_open") -> dict:
        """Solve for the target argument value."""
        self._parse_macros_from_content(code_context)
        
        if self.config.verbose:
            print(f"\n{C.CYAN}{'═' * 72}{C.RESET}")
            print(f"{C.CYAN}{C.BOLD}  CODE AGENT v9 - FULL VERBOSE MODE{C.RESET}")
            print(f"{C.CYAN}{'═' * 72}{C.RESET}")
            print(f"  {C.WHITE}Target:{C.RESET} {C.GREEN}{target_function}(){C.RESET} arg #{C.YELLOW}{self.config.arg_position}{C.RESET}")
            print(f"  {C.WHITE}Model:{C.RESET}  {C.GRAY}{self.config.model}{C.RESET}")
            print(f"  {C.WHITE}Macros:{C.RESET} {C.GRAY}{len(self.macros)} loaded{C.RESET}")
        
        system_prompt = self._build_system_prompt(target_function)
        user_prompt = self._build_user_prompt(code_context, target_function)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ========== SHOW FULL SYSTEM PROMPT ==========
        if self.config.verbose:
            self._print_box("SYSTEM PROMPT (sent to LLM)", system_prompt, C.BLUE)
        
        # ========== SHOW FULL USER PROMPT (with code) ==========
        if self.config.verbose:
            self._print_box("USER PROMPT (sent to LLM)", user_prompt, C.GREEN, max_lines=80)
        
        self.current_answer = None
        iterations = 0
        
        while iterations < self.config.max_iterations and self.current_answer is None:
            iterations += 1
            
            if self.config.verbose:
                print(f"\n{C.MAGENTA}{'━' * 72}{C.RESET}")
                print(f"{C.MAGENTA}{C.BOLD}  ITERATION {iterations}{C.RESET}")
                print(f"{C.MAGENTA}{'━' * 72}{C.RESET}")
                print(f"\n  {C.YELLOW}▶ Calling LLM with {len(messages)} messages...{C.RESET}")
            
            start_time = time.time()
            response = self._llm_call(messages)
            elapsed = time.time() - start_time
            
            msg = response.choices[0].message
            
            if self.config.verbose:
                print(f"  {C.GRAY}Response time: {elapsed:.2f}s{C.RESET}")
                
                # ========== SHOW FULL LLM RESPONSE TEXT ==========
                if msg.content:
                    self._print_box("LLM RESPONSE (text content)", msg.content, C.YELLOW)
                else:
                    print(f"\n  {C.GRAY}(No text content in response){C.RESET}")
            
            if not msg.tool_calls:
                if self.config.verbose:
                    print(f"\n  {C.RED}✗ No tool calls - prompting LLM to use tools{C.RESET}")
                messages.append({"role": "assistant", "content": msg.content})
                messages.append({
                    "role": "user", 
                    "content": "Use calculate() and submit_answer() now."
                })
                continue
            
            # ========== SHOW RAW TOOL CALLS ==========
            if self.config.verbose:
                raw_tool_calls = []
                for tc in msg.tool_calls:
                    raw_tool_calls.append(f"Tool: {tc.function.name}")
                    raw_tool_calls.append(f"  ID: {tc.id}")
                    raw_tool_calls.append(f"  Args: {tc.function.arguments}")
                    raw_tool_calls.append("")
                self._print_box(f"RAW TOOL CALLS ({len(msg.tool_calls)} calls)", '\n'.join(raw_tool_calls), C.CYAN)
            
            # Add assistant message
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            })
            
            # ========== EXECUTE AND SHOW TOOL RESULTS ==========
            if self.config.verbose:
                self._print_header(f"TOOL EXECUTION & RESULTS", C.BG_GREEN)
            
            tool_results_log = []
            
            for i, tc in enumerate(msg.tool_calls, 1):
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                
                if self.config.verbose:
                    print(f"\n  {C.CYAN}┌─ Tool {i}: {C.BOLD}{name}{C.RESET}")
                    print(f"  {C.CYAN}│{C.RESET}  {C.WHITE}Arguments:{C.RESET} {C.YELLOW}{json.dumps(args)}{C.RESET}")
                
                result = self.execute_tool(name, args)
                
                if self.config.verbose:
                    print(f"  {C.CYAN}│{C.RESET}  {C.WHITE}Result:{C.RESET} {C.GREEN}{json.dumps(result)}{C.RESET}")
                    
                    if name == 'calculate' and 'result' in result:
                        print(f"  {C.CYAN}└─ {C.GREEN}{C.BOLD}{args.get('expression')} = {result['result']}{C.RESET}")
                    elif name == 'submit_answer':
                        print(f"  {C.CYAN}└─ {C.GREEN}{C.BOLD}✓ SUBMITTED: {result.get('value')}{C.RESET}")
                    elif 'error' in result:
                        print(f"  {C.CYAN}└─ {C.RED}{C.BOLD}ERROR: {result.get('error')}{C.RESET}")
                    else:
                        print(f"  {C.CYAN}└─────────{C.RESET}")
                
                tool_results_log.append({
                    "tool_call_id": tc.id,
                    "name": name,
                    "args": args,
                    "result": result
                })
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result)
                })
            
            # ========== SHOW WHAT WE'RE SENDING BACK TO LLM ==========
            if self.config.verbose and self.current_answer is None:
                tool_msg_preview = []
                for tr in tool_results_log:
                    tool_msg_preview.append(f"[tool_call_id: {tr['tool_call_id']}]")
                    tool_msg_preview.append(f"  → {json.dumps(tr['result'])}")
                self._print_box("TOOL RESULTS (sent back to LLM)", '\n'.join(tool_msg_preview), C.MAGENTA)
        
        # Final Result
        if self.config.verbose:
            print(f"\n{C.CYAN}{'═' * 72}{C.RESET}")
        
        if self.current_answer is not None:
            if self.config.verbose:
                print(f"{C.BG_GREEN}{C.WHITE}{C.BOLD}  ✓ SOLVED  {C.RESET}")
                print(f"{C.CYAN}{'═' * 72}{C.RESET}")
                print(f"  {C.WHITE}Answer:{C.RESET}     {C.GREEN}{C.BOLD}{self.current_answer}{C.RESET}")
                print(f"  {C.WHITE}Iterations:{C.RESET} {C.YELLOW}{iterations}{C.RESET}")
                print(f"  {C.WHITE}Trace:{C.RESET}      {C.GRAY}{self.answer_trace}{C.RESET}")
                print(f"{C.CYAN}{'═' * 72}{C.RESET}\n")
            
            return {
                "answer": self.current_answer,
                "trace": self.answer_trace,
                "iterations": iterations,
                "success": True
            }
        else:
            if self.config.verbose:
                print(f"{C.BG_RED}{C.WHITE}{C.BOLD}  ✗ FAILED  {C.RESET}")
                print(f"{C.CYAN}{'═' * 72}{C.RESET}")
                print(f"  {C.WHITE}Iterations:{C.RESET} {C.RED}{iterations}{C.RESET}")
                print(f"  {C.WHITE}Reason:{C.RESET}     {C.RED}Max iterations reached{C.RESET}")
                print(f"{C.CYAN}{'═' * 72}{C.RESET}\n")
            
            return {
                "answer": None,
                "iterations": iterations,
                "success": False
            }
    
    def solve_from_file(self, path_trace_file: str, target_function: str = "mpf_mfs_open") -> dict:
        """Solve using a path trace file."""
        content = Path(path_trace_file).read_text(errors='ignore')
        return self.solve(content, target_function)


# ===================== MAIN =====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Code Agent v9")
    parser.add_argument("input", help="Path trace file or project directory")
    parser.add_argument("--target", default="mpf_mfs_open", help="Target function")
    parser.add_argument("--arg", type=int, default=3, help="Argument position")
    parser.add_argument("--model", default="gpt-4.1", help="Model")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--include", "-I", action="append", default=[], help="Include paths for headers")
    
    args = parser.parse_args()
    
    config = AgentConfig(
        model=args.model,
        arg_position=args.arg,
        verbose=not args.quiet
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Direct file input - single path
        agent = CodeAgent(config)
        result = agent.solve_from_file(str(input_path), args.target)
        exit(0 if result["success"] else 1)
    
    elif input_path.is_dir():
        # Check for existing path trace files
        multi_files = sorted(input_path.glob('_path_trace_*.c'))
        single_file = input_path / '_path_trace.c'
        
        # If no trace files exist, generate them
        if not multi_files and not single_file.exists():
            if config.verbose:
                print(f"\n{C.YELLOW}⚙ Generating path trace(s)...{C.RESET}")
            
            try:
                from path_formatter import format_path_to_file
            except ImportError:
                from src.path_formatter import format_path_to_file
            
            # Find include paths
            include_paths = args.include.copy()
            test_dir = Path(__file__).parent.parent / 'test_cases'
            if (test_dir / 'include').exists():
                include_paths.append(str(test_dir / 'include'))
            
            # Generate trace(s)
            format_path_to_file(str(input_path), args.target, include_paths)
            
            # Re-check for files
            multi_files = sorted(input_path.glob('_path_trace_*.c'))
        
        # Determine which files to process
        if multi_files:
            trace_files = [str(f) for f in multi_files]
        elif single_file.exists():
            trace_files = [str(single_file)]
        else:
            print(f"Error: No path trace files found in {input_path}")
            exit(1)
        
        # Process all paths
        results = []
        all_success = True
        
        for i, trace_file in enumerate(trace_files, 1):
            if config.verbose and len(trace_files) > 1:
                print(f"\n{C.CYAN}{'═' * 72}{C.RESET}")
                print(f"{C.CYAN}{C.BOLD}  PROCESSING PATH {i} of {len(trace_files)}{C.RESET}")
                print(f"{C.CYAN}  File: {Path(trace_file).name}{C.RESET}")
                print(f"{C.CYAN}{'═' * 72}{C.RESET}")
            
            agent = CodeAgent(config)
            result = agent.solve_from_file(trace_file, args.target)
            result['path_file'] = trace_file
            result['path_index'] = i
            results.append(result)
            
            if not result["success"]:
                all_success = False
        
        # Summary for multiple paths
        if len(results) > 1 and config.verbose:
            print(f"\n{C.CYAN}{'═' * 72}{C.RESET}")
            print(f"{C.CYAN}{C.BOLD}  SUMMARY - ALL PATHS{C.RESET}")
            print(f"{C.CYAN}{'═' * 72}{C.RESET}")
            print(f"\n  {'Path':<8} {'Answer':<12} {'Iters':<8} {'Status'}")
            print(f"  {'-'*8} {'-'*12} {'-'*8} {'-'*8}")
            
            for r in results:
                idx = r['path_index']
                ans = r.get('answer', 'N/A')
                iters = r.get('iterations', '?')
                status = f"{C.GREEN}✓{C.RESET}" if r['success'] else f"{C.RED}✗{C.RESET}"
                print(f"  Path {idx:<4} {str(ans):<12} {str(iters):<8} {status}")
            
            print(f"\n{C.CYAN}{'═' * 72}{C.RESET}\n")
        
        exit(0 if all_success else 1)
    
    else:
        print(f"Error: {args.input} not found")
        exit(1)

#!/usr/bin/env python3
"""
Data Flow Diagram Agent
=======================
Generates visual data flow diagrams showing how values flow through code
to reach the final answer (computed by code_agent_v9).

Supports: OpenAI and Azure OpenAI
Output: Mermaid diagram (renders in markdown viewers, GitHub, etc.)
"""

import re
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI

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
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'


# ===================== CONFIG =====================
@dataclass
class DiagramConfig:
    model: str = "gpt-4.1"
    arg_position: int = 3
    verbose: bool = True
    # Azure OpenAI settings
    use_azure: bool = False
    azure_endpoint: str = ""
    azure_deployment: str = ""
    azure_api_version: str = "2024-02-15-preview"
    # Output settings
    output_format: str = "mermaid"  # mermaid, ascii


# ===================== AGENT =====================
class DataFlowDiagramAgent:
    """Agent that generates data flow diagrams from C code traces."""
    
    def __init__(self, config: DiagramConfig = None):
        self.config = config or DiagramConfig()
        
        # Initialize client based on Azure or standard OpenAI
        if self.config.use_azure:
            self.client = AzureOpenAI(
                azure_endpoint=self.config.azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=self.config.azure_api_version
            )
            self.model_name = self.config.azure_deployment or self.config.model
        else:
            self.client = OpenAI()
            self.model_name = self.config.model
    
    def _build_system_prompt(self, target_function: str) -> str:
        ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(
            self.config.arg_position,
            f"{self.config.arg_position}th"
        )
        
        return f"""You are a C code analyzer that creates Data Flow Diagrams (DFD).

TASK: Create a proper DFD showing how data flows to compute the {ordinal} argument of {target_function}().

DFD NOTATION (use these Mermaid shapes):
- PROCESS (circle): ((Process Name)) - Functions that transform data
- EXTERNAL ENTITY (rectangle): [Entity Name] - Entry/exit points  
- DATA STORE (cylinder): [(Store Name)] - Variables, macros, constants
- DATA FLOW: Arrows with labels showing what data moves

OUTPUT FORMAT - Return ONLY valid Mermaid:

```mermaid
flowchart TD
    %% External Entities
    EXT1[main - Entry Point]
    EXT2[{target_function} - Target]
    
    %% Data Stores (macros/constants)
    DS1[(BASE = 1000)]
    DS2[(OFFSET = 100)]
    
    %% Processes (functions that transform data)
    P1((compute_value))
    P2((add_offset))
    
    %% Data Flows with labels
    EXT1 -->|"calls with args"| P1
    DS1 -->|"base value"| P1
    P1 -->|"intermediate = 2600"| P2
    DS2 -->|"offset"| P2
    P2 -->|"final = 2900"| EXT2
    
    %% Final Answer
    ANS[/"✅ arg{self.config.arg_position} = FINAL_VALUE"/]
    EXT2 -->|"result"| ANS
```

DFD RULES:
1. Use ((name)) for PROCESSES - functions that compute/transform values
2. Use [(name)] for DATA STORES - macros, constants, variables holding values
3. Use [name] for EXTERNAL ENTITIES - entry point (main) and target function
4. Use [/name/] for the final answer (parallelogram shape)
5. LABEL EVERY ARROW with the data being passed (use |"label"| syntax)
6. Show actual numeric values in labels: |"base = 1000"| not just |"base"|
7. Show step-by-step computation: |"100 << 4 = 1600"|
8. Keep process names short but descriptive
9. Data flows TOP to BOTTOM (TD direction)
10. Return ONLY the mermaid code block, nothing else

EXAMPLE for fileno = BASE + (mode << 4) + OFFSET:
```mermaid
flowchart TD
    EXT1[main]
    DS1[(BASE = 1000)]
    DS2[(OFFSET = 100)]
    P1((shift: mode<<4))
    P2((add base))
    P3((add offset))
    EXT2[target_func]
    
    EXT1 -->|"mode = 100"| P1
    P1 -->|"1600"| P2
    DS1 -->|"1000"| P2
    P2 -->|"2600"| P3
    DS2 -->|"100"| P3
    P3 -->|"fileno = 2700"| EXT2
```"""

    def _build_user_prompt(self, code_context: str, target_function: str) -> str:
        return f"""Create a data flow diagram for this code showing how we compute the {self.config.arg_position}{'st' if self.config.arg_position==1 else 'nd' if self.config.arg_position==2 else 'rd' if self.config.arg_position==3 else 'th'} argument to {target_function}():

{code_context}

Trace every value transformation. Show the complete data flow from entry to the final computed value."""

    def _llm_call(self, messages: list) -> object:
        """Make LLM call."""
        for attempt in range(3):
            try:
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0
                )
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    time.sleep(2 ** attempt)
                else:
                    raise
        raise Exception("Max retries exceeded")

    def _extract_mermaid(self, response_text: str) -> str:
        """Extract mermaid diagram from response."""
        # Try to find mermaid code block
        match = re.search(r'```mermaid\s*(.*?)\s*```', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Try to find any code block
        match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Return as-is if no code block found
        return response_text.strip()

    def generate(self, code_context: str, target_function: str = "mpf_mfs_open", 
                 final_answer: int = None) -> dict:
        """Generate data flow diagram."""
        
        if self.config.verbose:
            print(f"\n{C.CYAN}{'═' * 72}{C.RESET}")
            print(f"{C.CYAN}{C.BOLD}  DATA FLOW DIAGRAM AGENT{C.RESET}")
            print(f"{C.CYAN}{'═' * 72}{C.RESET}")
            provider = "Azure OpenAI" if self.config.use_azure else "OpenAI"
            print(f"  {C.WHITE}Provider:{C.RESET} {C.GRAY}{provider}{C.RESET}")
            print(f"  {C.WHITE}Model:{C.RESET}   {C.GRAY}{self.model_name}{C.RESET}")
            print(f"  {C.WHITE}Target:{C.RESET}  {C.GREEN}{target_function}(){C.RESET} arg #{C.YELLOW}{self.config.arg_position}{C.RESET}")
            if final_answer is not None:
                print(f"  {C.WHITE}Answer:{C.RESET}  {C.GREEN}{final_answer}{C.RESET}")
        
        # Add final answer hint if provided
        context = code_context
        if final_answer is not None:
            context += f"\n\n// KNOWN ANSWER: The {self.config.arg_position}{'st' if self.config.arg_position==1 else 'nd' if self.config.arg_position==2 else 'rd' if self.config.arg_position==3 else 'th'} argument = {final_answer}"
        
        system_prompt = self._build_system_prompt(target_function)
        user_prompt = self._build_user_prompt(context, target_function)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        if self.config.verbose:
            print(f"\n  {C.YELLOW}▶ Generating diagram...{C.RESET}")
        
        start_time = time.time()
        response = self._llm_call(messages)
        elapsed = time.time() - start_time
        
        response_text = response.choices[0].message.content or ""
        mermaid_code = self._extract_mermaid(response_text)
        
        if self.config.verbose:
            print(f"  {C.DIM}Response time: {elapsed:.2f}s{C.RESET}")
            print(f"\n{C.GREEN}{'═' * 72}{C.RESET}")
            print(f"{C.GREEN}{C.BOLD}  MERMAID DIAGRAM{C.RESET}")
            print(f"{C.GREEN}{'═' * 72}{C.RESET}")
            print(f"\n```mermaid")
            print(mermaid_code)
            print(f"```\n")
        
        return {
            "mermaid": mermaid_code,
            "raw_response": response_text,
            "elapsed": elapsed
        }

    def generate_from_file(self, path_trace_file: str, target_function: str = "mpf_mfs_open",
                           final_answer: int = None) -> dict:
        """Generate diagram from a path trace file."""
        content = Path(path_trace_file).read_text(errors='ignore')
        return self.generate(content, target_function, final_answer)

    def save_diagram(self, mermaid_code: str, output_path: str):
        """Save diagram to file."""
        # Save as .md for easy viewing
        md_content = f"""# Data Flow Diagram

```mermaid
{mermaid_code}
```

## How to View

1. **GitHub/GitLab**: Just open this .md file - they render Mermaid natively
2. **VS Code**: Install "Markdown Preview Mermaid Support" extension
3. **Online**: Paste the mermaid code at https://mermaid.live/
4. **CLI**: Use `mmdc` (mermaid-cli) to convert to PNG/SVG
"""
        Path(output_path).write_text(md_content)
        if self.config.verbose:
            print(f"  {C.GREEN}✓ Saved:{C.RESET} {output_path}")


# ===================== MAIN =====================
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Data Flow Diagram Agent - Full Pipeline")
    parser.add_argument("input", help="Project directory with C source files")
    parser.add_argument("--target", default="mpf_mfs_open", help="Target function")
    parser.add_argument("--arg", type=int, default=3, help="Argument position")
    parser.add_argument("--model", default="gpt-4.1", help="Model (or Azure deployment name)")
    parser.add_argument("--output", "-o", default=None, help="Output file path (.md)")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--include", "-I", action="append", default=[], help="Include paths for headers")
    # Azure OpenAI options
    parser.add_argument("--azure", action="store_true", help="Use Azure OpenAI")
    parser.add_argument("--azure-endpoint", default="", help="Azure OpenAI endpoint")
    parser.add_argument("--azure-deployment", default="", help="Azure deployment name")
    parser.add_argument("--azure-api-version", default="2024-02-15-preview", help="Azure API version")
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    # Import dependencies
    try:
        from path_formatter import format_path_to_file
        from code_agent_v9 import CodeAgent, AgentConfig
    except ImportError:
        from src.path_formatter import format_path_to_file
        from src.code_agent_v9 import CodeAgent, AgentConfig
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"{C.RED}Error: {input_path} not found{C.RESET}")
        sys.exit(1)
    
    # ============ STEP 1: Generate Path Traces ============
    if verbose:
        print(f"\n{C.CYAN}{'═' * 72}{C.RESET}")
        print(f"{C.CYAN}{C.BOLD}  STEP 1: PARSE & FORMAT PATHS{C.RESET}")
        print(f"{C.CYAN}{'═' * 72}{C.RESET}")
    
    trace_files = []
    
    if input_path.is_dir():
        # Clean old traces and diagrams
        for old_file in input_path.glob('_path_trace*.c'):
            old_file.unlink()
        for old_file in input_path.glob('_dataflow*.md'):
            old_file.unlink()
        
        # Generate path traces (this also parses)
        if verbose:
            print(f"  {C.YELLOW}⚙ Parsing project and generating path traces...{C.RESET}")
        
        format_path_to_file(
            str(input_path), 
            args.target,
            include_paths=args.include
        )
        
        # Collect all trace files
        multi_traces = sorted(input_path.glob('_path_trace_*.c'))
        single_trace = input_path / '_path_trace.c'
        
        if multi_traces:
            trace_files = list(multi_traces)
        elif single_trace.exists():
            trace_files = [single_trace]
        
        if not trace_files:
            print(f"{C.RED}Error: No path traces generated{C.RESET}")
            sys.exit(1)
        
        if verbose:
            print(f"  {C.GREEN}✓ Found {len(trace_files)} path(s){C.RESET}")
            
    else:
        trace_files = [input_path]
    
    # ============ STEP 2 & 3: Process Each Path ============
    all_results = []  # Store results for each path
    
    agent_config = AgentConfig(
        model=args.model,
        arg_position=args.arg,
        verbose=False,  # We'll control output ourselves
        use_azure=args.azure,
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment or args.model,
        azure_api_version=args.azure_api_version
    )
    
    diagram_config = DiagramConfig(
        model=args.model,
        arg_position=args.arg,
        verbose=False,  # We'll control output ourselves
        use_azure=args.azure,
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment or args.model,
        azure_api_version=args.azure_api_version
    )
    
    for i, trace_file in enumerate(trace_files, 1):
        path_num = i
        
        # Extract path description from trace file
        trace_content = trace_file.read_text(errors='ignore')
        path_desc = "Unknown path"
        for line in trace_content.split('\n'):
            if line.startswith('// PATH:'):
                path_desc = line.replace('// PATH:', '').strip()
                break
        
        if verbose:
            print(f"\n{C.MAGENTA}{'━' * 72}{C.RESET}")
            print(f"{C.MAGENTA}{C.BOLD}  PATH {path_num}/{len(trace_files)}: {path_desc}{C.RESET}")
            print(f"{C.MAGENTA}{'━' * 72}{C.RESET}")
        
        # ---- Step 2: Compute Answer ----
        if verbose:
            print(f"\n  {C.YELLOW}⚙ Computing answer (code_agent_v9)...{C.RESET}")
        
        code_agent = CodeAgent(agent_config)
        result = code_agent.solve_from_file(str(trace_file), args.target)
        
        if not result["success"]:
            if verbose:
                print(f"  {C.RED}✗ Failed to compute answer{C.RESET}")
            all_results.append({
                "path_num": path_num,
                "path_desc": path_desc,
                "answer": None,
                "mermaid": None,
                "trace_file": str(trace_file),
                "success": False
            })
            continue
        
        final_answer = result["answer"]
        if verbose:
            print(f"  {C.GREEN}✓ Answer: {C.BOLD}{final_answer}{C.RESET}")
        
        # ---- Step 3: Generate DFD ----
        if verbose:
            print(f"  {C.YELLOW}⚙ Generating DFD...{C.RESET}")
        
        diagram_agent = DataFlowDiagramAgent(diagram_config)
        diagram_result = diagram_agent.generate_from_file(str(trace_file), args.target, final_answer)
        
        if verbose:
            print(f"  {C.GREEN}✓ DFD generated{C.RESET}")
        
        all_results.append({
            "path_num": path_num,
            "path_desc": path_desc,
            "answer": final_answer,
            "mermaid": diagram_result["mermaid"],
            "trace_file": str(trace_file),
            "success": True
        })
    
    # ============ STEP 4: Combine All DFDs into One File ============
    if verbose:
        print(f"\n{C.CYAN}{'═' * 72}{C.RESET}")
        print(f"{C.CYAN}{C.BOLD}  STEP 4: SAVE COMBINED DIAGRAM{C.RESET}")
        print(f"{C.CYAN}{'═' * 72}{C.RESET}")
    
    # Build combined markdown
    md_parts = []
    md_parts.append(f"# Data Flow Diagrams - {args.target}()\n")
    md_parts.append(f"**Target:** `{args.target}()` argument #{args.arg}\n")
    md_parts.append(f"**Total Paths:** {len(all_results)}\n")
    md_parts.append("\n---\n")
    
    # Summary table
    md_parts.append("## Summary\n")
    md_parts.append("| Path | Route | Answer | Status |")
    md_parts.append("|------|-------|--------|--------|")
    for r in all_results:
        status = "✅" if r["success"] else "❌"
        answer = str(r["answer"]) if r["answer"] is not None else "N/A"
        md_parts.append(f"| {r['path_num']} | {r['path_desc']} | {answer} | {status} |")
    md_parts.append("\n---\n")
    
    # Individual DFDs
    for r in all_results:
        md_parts.append(f"## Path {r['path_num']}: {r['path_desc']}\n")
        md_parts.append(f"**Answer:** `{r['answer']}`\n")
        
        if r["mermaid"]:
            md_parts.append(f"\n```mermaid\n{r['mermaid']}\n```\n")
        else:
            md_parts.append("\n*Failed to generate diagram*\n")
        
        md_parts.append("\n---\n")
    
    # How to view section
    md_parts.append("## How to View\n")
    md_parts.append("1. **GitHub/GitLab**: Just open this .md file - they render Mermaid natively")
    md_parts.append("2. **VS Code**: Install \"Markdown Preview Mermaid Support\" extension")
    md_parts.append("3. **Online**: Paste the mermaid code at https://mermaid.live/")
    md_parts.append("4. **CLI**: Use `mmdc` (mermaid-cli) to convert to PNG/SVG")
    
    combined_md = '\n'.join(md_parts)
    
    # Save combined file
    if args.output:
        output_path = args.output
    else:
        output_path = str(input_path / "_dataflow_diagram.md") if input_path.is_dir() else str(input_path.parent / "_dataflow_diagram.md")
    
    Path(output_path).write_text(combined_md)
    
    if verbose:
        print(f"  {C.GREEN}✓ Saved:{C.RESET} {output_path}")
    
    # ============ FINAL SUMMARY ============
    successful = sum(1 for r in all_results if r["success"])
    
    if verbose:
        print(f"\n{C.GREEN}{'═' * 72}{C.RESET}")
        print(f"{C.GREEN}{C.BOLD}  ✅ PIPELINE COMPLETE{C.RESET}")
        print(f"{C.GREEN}{'═' * 72}{C.RESET}")
        print(f"  {C.WHITE}Target:{C.RESET}    {C.CYAN}{args.target}(){C.RESET} arg #{C.YELLOW}{args.arg}{C.RESET}")
        print(f"  {C.WHITE}Paths:{C.RESET}     {C.GREEN}{successful}/{len(all_results)} successful{C.RESET}")
        print(f"  {C.WHITE}Diagram:{C.RESET}   {C.GRAY}{output_path}{C.RESET}")
        print()
        print(f"  {C.WHITE}Results:{C.RESET}")
        for r in all_results:
            status = f"{C.GREEN}✓{C.RESET}" if r["success"] else f"{C.RED}✗{C.RESET}"
            answer = f"{C.BOLD}{r['answer']}{C.RESET}" if r["answer"] else "N/A"
            print(f"    {status} Path {r['path_num']}: {r['path_desc']} → {answer}")
        print(f"{C.GREEN}{'═' * 72}{C.RESET}\n")

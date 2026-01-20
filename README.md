# FileNo Extractor v2.0

A powerful tool to extract file numbers from `mpf_mfs_open()` calls in C projects.

## Features

- **Two Analysis Methods:**
  - 🔍 **Pattern-Based**: Fast, offline analysis using regex and AST parsing
  - 🤖 **AI-Based**: Azure OpenAI GPT-4 powered analysis for complex cases

- **Handles Complex Patterns:**
  - Direct numbers, macros, nested macros
  - Variable assignments, if-else, switch-case
  - Multi-level function calls, cross-file analysis
  - Arithmetic expressions
  - Function pointers/callbacks

- **Starts from `int main()`**: Only analyzes reachable code paths

- **Comprehensive HTML Visualization**: View complete AST, macros, functions, call graph

## Installation

```bash
# Clone the repository
git clone https://github.com/Pradyot14/c-lang.git
cd c-lang

# Install UV (Linux/Mac)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install UV (Windows)
pip install uv

# Setup environment
uv sync
```

## Configuration (for AI method)

Edit the `.env` file with your Azure OpenAI credentials:

```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

## Commands

### Pattern-Based Analysis (Default - No API needed)

```bash
# Analyze a project
uv run python src/main.py /path/to/project

# Analyze a specific test case
uv run python src/main.py test_cases/apl001

# Run on all test cases
uv run python src/main.py --all
```

### AI-Based Analysis (Requires Azure OpenAI)

```bash
# Analyze a project with AI
uv run python src/main.py /path/to/project --method ai

# Analyze a specific test case with AI
uv run python src/main.py test_cases/apl001 --method ai

# Run on all test cases with AI
uv run python src/main.py --all --method ai
```

### Compare Both Methods

```bash
uv run python src/main.py --compare
```

### Export Results to JSON

```bash
uv run python src/main.py /path/to/project -o results.json
```

### Tree Visualization (AST)

```bash
# ASCII tree (terminal)
uv run python src/main.py --tree apl006

# HTML tree (comprehensive - opens in browser)
uv run python src/main.py --tree apl006 --format html

# Mermaid diagram (for markdown)
uv run python src/main.py --tree apl006 --format mermaid

# Graphviz DOT (for images)
uv run python src/main.py --tree apl006 --format graphviz

# All test cases as HTML
uv run python src/main.py --tree-all --format html
```

### Help

```bash
uv run python src/main.py --help
```

## HTML Tree Visualization Includes

- 📋 **Analysis Summary**: File, line, function, resolved value
- 📝 **Resolution Steps**: Step-by-step trace
- 🌳 **Data Flow Tree**: Interactive tree with color-coded nodes
- 🔗 **Call Graph**: `main() → func1() → target()` path
- 🔧 **Macro Definitions**: ALL macros with expanded values
- 📞 **Functions**: ALL functions with collapsible code
- 📄 **Source Files**: Tabbed view with line highlighting

## Project Structure

```
.
├── src/
│   ├── main.py              # CLI entry point
│   ├── robust_analyzer.py   # Pattern-based analyzer
│   ├── ai_analyzer.py       # Azure OpenAI analyzer
│   ├── data_flow_analyzer.py
│   ├── ast_parser.py
│   ├── macro_extractor.py
│   ├── tree_visualizer.py   # HTML/ASCII/Mermaid visualization
│   └── config.py
├── test_cases/              # Sample test cases (apl001-apl012, apl100)
├── output/                  # Generated HTML trees
├── .env                     # Azure OpenAI credentials
├── pyproject.toml
└── README.md
```

## Method Comparison

| Feature | Pattern-Based | AI-Based |
|---------|--------------|----------|
| Speed | ⚡ Fast | 🐢 Slower |
| Offline | ✅ Yes | ❌ No |
| Cost | 💰 Free | 💸 API costs |
| Complex Cases | ⚠️ Limited | ✅ Better |

## License

MIT License

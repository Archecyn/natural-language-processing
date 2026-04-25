# Azure DevOps AI Code Review Agent

An agentic Python workflow that inspects an **Azure DevOps** Git repository,
analyses its code with a **local** Qwen2.5-Coder LLM (no API costs, fully
offline inference), and automatically opens a Pull Request with its
recommendations.

---

## Architecture

```
agent.py               ← Orchestrator — ties everything together
azure_devops_client.py ← Azure DevOps REST API: list files, read content, commit, PR
llm_client.py          ← Local LLM via llama_cpp (Qwen2.5-Coder-3B-Instruct-GGUF)
code_analyser.py       ← Prompt builder + response parser
pr_creator.py          ← Branch creation, file commits, PR body generation
```

### Agentic Workflow

```
Azure DevOps Repo
    │
    ▼
[1] List code files  (azure_devops_client.list_code_files)
    │
    ▼ for each file
[2] Fetch content    (azure_devops_client.get_file_content)
    │
    ▼
[3] LLM analysis     (code_analyser.analyse → llm_client.chat)
    │                 Qwen2.5-Coder-3B-Instruct running locally via llama_cpp
    ▼
[4] Parse response   → (revised_code, summary) or None
    │
    ▼ if changes found
[5] Create branch    (azure_devops_client.create_branch)
[6] Commit files     (azure_devops_client.commit_file) × N
[7] Open PR          (azure_devops_client.create_pull_request)
    │
    ▼
  PR URL ✅
```

---

## Model

| Setting | Value |
|---------|-------|
| Model   | `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` |
| File    | `qwen2.5-coder-7b-instruct-q4_k_m.gguf` |
| Quant   | Q4_K_M (~4.2 GB) |
| Context | 32768 tokens |
| Backend | `llama_cpp` via HuggingFace Hub |

**Available Model Sizes:**
- **3B** (~1.9GB): `Qwen/Qwen2.5-Coder-3B-Instruct-GGUF` - Faster, less accurate
- **7B** (~4.2GB): `Qwen/Qwen2.5-Coder-7B-Instruct-GGUF` - **Current default** - Better code analysis
- **14B** (~8.0GB): `Qwen/Qwen2.5-Coder-14B-Instruct-GGUF` - More accurate, slower
- **32B** (~18GB): `Qwen/Qwen2.5-Coder-32B-Instruct-GGUF` - Best accuracy, requires powerful hardware

The model is **downloaded automatically** on first run to `~/.cache/huggingface/`.
Subsequent runs reuse the cached file.

---

## Installation

### 1. Clone / copy the project files

```bash
git clone <your-fork>
cd ado_code_review_level1
```

### 2. Install Conda

If you don't have Conda installed, download and install it from one of these options:

- **Miniconda** (recommended, lightweight): https://docs.conda.io/projects/miniconda/en/latest/
- **Anaconda** (full distribution): https://www.anaconda.com/download

Verify installation:
```bash
conda --version
```

### 3. Create and activate the Conda environment

Create a new environment called `codeagent`:
```bash
conda create -n codeagent python=3.11
```

Activate the environment:
```bash
conda activate codeagent
```

### 4. Install Python dependencies

**CPU only (universal):**
```bash
pip install -r requirements.txt
```

**GPU acceleration (NVIDIA CUDA — recommended for speed):**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall
pip install requests huggingface-hub
```

**GPU acceleration (Apple Silicon Metal):**
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall
pip install requests huggingface-hub
```

### 5. Set up your Azure DevOps Personal Access Token

Create a PAT at https://dev.azure.com under your user profile. Grant **Code (read & write)**
permissions for the target repository.

```bash
export AZDO_TOKEN=YOUR_PERSONAL_ACCESS_TOKEN
```

Or pass it directly via `--token`.

---

## Usage

### Command line

```bash
# Basic — analyse everything, create PR on 'ai-code-review' branch
python agent.py myorg/myproj/myrepo

# Fix errors only
python agent.py myorg/myproj/myrepo --mode errors

# Optimise only (performance + readability)
python agent.py myorg/myproj/myrepo --mode optimise

# Focus on a specific folder
python agent.py myorg/myproj/myrepo --folder src/components --mode optimise

# Scan entire repository (unlimited files)
python agent.py myorg/myproj/myrepo --max-files 0 --mode optimise

# Analyse only SQL files
python agent.py myorg/myproj/myrepo --file-extensions .sql --mode errors

# Analyse Python and SQL files
python agent.py myorg/myproj/myrepo --file-extensions .py,.sql --mode errors

# Use larger model for better analysis
python agent.py myorg/myproj/myrepo --model-size 14b --mode optimise

# Full options
python agent.py myorg/myproj/myrepo \
  --mode both \
  --base-branch main \
  --pr-branch ai-review-$(date +%Y%m%d) \
  --max-files 15 \
  --folder src/components \
  --file-extensions .py,.js,.ts \
  --model-size 7b \
  --token $AZDO_TOKEN
```

### As a Python module

```python
from agent import AzureDevOpsAIAgent

dagent = AzureDevOpsAIAgent(
    azure_token="...",
    repo_full_name="myorg/myproj/myrepo",
)
url = dagent.run()
print(url)
```


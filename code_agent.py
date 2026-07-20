from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import select
import uuid
import os
import yaml
import json
import re
import subprocess
import operator

MARKER = '--COMMAND--FINISHED--\n'
FILE_REPO = 'https://github.com/polakowo/vectorbt'


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    data: dict # Github data
    files: list # github files to use as context
    paper: dict # original paper
    ROLES: dict # Roles for agent
    decision_tree: dict # Decision tree yaml
    code_agent: ChatOllama
    process: subprocess.Popen
    attempts: int
    context: str
    code: str
    idx: int
    file_path: str

def run_command(state: AgentState, command: str):
    process = state['process']
    assert process.stdin and process.stdout and process.stderr

    stdin, stdout, stderr = process.stdin, process.stdout, process.stderr

    def excecute_command(cmd):
        marker = uuid.uuid4().hex
        stdin.write(f'{cmd}\n')
        stdin.write(f'echo {marker}\n')
        stdin.flush()

        streams = {stdout: 'stdout', stderr: 'stderr'}
        done = False
        lines = []
        while not done:
            ready, _, _ = select.select(streams.keys(), [], [])
            for stream in ready:
                line = stream.readline()
                if not line:
                    continue
                line = line.rstrip('\n')
                if line == marker:
                    done = True
                    continue
                if stream == stderr:
                    lines.append(line)
        if len(lines) != 0:
            error = "\n".join(lines)
            print(error)
            return error
        return "Success"    
    response = excecute_command(command)
    
    return response

def router(state: AgentState) -> dict:
    role = state['ROLES']['Developer']
    decision = yaml.dump(state['decision_tree']['decision_tree'], sort_keys=False, default_flow_style=False)
    bins = yaml.dump(state['decision_tree']['bins'], sort_keys=False, default_flow_style=False)

    filter_text = role['system'].format(Strategy=f'{state["paper"]["Strategy Spec"]}')
    text_prompt = (
        f"Decision tree (routing rules):\n{decision}\n"
        f"Bins (candidate files, grouped by topic):\n{bins}\n"
        "Using the decision tree and bins above, respond with ONLY a single JSON object "
        "in exactly this format: {\"file_paths\": [<string>, <string>, ...]}. "
        "Do not respond with a bare array or with raw bin entries."
    )
    filter_prompt = [
        SystemMessage(content=filter_text),
        HumanMessage(text_prompt)
    ]

    response = state['code_agent'].invoke(filter_prompt)
    data = parse_file_paths(response.content)
    return {"messages": [response], 'files': data}

DEF_PATTERN = re.compile(r'^\s*(class|def)\s')

def generate_code(state: AgentState) -> dict:
    git_text = ''
    data = state['data']
    for path in state['files']:
        content = data[FILE_REPO]['files_data'][path]
        matches = [
            f'{i}:{line}'
            for i, line in enumerate(content.splitlines(), start=1)
            if DEF_PATTERN.search(line)
        ]
        git_text += f'# {path}\n' + '\n'.join(matches) + '\n'

    role = state['ROLES']['Code Generator']
    filter_text = role['system'].format(Strategy=f'{state["paper"]["Strategy Spec"]}', code=git_text)

    filter_prompt = [
        SystemMessage(content=filter_text)
    ]

    response = state['code_agent'].invoke(filter_prompt)
    code = strip_code_fences(response.content)

    default_start = 'import os\nfrom dotenv import dotenv_values\nconfig = dotenv_values()\nos.environ["ALPACA_KEY_ID"] = config["ALPACA_KEY_ID"]\nos.environ["ALPACA_SECRET_KEY"] = config["ALPACA_SECRET_KEY"]\n'

    return {"messages": [response], 'context': git_text, 'code': default_start + code}

FENCE_PATTERNS = [
    re.compile(r'^```[^\n]*\n|\n```\s*$'),
    re.compile(r"^'''[^\n]*\n|\n'''\s*$"),
]

def fix_syntax(state: AgentState) -> dict:
    code = state['code']
    role = state['ROLES']['Syntax Fixer']
    filter_text = role['system'].format(code=code)

    filter_prompt = [SystemMessage(content=filter_text)]

    response = state['code_agent'].invoke(filter_prompt)
    code = strip_code_fences(response.content)
    return {"messages": [response], 'code': code}

def evaluate_code(state: AgentState) -> dict:
    path = f'saved_scripts/{state["file_path"]}.py'
    with open(path, 'w') as f:
        f.write(state['code'])
    command = f'python3 {path}'
    response = run_command(state, command)

    return {"messages": [response]}

def route_back(state: AgentState) -> str:
    if state['attempts'] == 4:
        return END
    return END if state['messages'][-1] == 'Success' else 're_evaluate_code'

def re_evaluate_code(state: AgentState) -> dict:
    role = state['ROLES']['Code Fixer']
    code = state['code']
    filter_text = role['system'].format(code=code)
    traceback = state['messages'][-1]
    filter_prompt = [
                    SystemMessage(content=filter_text),
                    HumanMessage(traceback)
    ]

    response = state['code_agent'].invoke(filter_prompt)
    code = strip_code_fences(response.content)
    
    return {"messages": [response], 'code': code, 'attempts': state['attempts'] + 1}

def strip_code_fences(content: str) -> str:
    content = content.strip()
    changed = True
    while changed:
        changed = False
        for pattern in FENCE_PATTERNS:
            stripped = pattern.sub('', content).strip()
            if stripped != content:
                content = stripped
                changed = True
    return content

def parse_file_paths(content: str) -> list:
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data['file_paths']
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, dict) and 'file_paths' in data:
                return data['file_paths']
        except json.JSONDecodeError:
            pass

    paths = re.findall(r'"path"\s*:\s*"([^"]+)"', content)
    seen = []
    for path in paths:
        if path not in seen:
            seen.append(path)
    return seen

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import io
import yaml
import os
import subprocess
from dotenv import dotenv_values
import pickle
import re
from git_fetch import fetch_github_repo
from file_manager import get_all_relevant_files
import filter_papers as fp
import code_agent as ca

GITHUB_REPOS = ['https://github.com/polakowo/vectorbt']
RESEARCH_PATH = 'saved_papers/The Quarter-Hour Effect: Periodic Algorithmic Trading and Return Predictability in Cryptocurrency Futures-2607.09426v1.yaml'

if __name__ == '__main__':
    with open('roles.yaml', 'r') as f:
        ROLES = yaml.safe_load(f)


    files = get_all_relevant_files()
    
    filter_graph = StateGraph(fp.AgentState)

    filter_graph.add_node('user_input_node', fp.user_input_node)
    filter_graph.add_node('update_list', fp.update_list)

    filter_graph.set_entry_point('user_input_node')
    filter_graph.add_conditional_edges('user_input_node', fp.route_back)
    filter_graph.add_edge('update_list', 'user_input_node')

    graph = filter_graph.compile()

    filter_llm = ChatOllama(
        model='llama3.2',
        temperature=0.7
    )
    
    response = graph.invoke({
        'messages': [],
        'files': files,
        'is_satisfied': False,
        'ROLES': ROLES,
        'Filter_Agent': filter_llm,
    })
   
    files = response['files']

    if len(files) == 0:
        print('No found files to backtest')
        exit(0)

    current_env = os.environ.copy()

    config = dotenv_values()
    os.environ["ALPACA_KEY_ID"] = config["ALPACA_KEY_ID"]
    os.environ["ALPACA_SECRET_KEY"] = config["ALPACA_SECRET_KEY"]

    process = subprocess.Popen(
        ['/bin/bash'],
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True,
        env=current_env
    )    

    code_llm = ChatOllama(
        model='qwen2.5-coder',
        temperature=0.1,
        num_ctx=8192
    )
    data = fetch_github_repo(GITHUB_REPOS[0])

    with open('decision_tree.yaml', 'r') as f:
        decision_tree = yaml.safe_load(f)

    code_agent = StateGraph(ca.AgentState)

    code_agent.add_node('router', ca.router)
    code_agent.add_node('generate_code', ca.generate_code)
    code_agent.add_node('fix_syntax', ca.fix_syntax)
    code_agent.add_node('evaluate_code', ca.evaluate_code)
    code_agent.add_node('route_back', ca.route_back)
    code_agent.add_node('re_evaluate_code', ca.re_evaluate_code)


    code_agent.add_edge('router', 'generate_code')
    code_agent.add_edge('generate_code', 'fix_syntax')
    code_agent.add_edge('fix_syntax', 'evaluate_code')
    code_agent.add_conditional_edges('evaluate_code', ca.route_back)
    code_agent.add_edge('re_evaluate_code', 'fix_syntax')

    code_agent.set_entry_point('router')

    graph = code_agent.compile()

    for i, file in enumerate(files):
        with open(f'saved_papers/{file}', 'r') as f:
            file = yaml.safe_load(f)

        file_path = ''
        while True:
            file_path = input('Enter what you would want to name the python file (exclude .py): ')
            if os.path.exists('saved_scripts/' + file_path + '.py'):
                print('Enter a name that does not already exist: ')
                continue
            break
        response = graph.invoke({
            'messages': [],
            'data': data,
            'files': [],
            'paper': file,
            'ROLES': ROLES,
            'decision_tree': decision_tree,
            'code_agent': code_llm,
            'process': process,
            'attempts': 0,
            'idx': i,
            'file_path': file_path
        })
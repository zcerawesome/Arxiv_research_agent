from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
import manage_pdf as mp
from fetch_papers import find_and_score_papers
from AI_Agent import AI_Agent
import io
import yaml
import json
import os
from dotenv import dotenv_values
import pickle

SCORE_LIMIT = 70

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # messages accumulate
    research_goal: str
    paper: dict
    paper_stream: io.BytesIO 
    agent: AI_Agent

relevance_schema = {
    "type": "object",
    "properties": {
        "relevancy": {"type": "boolean"},
        "reason": {"type": "string"},
        "confidence": {"type": "integer"},
    },
    "required": ["relevancy", "reason", "confidence"],
}

def _parse_json_response(content: str) -> dict:
    """Strip optional ```json fences before parsing an LLM JSON reply."""
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()
    return json.loads(content)

# ── Define a node (a step in the graph) ───────────────────────
def filter_papers(state: AgentState) -> AgentState:
    role = ROLES['Relevance Filter']
    filter_text = role['system'].format(research_goal=state['research_goal'])
    pdf_stream = state['paper_stream']
    abstract_section = mp.extract_section(pdf_stream,
                                          'abstract',
                                          'introduction')
    conclusion_section = mp.extract_section(pdf_stream,
                                            'conclusion',
                                            ['acknowledgements', 'references'])
    filter_prompt = [
        SystemMessage(content=filter_text),
        HumanMessage(f'''Here is the data to evaluate\n\n
                         Abstract: {abstract_section}\n\n
                         Conclusion: {conclusion_section}\n\n
                         ''')] 
    response = filter_llm.invoke(filter_prompt)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state['messages'][-1]
    parsed = _parse_json_response(last.content)
    if parsed['relevancy'] == False:
        return END
    return "summarize_findings"

def summarize_findings(state: AgentState) -> AgentState:
    paper_str = mp.stream_to_text(state['paper_stream'])
    role = ROLES['Summarize Results']
    summarize_role = role['system']
    summarize_prompt = [
        SystemMessage(content=summarize_role),
        HumanMessage(f'''Here is the data to evaluate\n\n
                         {paper_str}''')
    ]
    response = state['agent'].invoke(summarize_prompt)
    return {"messages": [response]}

def wrap_by_words(text: str, words_per_line: int = 15) -> str:
    words = text.split(' ')
    lines = [
        " ".join(words[i:i + words_per_line])
        for i in range(0, len(words), words_per_line)
    ]
    return '\n'.join(lines)

def save_paper(paper: dict, result: dict):
    paper_title = paper['title'] + '-' + paper['arxiv_id']

    with open(f'saved_papers/{paper_title}.txt', 'w') as f:
        parsed = _parse_json_response(result["messages"][-1].content)
        key_findings = wrap_by_words('Key Findings:\n' + parsed['Key Findings']) + '\n\n'
        methods = wrap_by_words('Methods:\n' + parsed['Methodology']) + '\n\n'
        critiques = wrap_by_words('Critiques:\n' + parsed['criticisms']) + '\n\n'
        f.writelines([key_findings, methods, critiques])
        f.write(paper['url'])

if __name__ == '__main__':
    with open('roles.yaml', 'r') as f:
        ROLES = yaml.safe_load(f)

    config = dotenv_values()
    os.environ["GOOGLE_API_KEY"] = config["GOOGLE_API_KEY"]

    filter_llm = ChatOllama(
        model="llama3.2",
        temperature=0.7,
        format=relevance_schema,  # constrain Ollama to emit syntactically valid JSON
    )
    builder = StateGraph(AgentState)
    builder.add_node("filter_llm", filter_papers)
    builder.add_node("summarize_findings", summarize_findings)

    builder.add_conditional_edges('filter_llm', should_continue)

    builder.set_entry_point("filter_llm")

    graph = builder.compile()

    gemini_agents_list = [('gemini-2.5-flash', 250_000),
                    ('gemini-2.5-flash-lite', 250_000),
                    ('gemini-3-flash', 250_000),
                    ('gemini-3.1-flash-lite', 250_000),
                    ('gemini-3.5-flash', 250_000)
                    ]
    Gemini_agents = AI_Agent(gemini_agents_list)


    papers, success, total = find_and_score_papers(n=20)

    filtered_papers = []
    for paper in papers:
        if paper['Score'] >= SCORE_LIMIT:
            filtered_papers.append(paper)
    for paper in filtered_papers:
        try:
            result = graph.invoke({
                'research_goal': 'The research goal is to analyze stocks',
                'paper': papers[0],
                'paper_stream': mp.get_pdf_bytes(papers[0]['arxiv_id']),
                'agent': Gemini_agents
            })
            paper_title = paper['title'] + paper['arxiv_id']
            with open(f'saved_papesr/{paper_title}.txt', 'w') as f:
                f.write(result["messages"][-1].content)
                print('Successfully Wrote about ' + paper['title'])
        except:
            continue
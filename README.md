## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
To design and implement a multidocument retrieval agent using LlamaIndex that can efficiently extract and synthesize information from multiple research articles. The system should be capable of handling diverse queries and provide concise, relevant, and accurate responses to support effective information discovery and analysis.

### DESIGN STEPS:
#### STEP 1:
Set up the environment and integrate the LlamaIndex framework for document indexing and retrieval. Collect and preprocess multiple research articles to prepare them for efficient querying.

#### STEP 2:
Develop the retrieval agent to extract and synthesize relevant information from the indexed documents based on user queries. Implement mechanisms to combine insights from multiple sources into concise responses.

#### STEP 3:
Test the retrieval agent with a variety of queries to evaluate its accuracy, relevance, and response quality. Analyze the results to identify strengths and areas for improvement before final deployment.


### PROGRAM:
```py
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=LzPWWPAdY4",
    "https://openreview.net/pdf?id=VTF8yNQM66",
    "https://openreview.net/pdf?id=hSyW5go0v8",
    "https://openreview.net/pdf?id=9WD9KwssyT",
    "https://openreview.net/pdf?id=yV6fD7LYkF",
    "https://openreview.net/pdf?id=hnrB5YHoYu",
    "https://openreview.net/pdf?id=WbWtOYIzIK",
    "https://openreview.net/pdf?id=c5pwL0Soay",
    "https://openreview.net/pdf?id=TpD2aG1h0D"
]

papers = [
    "metagpt.pdf",
    "swebench.pdf",
    "selfrag.pdf",
    "zipformer.pdf",
    "vr_mcl.pdf"
]
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=llm, 
    system_prompt=""" \
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.\

""",
    verbose=True
)
agent = AgentRunner(agent_worker)

response = agent.query(
    "What is the architecture of MetaGPT and how does it differ from traditional LLM agents?"
)
print(str(response))
```

### OUTPUT:
<img width="1068" alt="Screen Shot 1947-02-31 at 08 08 53" src="https://github.com/user-attachments/assets/0dafd6af-c725-4712-81ce-8fe8e0ad63ed" />



### RESULT:
The developed multidocument retrieval agent effectively extracts and synthesizes information from multiple research articles using LlamaIndex. It demonstrates strong performance in delivering concise, relevant, and accurate responses across diverse queries, validating its capability as a reliable tool for comprehensive information retrieval and synthesis.

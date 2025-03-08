
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from dotenv import load_dotenv
from Document import policy_links, fetch_policy_text
import re
import streamlit as st

openai_api_key = st.secrets["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
Settings.embedding = OpenAIEmbedding(model="text-embedding-ada-002", api_key=openai_api_key)

# Define policy links
policy_docs = []
for policy_name, url in policy_links.items():
    print(f"⏳ Ingestion: {policy_name}...")
    
    # Fetch actual policy text using Document.py's function
    policy_text = fetch_policy_text(url)
    
    if policy_text:
        # Create Document with actual content
        policy_docs.append(
            Document(
                text=policy_text,
                metadata={
                    "policy_name": policy_name,
                    "policy_link": url
                }
            )
        )

# Create nodes with actual content
splitter = SentenceSplitter(chunk_size=512)
policy_nodes = splitter.get_nodes_from_documents(policy_docs)

# ======= Rest of RAG setup remains similar ========
# Create indexes
summary_index = SummaryIndex(nodes=policy_nodes)
vector_index = VectorStoreIndex(nodes=policy_nodes)
# Create query engines
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
    similarity_top_k=3  # Retrieve top 3 relevant policy sections
)

vector_query_engine = vector_index.as_query_engine()

# Define tools for routing engine
summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=("Useful for extracting key details from student policies. "
                 "Return a detailed accurate answer in 3-4 lines including references (policy name and URL) to the source policies.")
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=("Useful for retrieving policy documents and links. "
                 )
)


# Create router query engine
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[summary_tool, vector_tool],
    verbose=True,
)

# Intent classification function
def classify_intent(query):
    """
    Returns a list of the 3 most relevant policies from the policy_links
    that match the user query. If the question is too vague or classification
    fails, return an empty list.
    """
    # If the query is too short or obviously vague, return empty immediately
    if len(query.strip()) < 3:
        return []

    llm = OpenAI(model="gpt-3.5-turbo")
    prompt = (
        "You are an expert in university student affairs. "
        "Given the following question, FIRST determine if it relates to any of these 20 policies. "
        "First analyze the question well. If too vague/unclear, respond with 'TooVague'. Otherwise:"
        "1. Identify ALL potentially relevant policies (max 3 most relevant). You MUST choose 3 policies from this list, even if less seem relevant\n"
        "2. you MUST Return EXACTLY 3 comma-separated policy names from this list:\n"
        f"{list(policy_links.keys())}\n\n"
        f"Question: {query}\n"
        "Response format: 'TooVague' or 3 policy names separated by commas"
    )
    response = llm.complete(prompt)
    response_text = response.text.strip()

    if response_text.lower() == 'toovague':
        return []
    
    raw_policies = [p.strip() for p in response_text.split(",")]
    valid_policies = [p for p in raw_policies if p in policy_links]
    
    # Prevent duplicates while filling remaining slots
    while len(valid_policies) < 3:
        remaining = [p for p in policy_links.keys() if p not in valid_policies]
        if not remaining: break
        valid_policies.append(remaining[0])
    
    return valid_policies[:3]

# agnetic_rag_policies.py

def get_detailed_response(query: str) -> str:
    """
    Perform the same steps as the main CLI loop and return
    a single string that includes the entire verbose response.
    """
    # Step 1: Identify relevant policies
    relevant_policies = classify_intent(query)
    if not relevant_policies:
        return "❌ Your query is unclear. Please provide more specific details."
    

    # Step 2: Create focused query
    augmented_query = (
        f"Question: {query}\n\n"
        f"Relevant Policy: {relevant_policies[0]}\n\n"
        
        "You are a university policy expert. Provide:\n"
        "1. Detailed answer with specific requirements\n"
        "2. If there are *specific steps*, summarize them in bullet points"
        "Example Response:\n"
        "Missing 15% of classes results in AF grade"
    )

    # Step 3: Get answer
    response = query_engine.query(augmented_query)
    if not response or not response.response:
        return "❌ No relevant policy found. Try rephrasing your question."

    # Step 4: Find mentioned policies
    response_lower = response.response.lower()
    mentioned_policies = [p for p in relevant_policies if p.lower() in response_lower]

    # Step 5: Select primary policy for linking
    if mentioned_policies:
        primary_policy = mentioned_policies[0]
    else:
        primary_policy = relevant_policies[0]

    # Step 6: Build the multiline string to match your CLI output
    answer_str = (
    "**🤖 AI Response:**\n\n"
    f"**✅ Found Top 3 Relevant Policies:**\n"
    + "".join(f"- 📑 {pol}\n" for pol in relevant_policies)
    + f"\n**⭐ Most Relevant Policy:** {primary_policy}\n\n"
    f"**💡 Answer:**\n{response.response}\n\n"
    #  Here's the key line:
    f"**🔗 Key Policy Reference:** [Read the full {primary_policy} here]({policy_links[primary_policy]})"

)

    return answer_str


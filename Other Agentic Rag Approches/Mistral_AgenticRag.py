# Needs slight Modification 

from llama_index.core import Settings
import os
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv
import time
import pickle

load_dotenv()

# Debug: Check if MISTRAL_API_KEY is loaded
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("❌ MISTRAL_API_KEY is missing! Ensure it's set in your .env file.")

print(f"✅ Using Mistral API Key: {mistral_api_key[:5]}... (hidden for security)")



# Set Mistral Model for LLM and Embeddings
Settings.llm = MistralAI(model="open-mistral-7b")
Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed", batch_size=3)

# Retry function for handling rate limits
def retry_request(func, *args, max_retries=5, base_wait_time=10, cooldown_time=60, **kwargs):
    wait_time = base_wait_time
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e):  # Handle rate limit errors
                print(f"⚠️ Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 1.8  # Adaptive increase to prevent multiple failures
            else:
                raise e

    print("❌ Max retries exceeded. Entering cooldown mode for 60 seconds...")
    time.sleep(cooldown_time)
    return None



# Define policy links
policy_links = {
    "Student Conduct Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",
    "Academic Schedule Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy",
    "Student Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",
    "Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",
    "Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-standing-policy",
    "Transfer Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",
    "Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/admissions-policy",
    "Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",
    "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",
    "Sports and Wellness Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",
    "Scholarship and Financial Assistance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
    "Student Engagement Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-engagement-policy",
    "International Student Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
    "Graduation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
    "Student Counselling Services Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-counselling-services-policy",
    "Graduate Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",
    "Graduate Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-procedure",
    "Graduate Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",
    "Library Space Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy",
    "Examination Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy"
}

# Convert policies into LlamaIndex nodes
policy_texts = [
    Document(
        text=f"{policy} - {link}",
        metadata={"policy_name": policy, "policy_link": link}
    ) for policy, link in policy_links.items()
]


policy_nodes = []

splitter = SentenceSplitter(chunk_size=256)

for doc in policy_texts:
    nodes = splitter.get_nodes_from_documents([doc])
    for node in nodes:
        # Assign metadata to each node
        node.metadata["policy_name"] = doc.metadata["policy_name"]
        node.metadata["policy_link"] = doc.metadata["policy_link"]
        node.metadata["section"] = f"Section {nodes.index(node) + 1}"  # Simulate sections
        policy_nodes.append(node)


# **Batch Embedding to Avoid Rate Limits**
embedding_cache_path = "policy_embeddings.pkl"
embedded_nodes = []  # ✅ Define the variable early to avoid NameError

if os.path.exists(embedding_cache_path):
    print("✅ Using cached embeddings...")
    with open(embedding_cache_path, "rb") as f:
        embedded_nodes = pickle.load(f)
    
    # Ensure cached embeddings match policy nodes
    if len(embedded_nodes) != len(policy_nodes):
        print("⚠️ Mismatch between cached embeddings and policy nodes. Regenerating embeddings...")
        os.remove(embedding_cache_path)
        embedded_nodes = []  # Reset list if mismatch

# If embeddings not available or mismatched, generate them
if not embedded_nodes:
    print("🔄 Generating embeddings in smaller batches...")
    batch_size = 3  

    for i in range(0, len(policy_nodes), batch_size):
        batch = policy_nodes[i:i + batch_size]
        batch_embeddings = retry_request(Settings.embed_model.get_text_embedding_batch, [node.get_text() for node in batch])

        if batch_embeddings is None:
            raise ValueError("❌ Failed to generate embeddings due to rate limit issues.")

        embedded_nodes.extend(batch_embeddings)

    # Save embeddings for future use
    with open(embedding_cache_path, "wb") as f:
        pickle.dump(embedded_nodes, f)

# Assign embeddings to nodes (should now be safe)
if len(embedded_nodes) != len(policy_nodes):
    raise ValueError("❌ Embedding count mismatch after processing. Regenerate embeddings!")

for i, node in enumerate(policy_nodes):
    node.embedding = embedded_nodes[i]




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
    description="Useful for extracting key details from student policies."
)
vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving policy documents and links."
)


# Create router query engine
query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[vector_tool, summary_tool],  # Ensure vector search is first
    verbose=True,
)



# **Updated Classification Prompt**
def classify_intent(query):
    classification_prompt = f"""
    You are an expert **university administrator** at **University of Doha for Science and Technology (UDST)**.
    Your job is to **correctly classify the user's question** based on the university’s **official policies**.

    📜 **List of Available UDST Policies:**
    {', '.join(policy_links.keys())}.

    ---------------------
    **User Query:** {query}
    ---------------------

    **Classification Guidelines:**
    **Select exactly 3 policies** from the provided list that are **most relevant** to answering the question.
    **DO NOT guess or make assumptions**. Use only the policies explicitly provided.
    **DO NOT fabricate policies**—select only from the list.
    **If the question is unclear** or does not match any policy, return: `"UNCLEAR: Need Clarification"`.
    **Return only the policy names** as a **comma-separated list**, with **no extra text, no explanations, and no formatting.**

    **Example Output (Correct):**  
    **"Student Conduct Policy, Examination Policy, Academic Standing Policy"**  

    **Example Output (Incorrect):**  
    - **"The best policies are: Student Conduct Policy, Examination Policy."**  
    - **"Student Conduct Policy, Sports Policy (maybe?), Academic Standing Policy."**  
    - **"I'm not sure, but maybe Student Conduct Policy applies."**  

    Ensure that the response is **strictly formatted** as:  
    **Policy Name, Policy Name, Policy Name**  
    """

    messages = [ChatMessage(role="user", content=classification_prompt)]

    print("⏳ Waiting 10 seconds before querying Mistral to avoid rate limits...")
    time.sleep(10)

    response = retry_request(Settings.llm.chat, messages=messages)

    if response:
        policies = response.message.content.strip().split(",")
        policies = [policy.strip() for policy in policies if policy.strip() in policy_links]  # Ensure only valid policies
        return policies if policies else ["UNCLEAR: Need Clarification"]

    return ["UNCLEAR: Need Clarification"]



if __name__ == "__main__":
    while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break

        relevant_policies = classify_intent(query)
        print(f"Identified relevant policies: {relevant_policies}")

        filtered_nodes = [node for node in policy_nodes if any(policy in node.get_text() for policy in relevant_policies)]

        # **Updated Answer Prompt**
        answer_prompt = f"""
        You are an **AI assistant providing official university policy information** for the **University of Doha for Science and Technology (UDST)**.
        Provide **concise, factual, and policy-based answers**. Do **not assume or generate** extra information.

        **Policies Used for This Answer:**  
        {', '.join(relevant_policies)}

        ---------------------
        **Relevant Policy Sections:**
        {'\n\n'.join([f"🔹 **{node.metadata['policy_name']}**\n{node.get_text()}" for node in filtered_nodes])}
        ---------------------

        **Answering Guidelines:**
        - **Only include policy-backed information**. Do not add assumptions or unrelated facts.
        - If the query requires steps, **use bullet points** for clarity.
        - Always **mention the policy name** used to answer the query.
        - If details are missing, **guide the user to the official policy link**.

        ---------------------
        **User Query:** {query}
        ---------------------

         **Final Answer:**  
        """

        messages = [ChatMessage(role="user", content=answer_prompt)]
        response = retry_request(Settings.llm.chat, messages=messages)

        if response:
            response_text = response.message.content
        else:
            response_text = "⚠️ Unable to retrieve policy details. Please check the official documents."

        print(f"Answer: {response_text}")




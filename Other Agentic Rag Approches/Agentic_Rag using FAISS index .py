import requests
from bs4 import BeautifulSoup
import os
import numpy as np
import faiss
from mistralai import Mistral, UserMessage
import time
import pickle  # For saving embeddings

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    api_key = "MISTRAL_API_KEY"  # Replace with your API key

# Dictionary to map policies to their URLs (UPDATED TO 20 POLICIES)
policy_links = {
    # ✅ Student Affairs Policies and Procedures (Sorted)
    "Student Conduct Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-conduct-policy",  # Pl-ST-01
    "Academic Schedule Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/academic-schedule-policy",  # Pl-ST-02
    "Student Attendance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-attendance-policy",  # Pl-ST-03
    "Student Appeals Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/student-appeals-policy",  # Pl-ST-04
    "Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/academic-standing-policy",  # Pl-ST-05
    "Transfer Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/transfer-policy",  # Pl-ST-06
    "Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/admissions-policy",  # Pl-ST-07
    "Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/final-grade-policy",  # Pl-ST-08
    "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy",  # Pl-ST-09
    "Sports and Wellness Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/sport-and-wellness-facilities-and",  # Pl-ST-10
    "Scholarship and Financial Assistance Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",  # PL-ST-11
    "Student Engagement Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-engagement-policy",  # Pl-ST-12
    "International Student Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",  # Pl-ST-13
    "Graduation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",  # Pl-ST-14
    "Student Counselling Services Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/student-counselling-services-policy",  # Pl-ST-16
    "Graduate Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",  # Pl-ST-17
    "Graduate Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-procedure",  # Pl-ST-18
    "Graduate Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",  # Pl-ST-19
    "Library Space Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/use-library-space-policy",  # Pl-ST-22
    "Examination Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/examination-policy"
}
# ✅ Dynamically Generate URLs List from policy_links
urls = list(policy_links.values())


def fetch_policy_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Remove extra spaces, newlines, and non-ASCII characters
    text = soup.get_text(separator=" ").replace("\n", " ").strip()
    text = " ".join(text.split())  # Normalize extra spaces
    
    return text


policies = {url: fetch_policy_text(url) for url in urls}
#updated

chunk_size = 512
policy_chunks = []
policy_sources = []

for url, text in policies.items():
    chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
    policy_chunks.extend(chunks)
    policy_sources.extend([url] * len(chunks))



def get_text_embedding(list_txt_chunks, batch_size=50):  
    client = Mistral(api_key=api_key)
    embeddings = []

    for i in range(0, len(list_txt_chunks), batch_size):
        batch = list_txt_chunks[i:i + batch_size]

        retries = 5  # Allow 5 retries in case of failure
        while retries > 0:
            try:
                embeddings_batch_response = client.embeddings.create(
                    model="mistral-embed",
                    inputs=batch
                )
                batch_embeddings = [e.embedding for e in embeddings_batch_response.data]
                embeddings.extend(batch_embeddings)
                time.sleep(3)  # Increased delay to 3 sec (reduce risk of rate limit)
                break  # Exit retry loop if successful

            except Exception as e:
                print(f"Error in batch {i}-{i+batch_size}: {e}")
                retries -= 1
                if "429" in str(e):
                    print("Rate limit hit! Waiting for 20 seconds before retrying...")
                    time.sleep(20)  # Increased wait time for API limit

    return embeddings


embeddings_cache_file = "policy_embeddings.pkl"

# Check if embeddings are already stored
if os.path.exists(embeddings_cache_file):
    with open(embeddings_cache_file, "rb") as f:
        embeddings = pickle.load(f)
    print("✅ Loaded cached embeddings.")
else:
    print("🚀 Generating embeddings (this may take time)...")
    text_embeddings = get_text_embedding(policy_chunks)

    if not text_embeddings:
        print("No embeddings were retrieved. Exiting program.")
        exit()

    embeddings = np.array(text_embeddings)

    # Save embeddings to a file
    with open(embeddings_cache_file, "wb") as f:
        pickle.dump(embeddings, f)
    print("✅ Saved embeddings to cache.")

# Create FAISS index
d = len(embeddings[0])
index = faiss.IndexFlatL2(d)
index.add(embeddings)


d = len(embeddings[0])
index = faiss.IndexFlatL2(d)
index.add(embeddings)


def retrieve_relevant_chunks(query):
    query_embedding = np.array([get_text_embedding([query])[0]])
    D, I = index.search(query_embedding, k=6)  # Start with max depth
    
    # Convert FAISS distances to confidence scores
    confidence_scores = 1 / (1 + D[0])  # Normalize scores (higher = better)
    
    # Determine top_k dynamically
    if confidence_scores[0] > 0.9:  # High confidence
        top_k = 2
    elif confidence_scores[0] > 0.7:  # Medium confidence
        top_k = 4
    else:  # Low confidence, expand retrieval
        top_k = 6

    retrieved_chunks = [policy_chunks[i] for i in I[0][:top_k] if i < len(policy_chunks)]
    retrieved_chunks = [chunk for chunk in retrieved_chunks if len(chunk) > 30]  # Ignore cut-off responses
    
    return retrieved_chunks

def generate_response(user_query):
    # *Step 1: Intent Classification (Always 3 Policies)*
    classification_prompt = f"""
    You are a university administrator and expert in student affairs. 
    Given the user's query below, classify which *exactly 3* policies are most relevant.

    The university has the following policies: {', '.join(policy_links.keys())}.

    ---------------------
    User Query: {user_query}
    ---------------------

    - Return exactly *3 policy names*, no more, no less.
    - If the query is unclear, return: "UNCLEAR: Need Clarification".
    - Format output as a *comma-separated list* (e.g., "Library Space Policy, Student Conduct Policy, Examination Policy").
    """

    client = Mistral(api_key=api_key)
    messages = [UserMessage(content=classification_prompt)]

    classification_response = client.chat.complete(
        model="mistral-large-latest",
        messages=messages
    )

    top_policies_text = classification_response.choices[0].message.content.strip()

    # *Check if clarification is needed*
    if "UNCLEAR: Need Clarification" in top_policies_text:
        return (
        "🤔 Your query is too broad. Are you asking about:\n"
        "- Student Conduct Rules 📜\n"
        "- Attendance Requirements ⏳\n"
        "- Academic Regulations 🎓\n"
        "Please clarify!"
    )


    # *Ensure exactly 3 policies are selected*
    top_policies = [p.strip() for p in top_policies_text.split(",") if p.strip() in policy_links]

    if len(top_policies) != 3:
        return "🤔 Your query is broad. Do you mean student conduct, attendance, or academic rules? Please clarify."


    # *Step 2: Retrieve Chunks from the Selected Policies*
    relevant_chunks = []
    for policy in top_policies:
        retrieved = retrieve_relevant_chunks(policy)  # Fetch chunks only from classified policies
        relevant_chunks.extend(retrieved)

    # *Step 3: Generate the Final Answer*
    answer_prompt = f"""
    You are an AI assistant providing official university policy information.
    Provide *concise, to-the-point answers*. Only include the most relevant details.

    *Context from the Top 3 Policies:*  
    ---------------------
    {' '.join(relevant_chunks)}
    ---------------------

    - Answer the user query in *2-3 sentences max*.
    - If there are *specific steps*, summarize them in bullet points.
    - *Mention which policy you are using to answer the question.*
    - If certain details are missing, mention where users can find them.

    User Query: {user_query}
    Answer:
    """

    messages = [UserMessage(content=answer_prompt)]

    final_response = client.chat.complete(
        model="mistral-large-latest",
        messages=messages
    )

    final_answer = final_response.choices[0].message.content.strip()

    # *Step 4: Determine WHICH policies were actually used in the answer*
    used_policies = [policy for policy in top_policies if policy.lower() in final_answer.lower()]

    # *Step 5: Handle Cases Where Information is Missing*
    if "not mentioned" in final_answer.lower() or "not explicitly stated" in final_answer.lower():
        final_answer += (
            "\n\n📢 If this information is not covered in the provided policies, please contact university administration for clarification."
        )

    # *Step 6: Format Response Clearly*
    policy_list_formatted = "\n".join([f"📌 *{p}*" for p in top_policies])  # Shows all 3 classified policies
    used_policy_formatted = "\n".join([f"✅ *Used in Response: {p}*" for p in used_policies])  # Indicates which policy was used
    policy_links_html = "\n".join(
        f"🔗 *More Information:* [Read the full {policy} here]({policy_links[policy]})"
        for policy in used_policies  # Only link policies that were used in the answer
    )

    # *Step 7: Return Final Answer*
    return f"""✅ *The Query is relevant to these policies:*  
{policy_list_formatted}  

{used_policy_formatted}

🎯 *Answer:*  
{final_answer}  

{policy_links_html}"""

if __name__ == "__main__":
    print("📢 Welcome to the University Policy Chatbot! (Type 'exit' to stop)\n")
    
    while True:
        user_query = input("🔍 Enter your query: ")
        
        if user_query.lower() == "exit":
            print("👋 Exiting the chatbot. Have a great day!")
            break
        
        response = generate_response(user_query)
        print("\n🤖 AI Response:\n", response)
        print("\n" + "="*80 + "\n")  # Divider for readability



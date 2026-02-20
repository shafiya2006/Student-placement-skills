import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# =========================
# CONFIG
# =========================
CSV_FILE = r"C:\Users\shafi\OneDrive\Desktop\LLM\Student_Placement_Skills_2025.csv"
DB_LOCATION = "./chroma_student_db"
COLLECTION_NAME = "student_skill_placement"
BATCH_SIZE = 500

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_FILE)

# =========================
# EMBEDDING MODEL
# =========================
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# =========================
# VECTOR STORE
# =========================
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

existing_count = vector_store._collection.count()
print("Existing documents in DB:", existing_count)

# =========================
# PREPARE DOCUMENTS
# =========================
documents = []
ids = []

for i, row in df.iterrows():
    content = (
        f"Student ID {row['Student_ID']} is a {row['Age']} year old {row['Gender']} "
        f"studying {row['Degree']}. CGPA is {row['CGPA']}. "
        f"They completed {row['Internships_Count']} internships, "
        f"{row['Projects_Count']} projects, and hold {row['Certifications_Count']} certifications. "
        f"Technical skill score: {row['Technical_Skills_Score_100']}/100. "
        f"Communication skill score: {row['Communication_Skills_Score_100']}/100. "
        f"Aptitude score: {row['Aptitude_Test_Score_100']}/100. "
        f"Placement offer: {row['Placement_Offer']}. "
        f"Salary offered: {row['Salary_Offered_USD']} USD."
    )

    doc = Document(
        page_content=content,
        metadata={"student_id": row["Student_ID"]},
        id=str(i)
    )

    documents.append(doc)
    ids.append(str(i))

# =========================
# INGEST DATA
# =========================
if existing_count == 0:
    print("Ingesting student documents...")
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]

        vector_store.add_documents(
            documents=batch_docs,
            ids=batch_ids
        )
        print(f"Inserted {i} to {i + len(batch_docs)}")

    print("Final document count:", vector_store._collection.count())
else:
    print("Using existing student database. No re-ingestion needed.")

# =========================
# RETRIEVER
# =========================
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

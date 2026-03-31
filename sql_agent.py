import os
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import warnings

load_dotenv()

warnings.filterwarnings("ignore", category=UserWarning, message="Pydantic serializer warnings")



# ── 1. Structured output schema ───────────────────────────────────────────────

class SQLAgentResponse(BaseModel):
    """Structured response from the NedBank assistant."""
    answer: str = Field(description="The natural language answer to the user's prompt")
    sql: str = Field(description="The sql query answer to the user's prompt")
    confidence: str = Field(
        description="Confidence level based on how well the context supports the answer.",
        pattern="^(high|medium|low)$"
    )
    category: str = Field(
        description=(
            "Category of the question. One of: "
            "'general', 'query', 'unknown'."
        )
    )
    assumptions: Optional[list[str]] = Field(
        default=None,
        description="A list of assumptions the model made when generating the answer, if any."
    )
    needs_clarification: bool = Field(
        description="True if the question is too ambiguous or complex for the model to answer, without further clarification or human intervention."
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="If requires_human_agent is True, this field should contain a follow-up question to ask the user for clarification. Otherwise, it should be null."
    )


# ── 2. RAG Agent ──────────────────────────────────────────────────────────────

class SQLAgent:
    def __init__(self, docs_dir="docs", db_dir="chroma_db"):
        self.docs_dir = docs_dir
        self.db_dir = db_dir
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.structured_llm = self.llm.with_structured_output(SQLAgentResponse)
        self.vector_store = None
        self.chain = None
        self.chat_history = []

    def initialize(self, force_reload=False):
        """Initializes the vector store and the RAG chain."""

        # ── Vector store ──────────────────────────────────────────────────────
        if not os.path.exists(self.db_dir) or force_reload:
            print(f"Building vector store from {self.docs_dir}...")
            loader = DirectoryLoader(self.docs_dir, glob="./*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            splits = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            ).split_documents(documents)
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=self.db_dir
            )
        else:
            print(f"Loading vector store from {self.db_dir}...")
            self.vector_store = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )

        retriever = self.vector_store.as_retriever()

        # ── History-aware retriever (plain LLM — returns a string query) ──────
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question that can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # ── QA prompt + structured LLM ────────────────────────────────────────
        qa_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                    You are a data analyst specialist for a kids clinic called Kidz Kare.
                    You are tasked to convert text to SQL queries, to provide answers to questions related to the clinic operations, services, and policies, based on the provided database schema context.
                    
                    Rules:
                    - The sintax of the SQL code should be appropriate to SQL Server sintax.
                    - Do not include separators in the code like \\n or tabs, the code should be ready to be executed as is.
                    - Use only the tables and columns provided in SCHEMA.
                    - Generate read-only SQL only (SELECT). No INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/TRUNCATE.
                    - If the question is ambiguous or missing filters, set needs_clarification=true and ask a question.
                    - Prefer aggregated queries. Always include a TOP unless the user explicitly requests totals only.
                    - Output must be valid JSON with keys:
                    answer, sql, dialect, confidence, assumptions, needs_clarification, clarification_question
                    
                    Context:
                    {context}
                """
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # ── Manual chain: retriever → format docs → structured LLM ───────────
        def format_docs(inputs: dict) -> dict:
            """Formats retrieved docs into a context string, keeps other keys."""
            docs = inputs["context"]
            inputs["context"] = "\n\n".join(doc.page_content for doc in docs)
            return inputs

        self.chain = (
            RunnablePassthrough.assign(context=history_aware_retriever)
            | RunnableLambda(format_docs)
            | qa_prompt
            | self.structured_llm
        )

    def ask(self, query: str) -> SQLAgentResponse:
        """Returns a structured BankingResponse for the given query."""
        if not self.chain:
            raise ValueError("RAGAgent not initialized. Call initialize() first.")

        result: SQLAgentResponse = self.chain.invoke({
            "input": query,
            "chat_history": self.chat_history,
        })

        self.chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=result.sql),
        ])

        return result


# ── 3. Example usage ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = SQLAgent()
    agent.initialize()

    result = agent.ask("What is the percentage of insurance claims that each insurance company in the year 2024?")
    
    print("\nFull JSON output:")
    print(result.model_dump_json(indent=2))
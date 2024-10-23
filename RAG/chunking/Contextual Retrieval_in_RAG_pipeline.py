### From https://medium.com/the-ai-forum/implementing-contextual-retrieval-in-rag-pipeline-8f1bc7cbd5e0

%pip install langchain langchain-openai openai faiss-cpu python-dotenv rank_bm25  flashrank langchain_groq groq
%pip install -qU langchain-groq
%pip install -qU langchain-community
%pip install -qU sentence-transformers

import hashlib
import os
import getpass
from typing import List, Tuple
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi
from langchain.retrievers import ContextualCompressionRetriever,BM25Retriever,EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.embeddings import HuggingFaceEmbeddings

from google.colab import userdata
import os
os.environ["GROQ_API_KEY"] =userdata.get('GROQ_API_KEY')

class ContextualRetrieval:
    """
    A class that implements the Contextual Retrieval system.
    """

    def __init__(self):
        """
        Initialize the ContextualRetrieval system.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )
        #self.embeddings = OpenAIEmbeddings()

        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        # self.llm = ChatOpenAI(
        #     model="gpt-4o",
        #     temperature=0,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        # )
        self.llm = ChatGroq(
            model="llama-3.2-3b-preview",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    
    def process_document(self, document: str) -> Tuple[List[Document], List[Document]]:
        """
        Process a document by splitting it into chunks and generating context for each chunk.
        """
        chunks = self.text_splitter.create_documents([document])
        contextualized_chunks = self._generate_contextualized_chunks(document, chunks)
        return chunks, contextualized_chunks

    def _generate_contextualized_chunks(self, document: str, chunks: List[Document]) -> List[Document]:
        """
        Generate contextualized versions of the given chunks.
        """
        contextualized_chunks = []
        for chunk in chunks:
            context = self._generate_context(document, chunk.page_content)
            contextualized_content = f"{context}\n\n{chunk.page_content}"
            contextualized_chunks.append(Document(page_content=contextualized_content, metadata=chunk.metadata))
        return contextualized_chunks

    def _generate_context(self, document: str, chunk: str) -> str:
        """
        Generate context for a specific chunk using the language model.
        """
        prompt = ChatPromptTemplate.from_template("""
        You are an AI assistant specializing in financial analysis, particularly for Tesla, Inc. Your task is to provide brief, relevant context for a chunk of text from Tesla's Q3 2023 financial report.
        Here is the financial report:
        <document>
        {document}
        </document>

        Here is the chunk we want to situate within the whole document::
        <chunk>
        {chunk}
        </chunk>

        Provide a concise context (2-3 sentences) for this chunk, considering the following guidelines:
        1. Identify the main financial topic or metric discussed (e.g., revenue, profitability, segment performance, market position).
        2. Mention any relevant time periods or comparisons (e.g., Q3 2023, year-over-year changes).
        3. If applicable, note how this information relates to Tesla's overall financial health, strategy, or market position.
        4. Include any key figures or percentages that provide important context.
        5. Do not use phrases like "This chunk discusses" or "This section provides". Instead, directly state the context.

        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.

        Context:
        """)
        messages = prompt.format_messages(document=document, chunk=chunk)
        response = self.llm.invoke(messages)
        return response.content

    def create_vectorstores(self, chunks: List[Document]) -> FAISS:
        """
        Create a vector store for the given chunks.
        """
        return FAISS.from_documents(chunks, self.embeddings)

    def create_bm25_index(self, chunks: List[Document]) -> BM25Okapi:
        """
        Create a BM25 index for the given chunks.
        """
        tokenized_chunks = [chunk.page_content.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def create_flashrank_index(self,vectorstore):
        """
        Create a FlashRank index for the given chunks.
        """
        retriever = vectorstore.as_retriever(search_kwargs={"k":20})
        compression_retriever = ContextualCompressionRetriever(base_compressor=FlashrankRerank(), base_retriever=retriever)
        return compression_retriever

    def create_bm25_retriever(self, chunks: List[Document]) -> BM25Retriever:
        """
        Create a BM25 retriever for the given chunks.
        """
        bm25_retriever = BM25Retriever.from_documents(chunks)
        return bm25_retriever
    
    def create_ensemble_retriever_reranker(self, vectorstore, bm25_retriever) -> EnsembleRetriever:
        """
        Create an ensemble retriever for the given chunks.
        """
        retriever_vs = vectorstore.as_retriever(search_kwargs={"k":20})
        bm25_retriever.k =10
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever_vs, bm25_retriever],
            weights=[0.5, 0.5]
        )
        redundant_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        #
        reranker = FlashrankRerank()
        pipeline_compressor = DocumentCompressorPipeline(transformers=[redundant_filter, reranker])
        #
        compression_pipeline = ContextualCompressionRetriever(base_compressor=pipeline_compressor,
                                                      base_retriever=ensemble_retriever)
        return compression_pipeline

    @staticmethod
    def generate_cache_key(document: str) -> str:
        """
        Generate a cache key for a document.
        """
        return hashlib.md5(document.encode()).hexdigest()

    def generate_answer(self, query: str, relevant_chunks: List[str]) -> str:
        prompt = ChatPromptTemplate.from_template("""
        Based on the following information, please provide a concise and accurate answer to the question.
        If the information is not sufficient to answer the question, say so.

        Question: {query}

        Relevant information:
        {chunks}

        Answer:
        """)
        messages = prompt.format_messages(query=query, chunks="\n\n".join(relevant_chunks))
        response = self.llm.invoke(messages)
        return response.content
------------------------------------------------------------------------------------------------------
############ Insatiate RAG Pipeline
cr = ContextualRetrieval()

document = """
    Tesla, Inc. (TSLA) Financial Analysis and Market Overview - Q3 2023

    Executive Summary:
    Tesla, Inc. (NASDAQ: TSLA) continues to lead the electric vehicle (EV) market, showcasing strong financial performance and strategic growth initiatives in Q3 2023. This comprehensive analysis delves into Tesla's financial statements, market position, and future outlook, providing investors and stakeholders with crucial insights into the company's performance and potential.

    1. Financial Performance Overview:

    Revenue:
    Tesla reported total revenue of $23.35 billion in Q3 2023, marking a 9% increase year-over-year (YoY) from $21.45 billion in Q3 2022. The automotive segment remained the primary revenue driver, contributing $19.63 billion, up 5% YoY. Energy generation and storage revenue saw significant growth, reaching $1.56 billion, a 40% increase YoY.

    Profitability:
    Gross profit for Q3 2023 stood at $4.18 billion, with a gross margin of 17.9%. While this represents a decrease from the 25.1% gross margin in Q3 2022, it remains above industry averages. Operating income was $1.76 billion, resulting in an operating margin of 7.6%. Net income attributable to common stockholders was $1.85 billion, translating to diluted earnings per share (EPS) of $0.53.

    Cash Flow and Liquidity:
    Tesla's cash and cash equivalents at the end of Q3 2023 were $26.08 billion, a robust position that provides ample liquidity for ongoing operations and future investments. Free cash flow for the quarter was $0.85 billion, reflecting the company's ability to generate cash despite significant capital expenditures.

    2. Operational Highlights:

    Production and Deliveries:
    Tesla produced 430,488 vehicles in Q3 2023, a 17% increase YoY. The Model 3/Y accounted for 419,666 units, while the Model S/X contributed 10,822 units. Total deliveries reached 435,059 vehicles, up 27% YoY, demonstrating strong demand and improved production efficiency.

    Manufacturing Capacity:
    The company's installed annual vehicle production capacity increased to over 2 million units across its factories in Fremont, Shanghai, Berlin-Brandenburg, and Texas. The Shanghai Gigafactory remains the highest-volume plant, with an annual capacity exceeding 950,000 units.

    Energy Business:
    Tesla's energy storage deployments grew by 90% YoY, reaching 4.0 GWh in Q3 2023. Solar deployments also increased by 48% YoY to 106 MW, reflecting growing demand for Tesla's energy products.

    3. Market Position and Competitive Landscape:

    Global EV Market Share:
    Tesla maintained its position as the world's largest EV manufacturer by volume, with an estimated global market share of 18% in Q3 2023. However, competition is intensifying, particularly from Chinese manufacturers like BYD and established automakers accelerating their EV strategies.

    Brand Strength:
    Tesla's brand value continues to grow, ranked as the 12th most valuable brand globally by Interbrand in 2023, with an estimated brand value of $56.3 billion, up 4% from 2022.

    Technology Leadership:
    The company's focus on innovation, particularly in battery technology and autonomous driving capabilities, remains a key differentiator. Tesla's Full Self-Driving (FSD) beta program has expanded to over 800,000 customers in North America, showcasing its advanced driver assistance systems.

    4. Strategic Initiatives and Future Outlook:

    Product Roadmap:
    Tesla reaffirmed its commitment to launching the Cybertruck in 2023, with initial deliveries expected in Q4. The company also hinted at progress on a next-generation vehicle platform, aimed at significantly reducing production costs.

    Expansion Plans:
    Plans for a new Gigafactory in Mexico are progressing, with production expected to commence in 2025. This facility will focus on producing Tesla's next-generation vehicles and expand the company's North American manufacturing footprint.

    Battery Production:
    Tesla continues to ramp up its in-house battery cell production, with 4680 cells now being used in Model Y vehicles produced at the Texas Gigafactory. The company aims to achieve an annual production rate of 1,000 GWh by 2030.

    5. Risk Factors and Challenges:

    Supply Chain Constraints:
    While easing compared to previous years, supply chain issues continue to pose challenges, particularly in sourcing semiconductor chips and raw materials for batteries.

    Regulatory Environment:
    Evolving regulations around EVs, autonomous driving, and data privacy across different markets could impact Tesla's operations and expansion plans.

    Macroeconomic Factors:
    Rising interest rates and inflationary pressures may affect consumer demand for EVs and impact Tesla's profit margins.

    Competition:
    Intensifying competition in the EV market, especially in key markets like China and Europe, could pressure Tesla's market share and pricing power.

    6. Financial Ratios and Metrics:

    Profitability Ratios:
    - Return on Equity (ROE): 18.2%
    - Return on Assets (ROA): 10.3%
    - EBITDA Margin: 15.7%

    Liquidity Ratios:
    - Current Ratio: 1.73
    - Quick Ratio: 1.25

    Efficiency Ratios:
    - Asset Turnover Ratio: 0.88
    - Inventory Turnover Ratio: 11.2

    Valuation Metrics:
    - Price-to-Earnings (P/E) Ratio: 70.5
    - Price-to-Sales (P/S) Ratio: 7.8
    - Enterprise Value to EBITDA (EV/EBITDA): 41.2

    7. Segment Analysis:

    Automotive Segment:
    - Revenue: $19.63 billion (84% of total revenue)
    - Gross Margin: 18.9%
    - Key Products: Model 3, Model Y, Model S, Model X

    Energy Generation and Storage:
    - Revenue: $1.56 billion (7% of total revenue)
    - Gross Margin: 14.2%
    - Key Products: Powerwall, Powerpack, Megapack, Solar Roof

    Services and Other:
    - Revenue: $2.16 billion (9% of total revenue)
    - Gross Margin: 5.3%
    - Includes vehicle maintenance, repair, and used vehicle sales

    8. Geographic Revenue Distribution:

    - United States: $12.34 billion (53% of total revenue)
    - China: $4.67 billion (20% of total revenue)
    - Europe: $3.97 billion (17% of total revenue)
    - Other: $2.37 billion (10% of total revenue)

    9. Research and Development:

    Tesla invested $1.16 billion in R&D during Q3 2023, representing 5% of total revenue. Key focus areas include:
    - Next-generation vehicle platform development
    - Advancements in battery technology and production processes
    - Enhancements to Full Self-Driving (FSD) capabilities
    - Energy storage and solar technology improvements

    10. Capital Expenditures and Investments:

    Capital expenditures for Q3 2023 totaled $2.46 billion, primarily allocated to:
    - Expansion and upgrades of production facilities
    - Tooling for new products, including the Cybertruck
    - Supercharger network expansion
    - Investments in battery cell production capacity

    11. Debt and Capital Structure:

    As of September 30, 2023:
    - Total Debt: $5.62 billion
    - Total Equity: $43.51 billion
    - Debt-to-Equity Ratio: 0.13
    - Weighted Average Cost of Capital (WACC): 8.7%

    12. Stock Performance and Shareholder Returns:

    - 52-Week Price Range: $152.37 - $299.29
    - Market Capitalization: $792.5 billion (as of October 31, 2023)
    - Dividend Policy: Tesla does not currently pay dividends, reinvesting profits into growth initiatives
    - Share Repurchases: No significant share repurchases in Q3 2023

    13. Corporate Governance and Sustainability:

    Board Composition:
    Tesla's Board of Directors consists of 8 members, with 6 independent directors. The roles of CEO and Chairman are separate, with Robyn Denholm serving as Chairwoman.

    ESG Initiatives:
    - Environmental: Committed to using 100% renewable energy in all operations by 2030
    - Social: Focus on diversity and inclusion, with women representing 29% of the global workforce
    - Governance: Enhanced transparency in supply chain management and ethical sourcing of materials

    14. Analyst Recommendations and Price Targets:

    As of October 31, 2023:
    - Buy: 22 analysts
    - Hold: 15 analysts
    - Sell: 5 analysts
    - Average 12-month price target: $245.67

    15. Upcoming Catalysts and Events:

    - Cybertruck production ramp-up and initial deliveries (Q4 2023)
    - Investor Day 2024 (Date TBA)
    - Potential unveiling of next-generation vehicle platform (2024)
    - Expansion of FSD beta program to additional markets

    Conclusion:
    Tesla's Q3 2023 financial results demonstrate the company's continued leadership in the EV market, with strong revenue growth and operational improvements. While facing increased competition and margin pressures, Tesla's robust balance sheet, technological innovations, and expanding product portfolio position it well for future growth. Investors should monitor key metrics such as production ramp-up, margin trends, and progress on strategic initiatives to assess Tesla's long-term value proposition in the rapidly evolving automotive and energy markets.
    """

original_chunks, contextualized_chunks = cr.process_document(document)
original_vectorstore = cr.create_vectorstores(original_chunks)
contextualized_vectorstore = cr.create_vectorstores(contextualized_chunks)

original_bm25_index = cr.create_bm25_index(original_chunks)
contextualized_bm25_index = cr.create_bm25_index(contextualized_chunks)

original_reranker = cr.create_flashrank_index(original_vectorstore)
contextualized_reranker = cr.create_flashrank_index(contextualized_vectorstore)

contextualized_reranker.invoke("What was Tesla's total revenue in Q3 2023? what was the gross profit and cash position?")
-------------------------------------------------------------------------------------------------------------------------------
########### create retriver system with hybrid search with Reranker
# Crete ensemble retriver reranker
bm25_retriever_original = cr.create_bm25_retriever(original_chunks)
#
bm25_retriever_contextualized = cr.create_bm25_retriever(contextualized_chunks)
#
original_ensemble_retriever_reranker = cr.create_ensemble_retriever_reranker(original_vectorstore, bm25_retriever_original)
#
contextualized_ensemble_retriever_reranker = cr.create_ensemble_retriever_reranker(contextualized_vectorstore, bm25_retriever_contextualized)
-------------------------------------------------------------------------------------------------------------------------------
########## retrieve context from hybrid retriever
contextualized_ensemble_retriever_reranker.invoke("What was Tesla's total revenue in Q3 2023? what was the gross profit and cash position?")
-------------------------------------------------------------------------------------------------------------------------------
########## retrieve context from hybrid retriever
cache_key = cr.generate_cache_key(document)

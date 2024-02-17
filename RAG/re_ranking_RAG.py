## From https://pub.towardsai.net/advanced-rag-04-re-ranking-85f6ae8170b1

import os
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_KEY"

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.schema import QueryBundle

dir_path = "YOUR_DIR_PATH"
##### Build simple vector retriever
...

##### Build display_source_node function
from llama_index.schema import ImageNode, MetadataMode, NodeWithScore
from llama_index.utils import truncate_text

def display_source_node(
    source_node: NodeWithScore,
    source_length: int = 100,
    show_source_metadata: bool = False,
    metadata_mode: MetadataMode = MetadataMode.NONE,
) -> None:
    """Display source node"""
    source_text_fmt = truncate_text(
        source_node.node.get_content(metadata_mode=metadata_mode).strip(), source_length
    )
    text_md = (
        f"Node ID: {source_node.node.node_id} \n"
        f"Score: {source_node.score} \n"
        f"Text: {source_text_fmt} \n"
    )
    if show_source_metadata:
        text_md += f"Metadata: {source_node.node.metadata} \n"
    if isinstance(source_node.node, ImageNode):
        text_md += "Image:"

    print(text_md)

####### Add re-ranking 
print('------------------------------------------------------------------------------------------------')
print('Start reranking...')

reranker = FlagEmbeddingReranker(
    top_n = 3,
    model = "BAAI/bge-reranker-base",
)

query_bundle = QueryBundle(query_str=query)
ranked_nodes = reranker._postprocess_nodes(nodes, query_bundle = query_bundle)
for ranked_node in ranked_nodes:
    print('----------------------------------------------------')
    display_source_node(ranked_node, source_length = 500)


######## Add GPT re-ranking
from llama_index.postprocessor import RankGPTRerank
from llama_index.llms import OpenAI
reranker = RankGPTRerank(
    top_n = 3,
    llm = OpenAI(model="gpt-3.5-turbo-16k"),
    # verbose=True,
)
query_bundle = QueryBundle(query_str=query)
ranked_nodes = reranker._postprocess_nodes(nodes, query_bundle = query_bundle)
for ranked_node in ranked_nodes:
    print('----------------------------------------------------')
    display_source_node(ranked_node, source_length = 500)

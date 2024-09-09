## From https://medium.com/google-developer-experts/efficient-rag-for-mixed-context-texts-with-indexifys-framework-gemini-s-1m-context-arctic-s-df2882aad576
## Mixed-context texts, such as research papers, technical documents, or even web pages, often contain cross-domain information

!pip install -q -U indexify indexify-extractor-sdk
curl https://getindexify.ai | sh
./indexify server -d

!indexify-extractor download hub://pdf/marker
!indexify-extractor download hub://text/llm
!indexify-extractor download hub://text/chunking
!indexify-extractor download hub://embedding/arctic

!indexify-extractor join-server

from indexify import IndexifyClient
client = IndexifyClient()

from indexify import ExtractionGraph

extraction_graph_spec = """
name: 'llmarrag'
extraction_policies:
   - extractor: 'tensorlake/marker'
     name: 'mdextractor'
   - extractor: 'tensorlake/llm'
     name: 'txtprocessor'
     input_params:
        service: 'gemini'
        prompt: 'Rearrange and rewrite the following text by grouping similar topics together while preserving the original sentences.'
     content_source: 'mdextractor'
   - extractor: 'tensorlake/chunk-extractor'
     name: 'chunker'
     input_params:
        chunk_size: 1000
        overlap: 100
     content_source: 'txtprocessor'
   - extractor: 'tensorlake/arctic'
     name: 'embedder'
     content_source: 'chunker'
"""

extraction_graph = ExtractionGraph.from_yaml(extraction_graph_spec)
client.create_extraction_graph(extraction_graph)

client.upload_file("llmarrag", "random_topics.pdf")



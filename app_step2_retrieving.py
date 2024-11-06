"""
The focus for this script is on the retrieval of the best matching chunks from the
content store and the construction of the context for an LLM to answer the question
using retrieval strategies. We use Weaviate to experiment with different retrieval
strategies. Below is a summary of the TODOs:

TODO 1: Experiment with different retrieval strategies
TODO 2: Use different splitting methods through Weaviate collections
TODO 3: Find the best strategy to answer a specific question

Use the blocks between BEGIN SOLUTION and END SOLUTION to complete the TODOs.
"""
import logging

from dotenv import load_dotenv
from rag4p.integrations.openai.openai_answer_generator import OpenaiAnswerGenerator
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.integrations.weaviate.weaviate_retriever import WeaviateRetriever
from rag4p.rag.generation.answer_generator import AnswerGenerator
from rag4p.rag.retrieval.strategies.document_retrieval_strategy import DocumentRetrievalStrategy
from rag4p.rag.retrieval.strategies.hierarchical_retrieval_strategy import HierarchicalRetrievalStrategy
from rag4p.rag.retrieval.strategies.topn_retrieval_strategy import TopNRetrievalStrategy
from rag4p.rag.retrieval.strategies.window_retrieval_strategy import WindowRetrievalStrategy
from rag4p.util.key_loader import KeyLoader


def main():
    weaviate_collections = [
        "JfallOpenAiSentence",
        "JfallOpenAiMaxToken",
        "JfallOpenAiSemantic",
        "JfallOpenAiMaxTokenSentence"  # Used a SplitterChain with MaxTokenSplitter and SentenceSplitter
    ]
    question = "What is the workshop about?"
    retriever = create_weaviate_retriever(weaviate_collections[0])

    # TODO 1:Check the TODOs inside the retrieve_context method
    context = retrieve_context(retriever=retriever, question=question)
    logger.info(f"Context: {context}")

    retriever.weaviate_access.close()

    # TODO 2: Try different collections from Weaviate on line 36. These collections use different splitters:
    #  - JfallOpenAiSentence uses the SentenceSplitter
    #  - JfallOpenAiMaxToken uses the MaxTokenSplitter
    #  - JfallOpenAiSemantic uses the SemanticSplitter
    #  - JfallOpenAiMaxTokenSentence uses a SplitterChain with MaxTokenSplitter and SentenceSplitter.
    #    This works well with the HierarchicalRetrievalStrategy

    # TODO 3: Remove the comment below and select the right strategy to get an answer to the
    #  following question:  "Who are speaking about RAG?"
    # answer = retrieve_answer(context=context, question=question)
    # logger.info(f"Answer: {answer}")


def retrieve_context(retriever, question):
    # TODO 1: Experiment with different retrieval strategies and pay attention to the results
    #  in the logs. Use the following retrieval strategies:
    #  - TopNRetrievalStrategy
    #  - WindowRetrievalStrategy
    #  - DocumentRetrievalStrategy
    #  - HierarchicalRetrievalStrategy
    strategy = None
    # BEGIN SOLUTION

    # END SOLUTION
    result = strategy.retrieve_max_results(question=question, max_results=2)

    for chunk in result.items:
        logger.info(f"Chunk id: {chunk.chunk_id}")
        logger.info(f"Text: {chunk.text}")
        logger.info("--------------------------------------------------")

    return result.construct_context()


def retrieve_answer(context: str, question: str) -> str:
    answer_generator = OpenaiAnswerGenerator(openai_api_key=key_loader.get_openai_api_key(), openai_model="gpt-4o")
    answer = answer_generator.generate_answer(question=question, context=context)

    answer_generator.openai_client.close()
    return answer


def create_weaviate_retriever(collection_name: str):
    client = AccessWeaviate(
        url=key_loader.get_weaviate_url(),
        access_key=key_loader.get_weaviate_api_key()
    )

    openai_embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    return WeaviateRetriever(
        weaviate_access=client,
        embedder=openai_embedder,
        additional_properties=["title", "time", "room", "speakers", "tags"],
        hybrid=False,
        collection_name=collection_name)


if __name__ == '__main__':
    # Load environment variables from .env file
    load_dotenv()
    key_loader = KeyLoader()

    # Configure logging
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    main()

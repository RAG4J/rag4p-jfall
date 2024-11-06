"""
In the previous step you looked at the quality of the retriever. In this step,
you will look at the quality of the answer. The quality of the answer is
determined by the quality of the answer related to the question and the quality
of the answer related to the context. For both cases we use Ollama's LLM to
determine the quality of the answer.

TODO 1: Understand the mechanism to determine the quality of the answer using one document
TODO 2: Use all the documents in the Weaviate collection to determine the quality of the answer
"""

import logging

from dotenv import load_dotenv

from rag4p.integrations.ollama import MODEL_LLAMA3
from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.integrations.ollama.ollama_answer_generator import OllamaAnswerGenerator
from rag4p.integrations.ollama.ollama_embedder import OllamaEmbedder
from rag4p.integrations.ollama.quality.ollama_answer_quality_service import OllamaAnswerQualityService
from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.integrations.weaviate.weaviate_retriever import WeaviateRetriever
from rag4p.rag.generation.observed_answer_generator import ObservedAnswerGenerator
from rag4p.rag.retrieval.strategies.document_retrieval_strategy import DocumentRetrievalStrategy
from rag4p.rag.tracker.rag_tracker import global_data
from rag4p.util.key_loader import KeyLoader


def main():
    # TODO 1: Run the code and inspect the quality of the answer
    question = "What session does this text describe?"
    context = src_text

    print_quality_of_answer(question=question, context=context)

    #  TODO 2: Create a context that can be used to answer the question,
    #   use Weaviate to retrieve the most relevant document, and
    #   construct the context  using an appropriate strategy and determine
    #   the quality of the answer. Use the method print_quality_of_answer
    #   to print the results.

    retriever = create_weaviate_retriever("JfallOllamaMaxToken")
    question = "Who are speaking about RAG?"
    context = None
    # BEGIN SOLUTION
    # strategy = None
    # context = None
    # print_quality_of_answer(question=question, context=context)
    # END SOLUTION
    retriever.weaviate_access.close()
    # TODO 3: Ask other questions and determine the quality of the answers
    # TODO 4: Change the collection in the retriever and determine the quality of the answers
    # TODO 5: Inspect the prompt to determine the quality of the answer, you can try to improve it
    #  use methods quality_of_answer_to_question_system_prompt and
    #  quality_of_answer_from_context_system_prompt


def print_quality_of_answer(question: str, context: str):
    # Initialize the answer generator, we use the OllamaAnswerGenerator
    ollama_answer_generator = OllamaAnswerGenerator(access_ollama=access_ollama)

    # The observer wraps the answer generator to capture the generated answers
    answer_generator = ObservedAnswerGenerator(answer_generator=ollama_answer_generator)

    # Ask for the answer to the provided question given the provided context
    answer = answer_generator.generate_answer(question, context)

    logger.info(f"Answer: {answer}")

    # Gain access to the observer to retrieve the generated answers
    rag_observer = global_data["observer"]

    # Initialize the answer quality service
    answer_quality_service = OllamaAnswerQualityService(
        access_ollama=access_ollama,
        model=MODEL_LLAMA3
    )

    # Determine the quality of the answer related to the question
    quality = answer_quality_service.determine_quality_answer_related_to_question(
        rag_observer=rag_observer
    )
    logger.info(f"Quality answer -> question: {quality.quality}, Reason: {quality.reason}")

    # Determine the quality of the answer related to the context
    quality = answer_quality_service.determine_quality_answer_from_context(
        rag_observer=rag_observer
    )
    logger.info(f"Quality answer -> context: {quality.quality}, Reason: {quality.reason}")

    rag_observer.reset()


def create_weaviate_retriever(collection_name: str):
    client = AccessWeaviate(
        url=key_loader.get_weaviate_url(),
        access_key=key_loader.get_weaviate_api_key()
    )

    embedder = OllamaEmbedder(access_ollama=access_ollama)
    return WeaviateRetriever(
        weaviate_access=client,
        embedder=embedder,
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

    access_ollama = AccessOllama()

    src_text = ("Generative AI is here to stay. Tools to generate text, "
                "images, or data are now common goods. Large Language "
                "models (LLMs) only have the knowledge they acquired "
                "through learning, and even that knowledge does not include "
                "all the details. To overcome the knowledge problem, the "
                "Retrieval Augmented Generation (RAG) pattern arose. An "
                "essential part of RAG is the retrieval part. Retrieval is not new. "
                "The search or retrieval domain is rich with tools, metrics and "
                "research. The new kid on the block is semantic search using "
                "vectors. Vector search got a jump start with the rise of LLMs "
                "and RAG.\n\nThis workshop aims to build a high-quality "
                "retriever, integrate the retriever into your LLM solution and "
                "measure the overall quality of your RAG system.\n\n"
                "The workshop uses our Rag4j/Rag4p framework, which we "
                "created especially for workshops. It is easy to learn, so you "
                "can focus on understanding and building the details of the "
                "components during the workshop. You experiment with "
                "different chunking mechanisms (sentence, max tokens, "
                "semantic). After that, you use various strategies to construct "
                "the context for the LLM (TopN, Window, Document, "
                "Hierarchical). To find the optimum combination, you'll use "
                "quality metrics for the retriever as well as the other "
                "components of the RAG system.\n\nYou can do the "
                "workshop using Python or Java. We provide access to a "
                "remote LLM. You can also run an open-source LLM on "
                "Ollama on your local machine.")
    main()

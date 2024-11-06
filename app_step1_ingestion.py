"""
This script demonstrates the basic usage of the Rag4p framework.
Use the main() method to run the script. The script contains a number
of TODOs that you need to complete. Below is a summary of the TODOs:

TODO 1: Check the results of the different basic splitters.
TODO 2: Use the SemanticSplitter
TODO 3: Initialize the content store and retrieve best matching chunks
TODO 4: Use a retrieval strategy to answer a question using the context

Use the blocks between BEGIN SOLUTION and END SOLUTION to complete the TODOs.
"""

import logging

from dotenv import load_dotenv
from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitter_chain import SplitterChain
from rag4p.indexing.splitters.max_token_splitter import MaxTokenSplitter
from rag4p.indexing.splitters.section_splitter import SectionSplitter
from rag4p.indexing.splitters.semantic_splitter import SemanticSplitter
from rag4p.indexing.splitters.sentence_splitter import SentenceSplitter
from rag4p.integrations.openai.openai_answer_generator import OpenaiAnswerGenerator
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.openai.openai_knowledge_extractor import OpenaiKnowledgeExtractor
from rag4p.rag.model.chunk import Chunk
from rag4p.rag.retrieval.strategies.topn_retrieval_strategy import TopNRetrievalStrategy
from rag4p.rag.store.local.internal_content_store import InternalContentStore
from rag4p.util.key_loader import KeyLoader


def main():
    # Initialize the source document from the src_text variable
    src_doc = InputDocument(
        document_id="jfall-talk-jettro-daniel",
        text=src_text,
        properties={})

    # Split the source document into chunks using the basic splitting method
    # TODO 1: Check TODO in the basic_splitting method
    chunks = basic_splitting(src_doc)

    # Split the source document into chunks using the semantic splitting method
    # TODO 2: Check TODO in the semantic_splitting method and uncomment the line below
    # TODO 2: Notice the results of the semantic splitter and the difference with the basic splitter
    # TODO 2: Check the prompt in the class OpenaiKnowledgeExtractor, you can try to improve it.
    # chunks = semantic_splitting(src_doc)

    # Print the chunks
    for chunk in chunks:
        logger.info(f'Chunk id: {chunk.chunk_id}, text: {chunk.chunk_text}')

    # Initialize the content store and retrieve best matching chunks
    # TODO 3: Uncomment the code below and check the results
    # TODO 3: comment the semantic splitting and use basic splitting with your preferred splitter
    # TODO 3: Check the results of the retrieval, notice the scores of the chunks
    # TODO 3: Try different splitters and check the results
    # content_store = init_content_store(chunks=chunks)
    # retrieve_and_print(question="What is the workshop about?", content_store=content_store)

    # Use a retrieval strategy to find the most relevant chunks for a
    # given question and answer it using the LLM
    # TODO 4: Uncomment the code below and check the results
    # TODO 4: Notice the answer, try the sentence and the section splitter
    # TODO 4: Look at the prompt in the OpenaiAnswerGenerator, you can try to improve it.
    # answer = rag(question="What is the workshop about?", content_store=content_store)
    # logger.info(f'Answer: {answer}')


def basic_splitting(input_doc: InputDocument) -> list[Chunk]:
    """
    Split the input document into chunks using a basic splitter. You
    create the splitter and call the split method on the splitter.
    """
    logger.info('Starting the basic splitting application...')

    # Use a splitter to create chunks from the source document
    splitter = None
    # TODO 1: Check the results of the different splitters. Look at the chunk ids and the text
    #  of the chunks.
    #  - Initialize the sentence splitter,
    #  - Replace the splitter with a MaxTokensSplitter of 100 tokens
    #  - Replace the splitter with a SplitterChain containing the following
    #  splitters: SectionSplitter, SentenceSplitter
    # BEGIN SOLUTION
    splitter = None
    # END SOLUTION

    return splitter.split(input_doc)


def semantic_splitting(input_doc: InputDocument) -> list[Chunk]:
    """
    Split the input document into chunks using a semantic splitter. You
    create the splitter and call the split method on the splitter.
    """
    logger.info('Starting the semantic splitting application...')

    # Use a splitter to create chunks from the source document
    knowledge_extractor = OpenaiKnowledgeExtractor(
        openai_api_key=key_loader.get_openai_api_key()
    )

    # TODO 2: Use the SemanticSplitter to split the source document
    #  into chunks. Change the main method to call this method.
    # BEGIN SOLUTION
    splitter = None
    # END SOLUTION

    return splitter.split(input_doc)


def init_content_store(chunks: list[Chunk]) -> InternalContentStore:
    """
    Use the provided chunks to initialize the internal content store. The
    store uses an embedder to create embeddings for the chunks. The
    store uses an in memory storage to store the chunks.
    """
    # Create the content store from the provided chunks
    embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    content_store = InternalContentStore(embedder=embedder)
    content_store.store(chunks=chunks)

    return content_store


def retrieve_and_print(question: str, content_store: InternalContentStore):
    """
    Retrieve the most relevant chunks for the provided question and print
    the results.
    """
    # Find the most relevant chunk for the provided question
    relevant_chunks = content_store.find_relevant_chunks(
        query=question, max_results=2)

    # Print the relevant chunks
    for chunk in relevant_chunks:
        logger.info(f'Chunk id: {chunk.chunk_id}, text: {chunk.chunk_text},'
                    f' score: {chunk.score:.3f}')


def rag(question: str, content_store: InternalContentStore) -> str:
    """
    Use a retrieval strategy to find the most relevant chunks for the
    provided question and answer it using the LLM.
    """
    # Create a retrieval strategy to find the most relevant chunks
    strategy = TopNRetrievalStrategy(retriever=content_store)

    # Retrieve the most relevant chunks for the provided question
    results = strategy.retrieve_max_results(question=question, max_results=2)

    # Answer the question using the LLM answer generator
    answer_generator = OpenaiAnswerGenerator(openai_api_key=key_loader.get_openai_api_key(), openai_model="gpt-4o")
    return answer_generator.generate_answer(question=question, context=results.construct_context())


if __name__ == '__main__':
    # Load environment variables from .env file
    load_dotenv()
    key_loader = KeyLoader()

    # Configure logging
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

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

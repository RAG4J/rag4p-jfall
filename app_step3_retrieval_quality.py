"""
In this script, you explore the quality of the retriever. You use an LLM to generate
questions for all chunks in the document. The question should be answered by the
chunk it was generated from. This combination of questions and answers is used to
create a judgement list. The quality of the retriever is determined by comparing the
expected answers to the actual answers. Below is a summary of the TODOs in this script:

TODO 1: Learn the concepts with just one document
TODO 2: Use the judgement list for all chunks to determine the quality of the retriever

"""

import logging

from dotenv import load_dotenv
from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitters.section_splitter import SectionSplitter
from rag4p.integrations.openai import MODEL_GPT4O_MINI
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.openai.openai_question_generator import OpenAIQuestionGenerator
from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.integrations.weaviate.weaviate_retriever import WeaviateRetriever
from rag4p.rag.generation.question_generator_service import QuestionGeneratorService
from rag4p.rag.retrieval.quality.retrieval_quality_service import read_question_answers_from_file, \
    obtain_retrieval_quality
from rag4p.rag.store.local.internal_content_store import InternalContentStore
from rag4p.util.key_loader import KeyLoader


def main():
    # The next function call creates questions for all chunks in the document.
    # After that, the quality of the retriever is determined.
    # TODO 1: Run the script and check the quality of the retriever
    # TODO 1: Change the splitter to the sentence splitter and check the quality
    # TODO 1: Inspect the prompt to generate the questions in class OpenAIQuestionGenerator
    #  You can try to improve it.
    quality_for_single_document()

    # TODO 2: comment the runner for a single document and uncomment
    #  the runner for all documents
    # TODO 2: Inspect the contents of the file jfall_questions_answers.csv
    # TODO 2: Run the script and check the quality of the retriever
    # quality_for_all_documents()


def quality_for_single_document():
    # Initialize the source document from the src_text variable
    src_doc = InputDocument(
        document_id="jfall-talk-jettro-daniel",
        text=src_text,
        properties={})

    # Initialize the local content store
    splitter = SectionSplitter()
    embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    store = InternalContentStore(embedder=embedder)
    chunks = splitter.split(input_document=src_doc)
    store.store(chunks=chunks)

    # Generate questions for all chunks in the document
    question_generator = OpenAIQuestionGenerator(
        openai_api_key=key_loader.get_openai_api_key(),
        openai_model=MODEL_GPT4O_MINI
    )

    # Wrap the generator in a service with access to the content store
    question_generator_service = QuestionGeneratorService(
        retriever=store,
        question_generator=question_generator
    )

    # Generate the questions for all the chunks in the content store and
    # write them to a csv file as our judgement list
    question_generator_service.generate_question_answer_pairs(
        file_name="jfall_questions_answers_sample.csv"
    )

    # Read the judgement list that we have generated
    question_answer_records = read_question_answers_from_file(
        file_name="./data/jfall_questions_answers_sample.csv"
    )

    # Use the judgement list to determine the quality of the retriever
    retriever_quality = obtain_retrieval_quality(
        question_answer_records=question_answer_records,
        retriever=store
    )

    logger.info(f"Quality using precision: {retriever_quality.precision()}")
    logger.info(f"Total questions: {retriever_quality.total_questions()}")


def quality_for_all_documents():
    logger.info("Determining the quality of the retriever for all documents")

    # Initialize the Weaviate client
    client = AccessWeaviate(
        url=key_loader.get_weaviate_url(),
        access_key=key_loader.get_weaviate_api_key()
    )

    # Initialize the Weaviate retriever with the OpenAI embedder
    openai_embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    retriever = WeaviateRetriever(
        weaviate_access=client,
        embedder=openai_embedder,
        additional_properties=["title", "time", "room", "speakers", "tags"],
        hybrid=False,
        collection_name="JfallOpenAiMaxToken"
    )

    question_answer_records = read_question_answers_from_file(
        file_name="./data/jfall/jfall_questions_answers.csv"
    )

    logger.info(f"Number of question answer records: {len(question_answer_records)}")

    # Use the judgement list to determine the quality of the retriever
    retriever_quality = obtain_retrieval_quality(
        question_answer_records=question_answer_records,
        retriever=retriever
    )

    client.close()

    logger.info(f"Quality using precision: {retriever_quality.precision()}")
    logger.info(f"Total questions: {retriever_quality.total_questions()}")


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

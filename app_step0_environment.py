"""
This scripts checks the setup of your environment. It checks if you
created the .env file and if the OpenAI API key is available. Like the
readme tells you, there are two ways to set the OpenAI API key. You
can use the secret_key or use your own OPEN_AI key.

The script will print a message if the key is found or not. If the key
is not found, the script will print the exception that is raised. This
way you can see what went wrong.
"""
import logging

from dotenv import load_dotenv
from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.util.key_loader import KeyLoader


def main():
    try:
        key_loader.get_openai_api_key()
        logger.info(f"Found the key, your environment is set up correctly")

        client = AccessWeaviate(
            url=key_loader.get_weaviate_url(),
            access_key=key_loader.get_weaviate_api_key()
        )
        client.print_meta()
        client.close()

    except Exception as e:
        logger.error("Problem loading environment, check your .env file")
        logger.error(f"Problem is of type: {type(e).__name__}")

if __name__ == '__main__':
    # Load environment variables from .env file
    load_dotenv()
    key_loader = KeyLoader()

    # Configure logging
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    main()
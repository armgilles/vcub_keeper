import os
import shutil
import time

from dotenv import load_dotenv
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_mistralai import MistralAIEmbeddings
from langchain_unstructured import UnstructuredLoader

from vcub_keeper.config import ROOT_DATA_LLM

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
HG_TOKEN = os.getenv("HG_TOKEN")  # For MistralAI embeddings to DL from HuggingFace


def get_documents_about_project(path_directory: str, file_name: str = "about_project.md") -> list:
    """
    Read the content of a markdown file and split it into sections based on headers.

    Parameters
    ----------
    path_directory : str
        The directory where the markdown file is located.
    file_name : str, optional
        The name of the markdown file to read, by default "about_project.md".

    Returns
    --------
    list
        A list of sections from the markdown file, split by headers.

    Examples
    --------
    sections = get_documents_about_project(path_directory=ROOT_DATA_LLM, file_name="about_project.md")
    """

    file_path = os.path.join(path_directory, file_name)

    with open(file_path, encoding="utf-8") as file:
        content = file.read()

    # Initialize the MarkdownHeaderTextSplitter with a specific header level
    headers_to_split_on = [
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # Split the content into sections based on headers
    sections = markdown_splitter.split_text(content)

    return sections


def get_documents_from_website() -> list:
    """
    Read the content of a site and return filtered documents based on metadata.

    Parameters
    ----------
    None

    Return
    --------
    web_docs : list

    Example
    -------
    web_docs = get_documents_from_website()
    """

    loader = UnstructuredLoader(web_url="https://www.infotbm.com/fr/v3-comment-ca-marche.html")

    docs = []
    for doc in loader.load():
        docs.append(doc)

    # TODO : A voir si on garde cette partie
    # Filter documents to include only those with "emphasized_text_contents" in metadata
    # filtered_docs = [doc for doc in docs if "emphasized_text_contents" in doc.metadata]

    return docs


def create_vector_store_with_embed(path_to_db: str, force_to_recreate_vector_base: bool = False) -> Chroma:
    """
    Permets de créer un vecteur store avec les embeddings de Mistral

    Possible issue lors de la création de la base vectorielle :
        https://github.com/langchain-ai/langchain/issues/14872
        https://github.com/chroma-core/chroma/issues/3635

    Parameters
    ----------
    path_to_db : str
        Path to the database to persist the vector store.

    Returns
    --------
    vector_store : Chroma

    Example
    -------
    vector_store = create_vector_store_with_embed(path_to_db=ROOT_DATA_LLM)
    """

    # Load the MistralAI embeddings
    embeddings = MistralAIEmbeddings(api_key=MISTRAL_API_KEY)

    # Chemin vers le répertoire de persistance
    persist_directory = f"{path_to_db}chroma_langchain_db"

    if force_to_recreate_vector_base:
        delete_vector_store(path_to_db=path_to_db)

    # Vérifier si le répertoire contient une base existante
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Base vectorielle existante trouvée. Chargement...")
        vector_store = Chroma(
            collection_name="vcub_keeper_vector", embedding_function=embeddings, persist_directory=persist_directory
        )
    # TODO : Redondance de code
    else:
        print("Aucune base vectorielle existante trouvée. Création d'une nouvelle...")
        vector_store = Chroma(
            collection_name="vcub_keeper_vector", embedding_function=embeddings, persist_directory=persist_directory
        )

    print(f"Nombre de document dans la base vectorielle : {vector_store._collection.count()}")

    return vector_store


def delete_vector_store(path_to_db: str) -> None:
    """
    Permets de supprimer la base vectorielle Chroma si elle existe.

    Parameters
    ----------
    path_to_db : str

    Returns
    -------
    None

    Example
    -------
    delete_vector_store(path_to_db=ROOT_DATA_LLM)
    """

    # Chemin vers le répertoire de persistance
    persist_directory = f"{path_to_db}chroma_langchain_db"

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        time.sleep(1)
        print(f"Base vectorielle supprimée : {persist_directory}")
    else:
        print(f"Aucune base vectorielle trouvée à supprimer : {persist_directory}")


def build_retriever_rag(
    usual_number_of_docs: int = 49,
    force_rebuild_vector_store: bool = False,
    path_to_llm_dir: str = ROOT_DATA_LLM,
) -> VectorStoreRetriever:
    """
    Permets la création d'un retriever pour la recherche de documents à partir d'une base vectorielle
    Chroma. Cette base peut être chargé ou recréé si elle n'existe pas ou si l'option
    force_rebuild_vector_store est activée. Dans le cas ou l'on ne retrouve pas le nombre de documents
    (usual_number_of_docs) dans la base, on la recrée.
    Une fois celle-ci créée ou chargé, on la transforme en retriever

    Parameters
    ----------
    usual_number_of_docs : int
        Nombre habituel de documents que l'on doit retrouver dans la base vectorielle

    force_rebuild_vector_store : bool
        Si True, la base vectorielle est recréée même si elle existe déjà.

    path_to_llm : str
        Chemin vers le répertoire de la base vectorielle Chroma et
        les documents à charger.

    Returns
    -------
    retriever : VectorStoreRetriever

    Example
    -------
    retriever = build_retriver_rag()

    """

    if force_rebuild_vector_store:
        delete_vector_store(path_to_db=path_to_llm_dir)

    vector_store = create_vector_store_with_embed(path_to_db=path_to_llm_dir)

    if vector_store._collection.count() != usual_number_of_docs:
        print("La collection de documents est vide ou ne correspond pas au nombre attendu de documents.")
        print("Chargement des documents...")

        # Si il y a déjà des documents dans la base vectorielle, on les supprime
        if vector_store._collection.count() != 0:
            delete_vector_store(path_to_db=path_to_llm_dir)

        # Informations from markdown file
        sections = get_documents_about_project(path_directory=path_to_llm_dir)
        vector_store.add_documents(documents=sections)

        # Informations from web_site
        web_docs = get_documents_from_website()
        vector_store.add_documents(filter_complex_metadata(documents=web_docs))
        time.sleep(0.5)

    print(f"Nombre de documents dans la base vectorielle : {vector_store._collection.count()}")

    retriever = vector_store.as_retriever(
        k=3,
    )

    return retriever

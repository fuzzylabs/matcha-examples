"""ZenML step to build and push custom Docker image for llm model serving."""
import docker
from zenml.logger import get_logger
from zenml.steps import BaseParameters, step
from zenml.client import Client
logger = get_logger(__name__)

registry = Client().active_stack.container_registry


class DockerImageParameters(BaseParameters):
    """Docker image parameters."""

    # Path of the directory containing the files used for building the Docker image.
    docker_directory: str


@step
def build_docker_image(params: DockerImageParameters) -> str:
    """Step to build a custom Docker image.

    Args:
        params: Docker image parameters

    Returns:
        str: resulting Docker image tag
    """
    client = docker.DockerClient()
    tag = f"{registry.config.uri}/llm-summarization"
    logger.info(f"Building docker image {tag} in {params.docker_directory}")

    # To push image to private container registry, we need to tag it with the registry name first
    image, _ = client.images.build(path=params.docker_directory, tag=tag)

    logger.info(f"Pushing docker image {tag} to the registry")
    client.images.push(repository=tag)

    return tag

from pinecone import Pinecone, ServerlessSpec
from llama_index.vector_stores.pinecone import PineconeVectorStore

class VectorStoreFactory:
    @staticmethod
    def create_vector_store(config):
        api_key = config['vector_store']['api_key']
        index_name = config['vector_store']['index_name']
        dimension = config['vector_store']['dimension']
        metric = config['vector_store']['metric']
        cloud = config['vector_store']['cloud']
        region = config['vector_store']['region']

        pc = Pinecone(api_key=api_key)
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
        return PineconeVectorStore(index_name=index_name)

from llama_index.core.node_parser import SimpleNodeParser

class ChunkerFactory:
    @staticmethod
    def create_chunker(config):
        if config['chunking']['strategy'] == 'simple':
            return SimpleNodeParser.from_defaults(
                chunk_size=config['chunking']['chunk_size'],
                chunk_overlap=config['chunking']['chunk_overlap']
            )
        raise ValueError("Unsupported chunking strategy")

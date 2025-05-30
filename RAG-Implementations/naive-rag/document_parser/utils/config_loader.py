import configparser

def load_config(path: str = "document_parser/config/config.ini"):
    config = configparser.ConfigParser()
    config.read(path)
    return config

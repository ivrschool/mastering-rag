import configparser

def load_config(path: str = "config/config.ini"):
    config = configparser.ConfigParser()
    config.read(path)
    return config

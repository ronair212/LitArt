import sys
sys.path.insert(1,'/home/nair.ro/LitArt/falcon')


from lightning.pytorch.loggers import TensorBoardLogger

def get_logger(log_path):
    return TensorBoardLogger(log_path, name="my_model")

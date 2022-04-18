import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
import global_config.config as config
import pandas as pd
from src.data.make_dataset import make_dataset
from src.models.create_model import create_model
from src.models.train_model import train_model
from src.models.predict_model import predict_model
import hydra
import warnings
warnings.filterwarnings("ignore")


def run_model(cfg: DictConfig):
    train_raw_path = config.DATA_PATH+config.RAW_TRAIN_DATA
    test_raw_path = config.DATA_PATH+config.RAW_TEST_DATA
    train_processed_path = config.DATA_PATH + config.PROCESSED_TRAIN_DATA
    test_processed_path = config.DATA_PATH + config.PROCESSED_TEST_DATA

    make_dataset(train_raw_path, train_processed_path)
    make_dataset(test_raw_path, test_processed_path)

    train = pd.read_csv(train_processed_path)
    test = pd.read_csv(test_processed_path)

    model = create_model(cfg)
    model = train_model(train, model, cfg)
    predict_model(test, model)


@hydra.main(config_name='config/experiment_1.yaml')
def run(cfg: DictConfig):
    run_model(cfg.config)


run()

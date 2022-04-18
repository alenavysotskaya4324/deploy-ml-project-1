# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import global_config.config as config
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import src.features.build_features as bf


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def make_dataset(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data = pd.read_csv(input_filepath)

    data = bf.create_features(data)
    data = bf.transform_wind_speed_feature(data)

    if not output_filepath:
        if 'train' in input_filepath:
            output_filepath = config.DATA_PATH + config.PROCESSED_TRAIN_DATA
        else:
            output_filepath = config.DATA_PATH + config.PROCESSED_TEST_DATA
    data.to_csv(output_filepath, index=False)

    logger.info('raw data were processed')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    make_dataset()

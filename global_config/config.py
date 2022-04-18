import os

DATA_PATH = r'%s' % os.path.abspath(os.path.join(
    os.path.dirname("src"), 'data')).replace('\\', '/')

PROCESSED_TRAIN_DATA = "/processed/processed_train.csv"
PROCESSED_TEST_DATA = "/processed/processed_test.csv"

RAW_TRAIN_DATA = "/raw/train.csv"
RAW_TEST_DATA = "/raw/test.csv"

SAMPLE_SUBMISSION = "/submission/sampleSubmission.csv"
SAMPLE_SUBMISSION_WITH_ANSWERS = "/submission/SampleSubmissionWithAnswers.csv"

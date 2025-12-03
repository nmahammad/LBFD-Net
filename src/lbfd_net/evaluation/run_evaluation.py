from lbfd_net.evaluation.evaluation_pipeline import EvaluationPipeline
from lbfd_net.helpers.constants import SEED
from lbfd_net.helpers.set_random_seed import set_random_seed

MODEL_NAME = "alexnet"
TEST_BATCH_SIZE = 16
RUN_INDEX = 1

def main():

    set_random_seed(SEED)
    
    evaluation_pipeline = EvaluationPipeline(
        model_name=MODEL_NAME,
        test_batch_size=TEST_BATCH_SIZE,
        manual_run_index=RUN_INDEX,
        log_to_mlflow=True  
    )
    evaluation_pipeline.evaluate()

if __name__ == "__main__":
    main()

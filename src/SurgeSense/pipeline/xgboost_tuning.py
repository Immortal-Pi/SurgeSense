from SurgeSense.config.configuration import ConfigurationManager
from SurgeSense.components.hyperparameter_xgboost import hyperOptTraining
from SurgeSense import logger 

STAGE_NAME='Model Trainer Stage'


class ModelTrainerTrainingPipelineXGBoost:
    def __init__(self):
        pass

    def main(self):
        config=ConfigurationManager()
        hyperopt_config=config.get_hyperopt_config_XGBoost()
        hyperopt_config_training=hyperOptTraining(config=hyperopt_config)
        best_results,trails=hyperopt_config_training.train()
        hyperopt_config_training.register_best_model(best_results,trails)


if __name__=='__main__':
    try:
        logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
        obj=ModelTrainerTrainingPipelineXGBoost()
        obj.main()
        logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<<\n\nx==========x')
    except Exception as e :
        logger.exception(e)
        raise e 
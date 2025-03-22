import os 
from SurgeSense.config.configuration import ConfigurationManager
from SurgeSense.components.data_transformation import DataTransformation
from SurgeSense import logger 
from pathlib import Path 

STAGE_NAME='Data transformation stage'

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path('artifacts/data_validation/status.txt'),'r') as f:
                status=f.read().split(' ')[-1]
            if status=='True':
                config=ConfigurationManager()
                data_transformation_config=config.get_data_transformation_config()
                data_transform=DataTransformation(config=data_transformation_config)
                pipeline=data_transform.transform_data_pipeline()
                data_transform.train_test_spliting(pipeline)
            else:
                raise Exception('your schema is not valid')
        except Exception as e:
            print(e)


if __name__=='__main__':
    try:
        logger.info(f'>>>>> stage {STAGE_NAME} started <<<<<')
        data_transformation=DataTransformationTrainingPipeline()
        data_transformation.main()
        logger.info(f'>>>>> stage {STAGE_NAME} completed <<<<<\n\n x==========x')
    except Exception as e:
        logger.exception(e)
        raise e 



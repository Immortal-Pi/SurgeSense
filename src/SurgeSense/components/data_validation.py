from SurgeSense.config.configuration import DataValidationConfig
from SurgeSense import logger
import pandas as pd 

class DataValidation:
    def __init__(self,config:DataValidationConfig):
        self.config=config

    def validate_all_columns(self)->bool:
        try:
            validate_status=None 
            data=pd.read_csv(self.config.unzip_data_dir)
            all_cols=data.dtypes.to_dict()
            all_schema=self.config.all_schema
            
            if list(all_cols.keys())!=list(all_schema.keys()):
                validate_status=False

            else:
                # for col in all_schema:
                #     print(all_schema[col])
                #     print(all_cols[col])
                dtype_match=all(all_cols[col] == all_schema[col] for col in all_schema)
                expected_columns=data.shape[1]
                column_count_match=(len(all_schema))
            #    print(dtype_match, expected_columns, column_count_match)
                validate_status=expected_columns==column_count_match and dtype_match
            with open(self.config.STATUS_FILE,'w') as f:
                f.write(f'Validation status: {validate_status}')
            logger.info(f'Validation status: {validate_status}')
            return validate_status
        except Exception as e:
            raise e
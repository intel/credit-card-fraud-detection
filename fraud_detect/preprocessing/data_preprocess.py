import pandas as pd
import yaml
from category_encoders import TargetEncoder
from sklearn import preprocessing
import pickle

class XGBoostPreprocessor:
    def __init__(self, preprocessing_config, model_feature_names=[], dp_engine='pandas'):
        """
        :param preprocessing_config: configuration
        """
        if isinstance(preprocessing_config, dict):
            self.config = preprocessing_config
        elif isinstance(preprocessing_config, str):
            self.config = self._load_configuration(preprocessing_config)
        else: 
            raise ValueError("Unsupported data preprocessing configuration, it should be either a dict or a file path string.")
        
        self.dp_engine = dp_engine
        if not self._support_dp_engine_check():
            raise ValueError("Unsupported data processing engine")
        self.tmp = {}
        
        self.loaded_dict = {}
        for step in self.config['post_splitting_transformation']:
            if 'target_encoding' in step and step['target_encoding']['encoders_path'] is not None:
                target_encoder_path = step['target_encoding']['encoders_path']
                with open(target_encoder_path, 'rb') as file:
                    self.loaded_dict['loaded_encoders']  = pickle.load(file)
                    
        self.model_feature_names = model_feature_names

    def _support_dp_engine_check(self):
        if self.dp_engine.lower() not in ['pandas']:
            return False
        return True
    def _load_configuration(self, config_path: str):
        if config_path.endswith('yaml') or config_path.endswith('yml'):
            with open(config_path,'r') as file:
                return yaml.safe_load(file)
            
    def load_data(self, data):
        """
        :param data: data source, could be file path, file object or pandas DataFrame
        """
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise ValueError("Unsupported data format")
        self.data = df
            
    def preprocess_data(self, input) -> pd.DataFrame:
        """
        Execute the preprocessing

        :param df: DataFrame to be processed.
        :return: Processed DataFrame
        """
        
        self.load_data(input)
        
        # data transform
        if 'pre_splitting_transformation' in self.config:
            transform_steps = self.config['pre_splitting_transformation']
            for step in transform_steps:
                first_key = list(step.keys())[0]
                if first_key == 'normalize_feature_names': 
                    self.normalize_feature_names(list(step.values())[0])
                elif first_key == 'rename_feature_names':
                    raise NotImplementedError 
                elif first_key == 'drop_features':
                    raise NotImplementedError
                elif first_key == 'outlier_treatment':
                    raise NotImplementedError
                elif first_key == 'categorify': 
                    self.categorify(list(step.values())[0])
                elif first_key == 'strip_chars':
                    self.strip_chars(list(step.values())[0])
                elif first_key == 'combine_cols':
                    self.combine_cols(list(step.values())[0])
                elif first_key == 'change_datatype':
                    self.change_datatype(list(step.values())[0])
                elif first_key == 'time_to_seconds':
                    self.time_to_seconds(list(step.values())[0])
                elif first_key == 'min_max_normalization':
                    self.min_max_normalization(list(step.values())[0])
                elif first_key == 'one_hot_encoding':
                    self.one_hot_encoding(list(step.values())[0])
                elif first_key == 'string_to_list':
                    self.string_to_list(list(step.values())[0])
                elif first_key == 'multi_hot_encoding':
                    self.multi_hot_encoding(list(step.values())[0])
                elif first_key == 'add_constant_feature':
                    self.add_constant_feature(list(step.values())[0])
                elif first_key == 'define_variable':
                    self.define_variable(list(step.values())[0])
                elif first_key == 'modify_on_conditions':
                    self.modify_on_conditions(list(step.values())[0])
           
        # expand features.
        current_feature_names = self.data.columns.tolist()
        missing_features = set(self.model_feature_names) - set(current_feature_names)
        
        for feature in missing_features:
            self.data[feature] = 0.0
        
        if 'post_splitting_transformation' in self.config:
            encoding_steps = self.config['post_splitting_transformation']
            for step in encoding_steps:
                first_key = list(step.keys())[0]
                if first_key == 'target_encoding': 
                    params = list(step.values())[0]
                    self.target_encoding(params)
                elif first_key == 'label_encoding':
                    params = list(step.values())[0]
                    self.label_encoding(params)
            
        
        if 'data_spec' in self.config:
            data_spec = self.config['data_spec']
            if 'ignore_cols' in data_spec:
                self.ignore_cols = data_spec['ignore_cols']
                self.data = self.data.drop(columns=self.ignore_cols)                
        
        # Only remain the features that the model needs, and reorder the features.
        self.data = self.data[self.model_feature_names]
        self.data = self.data.astype('float32')
        
        return self.data

    def target_encoding(self, params):
        feature_cols = params['feature_cols']
        
        for col in feature_cols:
            if col in self.loaded_dict['loaded_encoders']:
                tgt_encoder = self.loaded_dict['loaded_encoders'][col]
                self.data[col] = tgt_encoder.transform(self.data[col]).astype('float32')
            
    def label_encoding(self, params):
        feature_cols = params['feature_cols']
        for col in feature_cols: 
            label_encoder = preprocessing.LabelEncoder()
            self.data[col] = label_encoder.transform(self.data[col]).astype('int64')
            
    def normalize_feature_names(self, steps):
        for step in steps:
            first_key = list(step.keys())[0]
            if first_key == 'replace_chars':
                self.replace_chars(list(step.values())[0])
            elif first_key == 'lowercase':
                if list(step.values())[0]:
                    self.to_lowercase()

    def replace_chars(self, replacements):
        for key, value in replacements.items():
            self.data.columns = self.data.columns.str.replace(key, value)

    def to_lowercase(self):
        self.data.columns = self.data.columns.str.lower()

    def categorify(self, features):
        for target_feature, new_feature in features.items():
            self.data[new_feature] = self.data[target_feature].astype('category').cat.codes

    def strip_chars(self, features):
        for old_feature, mapping in features.items():
            for new_feature, char in mapping.items():
                self.data[new_feature] = self.data[old_feature].str.strip(char)
    
    def combine_cols(self, features):
        for new_feature, content in features.items():
            for operation, target_feature_list in content.items(): 
                if len(target_feature_list) < 2:
                    raise ValueError('there is less than 2 items in the list, cannot concatenate')
                else:
                    if operation == 'concatenate_strings':
                        tmp_feature = self.data[target_feature_list[0]].astype('str')
                        for feature in target_feature_list[1:]:
                            tmp_feature = tmp_feature + self.data[feature].astype('str')
                        self.data[new_feature] = tmp_feature 
    
    def change_datatype(self, col_dtypes):
        for col, dtype in col_dtypes.items():
            if isinstance(dtype, list):
                for type in dtype:
                    self.data[col] = self.data[col].astype(type)
            else:
                self.data[col] = self.data[col].astype(dtype)

    def time_to_seconds(self, features):
        for old_feature, new_feature in features.items():
            self.data[old_feature] = self.data[old_feature].astype('datetime64[s]')
            self.data[new_feature] = self.data[old_feature].dt.hour*60 + self.data[old_feature].dt.minute 
    
    def min_max_normalization(self, features):
        for old_feature, new_feature in features.items():
            self.data[new_feature] = (self.data[old_feature] - self.data[old_feature].min())/(self.data[old_feature].max() -  self.data[old_feature].min())
         
    def one_hot_encoding(self, features):
        if self.dp_engine == 'modin':
            import modin.pandas as pd
        elif self.dp_engine == 'pandas':
            import pandas as pd            
        else:
            raise ValueError("only modin or pandas is accepted as dp_engine")
        
        for feature, is_drop in features.items():
            self.data = pd.get_dummies(self.data, columns=[feature], drop_first=True)
            if is_drop and feature in self.data:
                self.data.drop(columns=[feature], axis=1, inplace=True)

    def string_to_list(self, features):
        for old_feature, mapping in features.items():
            for new_feature, sep in mapping.items():
                self.data[new_feature] = self.data[old_feature].map(lambda x: str(x).split(sep))

    def multi_hot_encoding(self, features):
        if self.dp_engine == 'modin':
            import modin.pandas as pd
        elif self.dp_engine == 'pandas':
            import pandas as pd            
        else:
            raise ValueError("only modin or pandas is accepted as dp_engine")

        if self.dp_engine == 'modin':
            for feature, is_drop in features.items():
                exploded = self.data[feature].explode().to_frame()
                raw_one_hot = pd.get_dummies(exploded, columns=[feature])
                tmp_df = raw_one_hot.groupby(level=0).sum()
                self.data = pd.concat([self.data, tmp_df], axis=1) 
                to_be_replaced = feature.split('?')[0]
                self.data.columns = self.data.columns.str.replace(to_be_replaced+'\?'+'_', '')
                col_names = self.data.columns 
                if '' in col_names or 'nan' in col_names: 
                    self.data.drop('', axis=1, inplace=True, errors='ignore')
                    self.data.drop('nan', axis=1, inplace=True, errors='ignore')
                if is_drop: 
                    self.data.drop(columns=[feature], axis=1, inplace=True)
        else: 
            for feature, is_drop in features.items():
                exploded = self.data[feature].explode()
                raw_one_hot = pd.get_dummies(exploded, columns=[feature])
                tmp_df = raw_one_hot.groupby(raw_one_hot.index).sum()
                self.data = pd.concat([self.data, tmp_df], axis=1) 
                col_names = self.data.columns 
                if '' in col_names or 'nan' in col_names: 
                    self.data.drop('', axis=1, inplace=True, errors='ignore')
                    self.data.drop('nan', axis=1, inplace=True, errors='ignore')
                if is_drop: 
                    self.data.drop(columns=[feature], axis=1, inplace=True)
    
    def add_constant_feature(self, features):
        
        if self.dp_engine == 'modin':
            import modin.pandas as pd
        elif self.dp_engine == 'pandas':
            import pandas as pd            
        else:
            raise ValueError("only modin or pandas is accepted as dp_engine")
        
        if self.dp_engine == 'modin':
            for target_feature, const_value in features.items():
                self.data[target_feature] = pd.Series(np.zeros(self.data.shape[0]), dtype=np.int8)
        else:
            for target_feature, const_value in features.items():
                self.data[target_feature] = const_value

    def define_variable(self, definitions):
        df = self.data
        for var_name, expression in definitions.items():
            self.tmp[var_name] = eval(expression)

    def modify_on_conditions(self, map):
        tmp = self.tmp 
        for col, conditions in map.items():
    
            for condition, value in conditions.items():
                df = self.data
                df.loc[eval(condition), col] = value 
                self.data = df
     
    def get_data(self):
        return self.data
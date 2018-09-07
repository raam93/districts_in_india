import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

class data_interface:
    
    def __init__(self, data_name="LSAT"):
        
        self.data_name = data_name
        if self.data_name == "LSAT":
            self.train_df = pd.read_csv("data\LSAT_train.csv")
            self.test_df = pd.read_csv("data\LSAT_test.csv")
            
            self.column_names = self.train_df.columns.tolist()
            self.feature_names = ['gpa', 'lsat', 'isblack']
            self.target_name = ['fya']
            self.unused_features = []
            
            self.feature_description = {'gpa':'High School GPA',
                                        'lsat': 'LSAT score',
                                        'isblack': 'Race = Black',
                                        'fya': 'First Year Average in college'}
            
            # setting the data type of columns
            self.train_df[['gpa', 'lsat', 'fya']] = self.train_df[['gpa', 'lsat', 'fya']].astype(float)
            self.train_df[['isblack']] = self.train_df[['isblack']].astype('category')
            self.test_df[['gpa', 'lsat', 'fya']] = self.test_df[['gpa', 'lsat', 'fya']].astype(float)
            self.test_df[['isblack']] = self.test_df[['isblack']].astype('category')

            self.categorical_feature_indexes = [2]
            self.continuous_feature_indexes = [0,1]
            self.continuous_feature_names = [self.column_names[i] for i in self.continuous_feature_indexes]
            self.categorical_feature_names = [self.column_names[i] for i in self.categorical_feature_indexes]
            
            # MAD for LSAT dataset
            self.mads = self.get_mads_from_training_data()

            self.encoded_feature_names = None  

            # TODO: range of an (encoded) instance - harcoded for now - need to change this in data interface
            self.minx = np.array([[0.0, 11.0, 0.0]])
            self.maxx = np.array([[4.2, 48.0, 1.0]])

            
        if self.data_name == "COMPAS":
            data_df = pd.read_csv("data\compas-scores-two-years.csv")
            self.data_df = data_df[['sex', 'race', 'age_cat', 'c_charge_degree', 'priors_count', 'two_year_recid', 'decile_score']]
            
            self.column_names = self.data_df.columns.tolist()
            self.feature_names = ['sex', 'race', 'age_cat', 'c_charge_degree', 'priors_count']
            self.target_name = ['two_year_recid']
            self.unused_features = ['decile_score'] # used during building the model for comparision # TODO: add more info
            
            self.feature_description = {'sex':'Male/Female',
                                        'race': 'African-American/Caucasian/Hispanic/Asian/Native American/Other)',
                                        'age_cat': '<25/25-45/>45',
                                        'c_charge_degree': 'Felony(F)/Misdemeanor(M)',
                                        'priors_count': 'Number of prior criminal records',
                                        'two_year_recid': 'Will commit another crime within 2 years?'}
        
            self.data_df[['sex', 'race', 'age_cat', 'c_charge_degree']] = self.data_df[['sex', 'race', 'age_cat', 'c_charge_degree']].astype('category')
            self.data_df[['priors_count']] = self.data_df[['priors_count']].astype(int)
            
            self.categorical_feature_indexes = [0,1,2,3]
            self.continuous_feature_indexes = [4]
            self.continuous_feature_names = [self.column_names[i] for i in self.continuous_feature_indexes]
            self.categorical_feature_names = [self.column_names[i] for i in self.categorical_feature_indexes]
            
            self.train_df, self.test_df = self.split_data(self.data_df)

            # TODO: range of an (encoded) instance - harcoded for now - need to change this in data interface
            self.minx = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            self.maxx = np.array([[38.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

            #self.mads = self.get_mads_from_training_data() #TODO:
            self.mads = np.array([[2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

            self.data_df_encoded =  self.encode_data(self.data_df.drop(self.unused_features, axis=1)) # becoz there could be unused features
            self.encoded_feature_names = [x for x in self.data_df_encoded.columns.tolist() if x not in self.target_name]            

            self.encoded_categorical_feature_indexes = self.get_encoded_categorical_feature_indexes() # TODO: description
            
    
    def get_mads_from_training_data(self): # works only for LSAT type data. TODO:
        mads = []
        for k in range(0, len(self.feature_names)):
            if k in self.categorical_feature_indexes:
                mads.append(1.0)
            else:
                mads.append(np.median(abs(self.train_df.values[:,k] - np.median(self.train_df.values[:,k]))))
        return mads
    
    
    def get_meta_data(self):
        print("\nFeature description:\n\n{" + "\n".join("{}:   {}".format(k, v) for k, v in self.feature_description.items()) + "}")
        print("\n#############################################")
        
        print("\n\nContinuous feature names:  \n", self.continuous_feature_names)
        print("\n\nTotal number of Continuous features:  ", len(self.continuous_feature_names))
        print("\n\nContinuous features summary:  \n\n", self.train_df.describe().transpose())

        print("\n#############################################")
        print("\n\nCategorical feature names:  \n", self.categorical_feature_names)
        print("\n\nTotal number of Categorical features:  ", len(self.categorical_feature_names))
        print("\n\nCategorical features summary:  \n\n")
        for col in self.categorical_feature_names:
            print(self.train_df[col].value_counts(),"\n")

        print("\n**Note: All the summaries are based on train data only")
        
    
    def split_data(self, data, test_size=0.20):
        train_df, test_df = train_test_split(data, test_size=test_size, random_state=17)
        return train_df, test_df
    

    def encode_data(self, data):
        return pd.get_dummies(data, drop_first=True, columns = self.categorical_feature_names)
    
    def get_encoded_data(self, return_format): # TODO: cnofusion in return_format
        X_train = self.train_df_encoded[self.encoded_feature_names]
        X_test = self.test_df_encoded[self.encoded_feature_names]
        y_train = self.train_df_encoded[self.target_name]
        y_test = self.test_df_encoded[self.target_name]
        if return_format == "array": 
            return X_train.values, X_test.values, y_train.values, y_test.values
        else: 
            return X_train, X_test, y_train, y_test

    
    # TODO: Add description for the dataframe 
    def prepare_df_for_encoding(self, test):
        levels = []
        colnames = self.categorical_feature_names
        for cat_feature in colnames:
            levels.append(self.data_df[cat_feature].cat.categories.tolist())

        df = pd.DataFrame({colnames[0]:levels[0]})
        for col in range(1,len(colnames)):
            temp_df = pd.DataFrame({colnames[col]:levels[col]})
            df = pd.concat([df, temp_df], axis=1)

        colnames = self.continuous_feature_names
        for col in range(0,len(colnames)):
            temp_df = pd.DataFrame({colnames[col]:[]})
            df = pd.concat([df, temp_df], axis=1)

        return df

    
    def get_test_inputs(self, encode, *args, **kwargs):
        params = kwargs.get('params', None)
        no_of_random_test_instances = kwargs.get('no_of_random_test_instances', None)
        
        test = pd.DataFrame.from_dict(params, orient='index', columns=self.feature_names) if params is not None else self.test_df[self.feature_names].sample(n=no_of_random_test_instances)   

        if encode is False:
            return test
        else:
            temp = self.prepare_df_for_encoding(test)
            temp = temp.append(test, ignore_index=True)
            temp = self.encode_data(temp)
            return temp.tail(test.shape[0])
        
    # categorical_cols_first is a dictionary of first levels (of each categorical variables) that will be dropped by pd.get_dummies()
    def get_cat_cols_first(self):
        categorical_cols_first = []
        for col in self.categorical_feature_names:
            categorical_cols_first.append(self.data_df[col].value_counts().sort_index().keys()[0]) # TODO: see if data_df will cause any problem here - should I include data_df
        categorical_cols_first = dict(zip(self.categorical_feature_names, categorical_cols_first))
        return categorical_cols_first

    # function to get the original data from dummy encoded data
    ## based on my comment: https://github.com/pandas-dev/pandas/issues/8745
    def from_dummies(self, data, prefix_sep='_'):
        out = data.copy()

        categorical_cols_first = self.get_cat_cols_first()
        
        for col_parent in self.categorical_feature_names:

            filter_col = [col for col in data if col.startswith(col_parent)]

            cols_with_ones = np.argmax(data[filter_col].values, axis=1)
            
            org_col_values = []
            
            for row, col in enumerate(cols_with_ones):
                if((col==0) & (data[filter_col].iloc[row][col] < 1)):
                    org_col_values.append(categorical_cols_first.get(col_parent))
                else:
                    org_col_values.append(data[filter_col].columns[col].split(col_parent+prefix_sep,1)[1])

            out[col_parent] = pd.Series(org_col_values).values
            out.drop(filter_col, axis=1, inplace=True)    

        return out


    def get_decoded_data(self, data):
        if isinstance(data, np.ndarray):
            index = [i for i in range(0, len(data))]
            data = pd.DataFrame(data = data, index = index, columns = self.encoded_feature_names)
        return self.from_dummies(data)


    # TODO: add description
    def get_encoded_categorical_feature_indexes(self):
        cols = []
        for col_parent in self.categorical_feature_names:
            temp = [self.encoded_feature_names.index(col) for col in self.encoded_feature_names if col.startswith(col_parent)]
            cols.append(temp)
        return cols
        






RANDOM_SEED = 42
DATA_PATH = 'D:\proj\credit default\DATA1030_MIDTERM_PROJECT\data\data_imputed.csv'
TEST_SIZE = 0.2
ORDINAL_FEATURES = ['REGION_RATING_CLIENT',
                    'REGION_RATING_CLIENT_W_CITY',
                    'HOUR_APPR_PROCESS_START',
                    'WEEKDAY_APPR_PROCESS_START']
CONTINUOUS_FEATURES = ['FLOORSMAX_MEDI',
                       'TOTALAREA_MODE',
                       'EXT_SOURCE_2',
                       'EXT_SOURCE_3',
                       'AMT_INCOME_TOTAL',
                       'AMT_CREDIT',
                       'AMT_ANNUITY',
                       'AMT_GOODS_PRICE',
                       'AMT_REQ_CREDIT_BUREAU_YEAR',
                       'CNT_CHILDREN',
                       'CNT_FAM_MEMBERS',
                       'DAYS_BIRTH',
                       'DAYS_EMPLOYED',
                       'DAYS_REGISTRATION',
                       'DAYS_ID_PUBLISH',
                       'DAYS_LAST_PHONE_CHANGE',
                       'REGION_POPULATION_RELATIVE']
CATEGORICAL_FEATURES = ['EMERGENCYSTATE_MODE',
                        'OCCUPATION_TYPE',
                        'NAME_CONTRACT_TYPE',
                        'NAME_TYPE_SUITE',
                        'NAME_INCOME_TYPE',
                        'NAME_EDUCATION_TYPE',
                        'NAME_FAMILY_STATUS',
                        'NAME_HOUSING_TYPE',
                        'FLAG_OWN_CAR',
                        'FLAG_OWN_REALTY',
                        'FLAG_EMP_PHONE',
                        'FLAG_WORK_PHONE',
                        'FLAG_PHONE',
                        'FLAG_EMAIL',
                        'FLAG_DOCUMENT_2',
                        'FLAG_DOCUMENT_3',
                        'FLAG_DOCUMENT_4',
                        'FLAG_DOCUMENT_5',
                        'FLAG_DOCUMENT_6',
                        'FLAG_DOCUMENT_7',
                        'FLAG_DOCUMENT_8',
                        'FLAG_DOCUMENT_9',
                        'FLAG_DOCUMENT_10',
                        'FLAG_DOCUMENT_11',
                        'FLAG_DOCUMENT_12',
                        'FLAG_DOCUMENT_13',
                        'FLAG_DOCUMENT_14',
                        'FLAG_DOCUMENT_15',
                        'FLAG_DOCUMENT_16',
                        'FLAG_DOCUMENT_17',
                        'FLAG_DOCUMENT_18',
                        'FLAG_DOCUMENT_19',
                        'FLAG_DOCUMENT_20',
                        'FLAG_DOCUMENT_21',
                        'CODE_GENDER',
                        'ORGANIZATION_TYPE',
                        'LIVE_REGION_NOT_WORK_REGION',
                        'LIVE_CITY_NOT_WORK_CITY',
                        'REG_REGION_NOT_LIVE_REGION',
                        'REG_REGION_NOT_WORK_REGION',
                        'REG_CITY_NOT_LIVE_CITY',
                        'REG_CITY_NOT_WORK_CITY']

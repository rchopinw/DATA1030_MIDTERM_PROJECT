RANDOM_SEED = 42
DATA_PATH = '/Users/bangxixiao/Desktop/python_projects/DATA1030_MIDTERM_PROJECT/data/data_imputed.csv'
TEST_SIZE = 0.2

DI2_FEATURES = dict(
    ORDINAL_FEATURES=['REGION_RATING_CLIENT',
                      'REGION_RATING_CLIENT_W_CITY'],
    CONTINUOUS_FEATURES=['EXT_SOURCE_2',
                         'EXT_SOURCE_3',
                         'AMT_INCOME_TOTAL',
                         'AMT_CREDIT',
                         'AMT_ANNUITY',
                         'AMT_GOODS_PRICE',
                         'AMT_REQ_CREDIT_BUREAU_MON',
                         'AMT_REQ_CREDIT_BUREAU_DAY',
                         'AMT_REQ_CREDIT_BUREAU_WEEK',
                         'AMT_REQ_CREDIT_BUREAU_QRT',
                         'AMT_REQ_CREDIT_BUREAU_HOUR',
                         'OBS_30_CNT_SOCIAL_CIRCLE',
                         'DEF_30_CNT_SOCIAL_CIRCLE',
                         'OBS_60_CNT_SOCIAL_CIRCLE',
                         'DEF_60_CNT_SOCIAL_CIRCLE',
                         'CNT_CHILDREN',
                         'CNT_FAM_MEMBERS',
                         'CNT_DOCUMENT',
                         'DAYS_BIRTH',
                         'DAYS_EMPLOYED',
                         'DAYS_REGISTRATION',
                         'DAYS_ID_PUBLISH',
                         'REGION_POPULATION_RELATIVE'],
    CATEGORICAL_FEATURES=['NAME_CONTRACT_TYPE',
                          'NAME_INCOME_TYPE',
                          'NAME_EDUCATION_TYPE',
                          'NAME_FAMILY_STATUS',
                          'NAME_HOUSING_TYPE',
                          'FLAG_OWN_CAR',
                          'FLAG_OWN_REALTY',
                          'CODE_GENDER',
                          'ORGANIZATION_TYPE',
                          'LIVE_REGION_NOT_WORK_REGION',
                          'LIVE_CITY_NOT_WORK_CITY',
                          'REG_REGION_NOT_LIVE_REGION',
                          'REG_REGION_NOT_WORK_REGION',
                          'REG_CITY_NOT_LIVE_CITY',
                          'REG_CITY_NOT_WORK_CITY'],
    DATA_PATH='data/data_imputed_2.csv'
)
DI1_FEATURES = dict(
    ORDINAL_FEATURES=['REGION_RATING_CLIENT',
                      'REGION_RATING_CLIENT_W_CITY',
                      'HOUR_APPR_PROCESS_START',
                      'WEEKDAY_APPR_PROCESS_START'],
    CONTINUOUS_FEATURES=['FLOORSMAX_MEDI',
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
                         'REGION_POPULATION_RELATIVE'],
    CATEGORICAL_FEATURES=['EMERGENCYSTATE_MODE',
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
                          'FLAG_DOCUMENT_11',
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
                          'REG_CITY_NOT_WORK_CITY'],
    DATA_PATH='data/data_imputed.csv'
)
DU2_FEATURES = dict(
    ORDINAL_FEATURES=['REGION_RATING_CLIENT',
                      'REGION_RATING_CLIENT_W_CITY',
                      'HOUR_APPR_PROCESS_START',
                      'WEEKDAY_APPR_PROCESS_START'],
    CONTINUOUS_FEATURES=['FLOORSMAX_MEDI',
                         'TOTALAREA_MODE',
                         'OWN_CAR_AGE',
                         'APARTMENTS_MEDI',
                         'BASEMENTAREA_MEDI',
                         'YEARS_BEGINEXPLUATATION_MEDI',
                         'YEARS_BUILD_MEDI',
                         'COMMONAREA_MEDI',
                         'ELEVATORS_MEDI',
                         'ENTRANCES_MEDI',
                         'FLOORSMIN_MEDI',
                         'LANDAREA_MEDI',
                         'LIVINGAPARTMENTS_MEDI',
                         'LIVINGAREA_MEDI',
                         'NONLIVINGAPARTMENTS_MEDI',
                         'NONLIVINGAREA_MEDI',
                         'EXT_SOURCE_1',
                         'EXT_SOURCE_2',
                         'EXT_SOURCE_3',
                         'AMT_INCOME_TOTAL',
                         'AMT_CREDIT',
                         'AMT_ANNUITY',
                         'CNT_DOCUMENT',
                         'AMT_GOODS_PRICE',
                         'AMT_REQ_CREDIT_BUREAU_MON',
                         'AMT_REQ_CREDIT_BUREAU_DAY',
                         'AMT_REQ_CREDIT_BUREAU_WEEK',
                         'AMT_REQ_CREDIT_BUREAU_QRT',
                         'AMT_REQ_CREDIT_BUREAU_HOUR',
                         'OBS_30_CNT_SOCIAL_CIRCLE',
                         'DEF_30_CNT_SOCIAL_CIRCLE',
                         'OBS_60_CNT_SOCIAL_CIRCLE',
                         'DEF_60_CNT_SOCIAL_CIRCLE',
                         'CNT_CHILDREN',
                         'CNT_FAM_MEMBERS',
                         'DAYS_BIRTH',
                         'DAYS_EMPLOYED',
                         'DAYS_REGISTRATION',
                         'DAYS_ID_PUBLISH',
                         'DAYS_LAST_PHONE_CHANGE',
                         'REGION_POPULATION_RELATIVE'],
    CATEGORICAL_FEATURES=['EMERGENCYSTATE_MODE',
                          'NAME_CONTRACT_TYPE',
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
                          'CODE_GENDER',
                          'ORGANIZATION_TYPE',
                          'LIVE_REGION_NOT_WORK_REGION',
                          'LIVE_CITY_NOT_WORK_CITY',
                          'REG_REGION_NOT_LIVE_REGION',
                          'REG_REGION_NOT_WORK_REGION',
                          'REG_CITY_NOT_LIVE_CITY',
                          'REG_CITY_NOT_WORK_CITY'],
    DATA_PATH='data/data_unimputed_2.csv'
)
DU1_FEATURES = dict(
    ORDINAL_FEATURES=['REGION_RATING_CLIENT',
                      'REGION_RATING_CLIENT_W_CITY',
                      'HOUR_APPR_PROCESS_START',
                      'WEEKDAY_APPR_PROCESS_START'],
    CONTINUOUS_FEATURES=['FLOORSMAX_MEDI',
                         'TOTALAREA_MODE',
                         'OWN_CAR_AGE',
                         'APARTMENTS_MEDI',
                         'BASEMENTAREA_MEDI',
                         'YEARS_BEGINEXPLUATATION_MEDI',
                         'YEARS_BUILD_MEDI',
                         'COMMONAREA_MEDI',
                         'ELEVATORS_MEDI',
                         'ENTRANCES_MEDI',
                         'FLOORSMAX_MEDI',
                         'FLOORSMIN_MEDI',
                         'LANDAREA_MEDI',
                         'LIVINGAPARTMENTS_MEDI',
                         'LIVINGAREA_MEDI',
                         'NONLIVINGAPARTMENTS_MEDI',
                         'NONLIVINGAREA_MEDI',
                         'EXT_SOURCE_1',
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
                         'REGION_POPULATION_RELATIVE'],
    CATEGORICAL_FEATURES=['EMERGENCYSTATE_MODE',
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
                          'FLAG_DOCUMENT_11',
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
                          'REG_CITY_NOT_WORK_CITY'],
    DATA_PATH='data/data_unimputed.csv'
)

DATA_FILES = {
    'data_imputed_1': DI1_FEATURES,
    'data_imputed_2': DI2_FEATURES,
    'data_unimputed_1': DU1_FEATURES,
    'data_unimputed_2': DU2_FEATURES
}
import os

############ For Windows ##############

DATA_DIR_Win = {
	'CMUMOSI': '/YOUR/PATH/TO/SDR-GNN/dataset/CMUMOSI',
	'CMUMOSEI': '/YOUR/PATH/TO/SDR-GNN/dataset/CMUMOSEI',
	'IEMOCAPFour': '/YOUR/PATH/TO/SDR-GNN/dataset/IEMOCAPFour',
	'IEMOCAPSix': '/YOUR/PATH/TO/SDR-GNN/dataset/IEMOCAP',

}

PATH_TO_FEATURES_Win = {
	'CMUMOSI': os.path.join(DATA_DIR_Win['CMUMOSI'], 'features'),
    'CMUMOSEI': os.path.join(DATA_DIR_Win['CMUMOSEI'], 'features'),
	'IEMOCAPFour': os.path.join(DATA_DIR_Win['IEMOCAPFour'], 'features'),
	'IEMOCAPSix': os.path.join(DATA_DIR_Win['IEMOCAPSix'], 'features'),
}

PATH_TO_LABEL_Win = {
	'CMUMOSI': os.path.join(DATA_DIR_Win['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
	'CMUMOSEI': os.path.join(DATA_DIR_Win['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
	'IEMOCAPSix': os.path.join(DATA_DIR_Win['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),
	'IEMOCAPFour': os.path.join(DATA_DIR_Win['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),
}

PATH_TO_SAVE_ROOT = '/YOUR/PATH/TO/SDR-GNN/result/'

############ For LINUX ##############

DATA_DIR_LINUX = {
	'CMUMOSI': '',
	'CMUMOSEI': '',
	'IEMOCAPSix': '',
	'IEMOCAPFour': '',
}


PATH_TO_FEATURES_LINUX = {
	'CMUMOSI': os.path.join(DATA_DIR_LINUX['CMUMOSI'], 'features'),
	'CMUMOSEI': os.path.join(DATA_DIR_LINUX['CMUMOSEI'], 'features'),
	'IEMOCAPSix': os.path.join(DATA_DIR_LINUX['IEMOCAPSix'], 'features'),
	'IEMOCAPFour': os.path.join(DATA_DIR_LINUX['IEMOCAPFour'], 'features'),
}

PATH_TO_LABEL_LUNUX = {
	'CMUMOSI': os.path.join(DATA_DIR_LINUX['CMUMOSI'], 'CMUMOSI_features_raw_2way.pkl'),
	'CMUMOSEI': os.path.join(DATA_DIR_LINUX['CMUMOSEI'], 'CMUMOSEI_features_raw_2way.pkl'),
	'IEMOCAPSix': os.path.join(DATA_DIR_LINUX['IEMOCAPSix'], 'IEMOCAP_features_raw_6way.pkl'),
	'IEMOCAPFour': os.path.join(DATA_DIR_LINUX['IEMOCAPFour'], 'IEMOCAP_features_raw_4way.pkl'),
}


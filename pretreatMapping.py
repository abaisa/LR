from numpy import nan as NA

def base_mapping(data):
    if type(data) == None:
        return NA
    if type(data) == str:
        new_data = data.strip()
        if not new_data:
            return NA
        if new_data[-1] == '.':
            new_data = new_data[:-1]
        if new_data == '?':
            return NA
        return new_data
    return data

def mapping_2(data):
    if data == 'Private':
        return 1
    if data == 'Self-emp-not-inc':
        return 2
    if data == 'Self-emp-inc':
        return 3
    if data == 'Federal-gov':
        return 4
    if data == 'Local-gov':
        return 5
    if data == 'State-gov':
        return 6
    else:
        return 0
'''
mapping_2 = {
    'Private': 1,
    'Self-emp-not-inc': 2,
    'Self-emp-inc': 3,
    'Federal-gov': 4,
    'Local-gov': 5,
    'State-gov': 6
}
'''
mapping_4 = {
    'Bachelors': 16,
    'Some-college': 15,
    '11th': 14,
    'HS-grad': 13,
    'Prof-school': 12,
    'Assoc-acdm': 11,
    'Assoc-voc': 10,
    '9th': 9,
    '7th-8th': 8,
    '12th': 7,
    'Masters': 6,
    '1st-4th': 5,
    '10th': 4,
    'Doctorate': 3,
    '5th-6th': 2,
    'Preschool': 1
}

mapping_6 = {
    'Married-civ-spouse': 7,
    'Divorced': 6,
    'Never-married': 5,
    'Separated': 4,
    'Widowed': 3,
    'Married-spouse-absent': 2,
    'Married-AF-spouse': 1
}

mapping_7 = {
    'Tech-support': 1,
    'Craft-repair': 2,
    'Other-service': 3,
    'Sales': 4,
    'Exec-managerial': 5,
    'Prof-specialty': 6,
    'Handlers-cleaners': 7,
    'Machine-op-inspct': 8,
    'Adm-clerical': 8,
    'Farming-fishing': 10,
    'Transport-moving': 11,
    'Priv-house-serv': 12,
    'Protective-serv': 13,
    'Armed-Forces': 14
}

mapping_8 = {
    'Wife': 4,
    'Own-child': 3,
    'Husband': 6,
    'Not-in-family': 5,
    'Other-relative': 2,
    'Unmarried': 1
}

mapping_9 = {
    'White': 6,
    'Asian-Pac-Islander': 5,
    'Amer-Indian-Eskimo': 2,
    'Other': 2,
    'Black': 3
}

mapping_10 = {
    'Female': 1,
    'Male': 2
}

def mapping_14(str):
    if str == 'United-States' or 'England' or 'Canada' or 'Germany' or 'Japan' or 'France' or 'Hong':
        return 1
    else:
        return 0

mapping_15 = {
    '<=50K': 0,
    '>50K': 1
}

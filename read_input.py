
def load_data(train_directory):
    train_data = []
    with open(train_directory,'r',encoding='utf-8') as file:
        for line in file:
            line = line.rstrip()
            train_data.append(line.split(' '))
    return train_data

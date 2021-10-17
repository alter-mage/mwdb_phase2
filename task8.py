import pickle

def start_task8():
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)
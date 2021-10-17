import pickle


def start_task8():
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    with open('simp.pickle', 'rb') as handle:
        simp = pickle.load(handle)
    print('here')

if __name__ == '__main__':
    start_task8()
#• Task 5 Implement a program which, given the filename of a query image which may not be in the database and a latent
#semantics file, identifies and visualizes the most similar n images under the selected latent semantics.
import pickle
import utilities
import csv


def start_task5():
    with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)

    with open('simp.pickle', 'rb') as handle:
        simp = pickle.load(handle)


if __name__ == '__main__':
    start_task5()
import sys
import os

import task0
import task1

if __name__ == '__main__':
    metadata_file = os.path.join(os.getcwd(), 'metadata.pickle')
    if not os.path.isfile(metadata_file):
        task0.generate_pickle(metadata_file)

    while(True):
        task = int(input('Enter task number: '))
        if task == 1:
            task1.start_task1()

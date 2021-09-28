import sys
import os

import task0
import task1
import task2

task_map = [None, task1.start_task1, task2.start_task2]

if __name__ == '__main__':
    metadata_file = os.path.join(os.getcwd(), 'metadata.pickle')
    if not os.path.isfile(metadata_file):
        task0.generate_pickle(metadata_file)

    while(True):
        task = int(input('Enter task number: '))
        try:
            task_map[task]()
        except:
            print('Invalid task selection, please select from 1-9')

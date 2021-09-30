import sys
import os

import task0
import task1
import task2
import task3
import task4

task_map = [task0.start_task0, task1.start_task1, task2.start_task2, task3.start_task3, task4.start_task4]

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

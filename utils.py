import os

def create_dirs_if_not_exist(path):
    isExist = os.path.exists(path)

    if not isExist:
        os.makedirs(path)
        print("The new directory is created!")
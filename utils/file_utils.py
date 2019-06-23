import os
'''
dir_name : dir_name (inc. path) that will be created ( full-path name )
'''
def create_folder(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
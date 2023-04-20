import logging
def mkfolder(folder):
    import os
    try:
        os.makedirs(folder)
    except FileExistsError:
        logging.warning("The {0} folder already exists  \n The old {0} folder will be renamed (to {0}~)".format(folder))
        import shutil
        if os.path.exists(folder+'~'):
            shutil.rmtree(folder+'~')
        os.system('mv {} {}'.format(folder, folder+'~'))
        os.makedirs(folder)
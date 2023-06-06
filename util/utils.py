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


def process_gpuID(gpuID):

    if type(gpuID) == str:
        gpuID_list = list(set(gpuID.split(',')))
        gpuID_list = list(map(int,gpuID_list))
        ngpus = len(gpuID_list)

    elif type(gpuID) == tuple or type(gpuID) == list:
        gpuID_list = gpuID
        ngpus = len(gpuID)
        gpuID = ','.join(map(str, gpuID_list))

    elif type(gpuID) == int:
        ngpus = 1
        gpuID_list = [gpuID]
        gpuID = str(gpuID)
    
    return ngpus, gpuID, gpuID_list
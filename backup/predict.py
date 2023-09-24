#!/usr/bin/env python3
import os, sys
#from spIsoNet.util.image import *
from spIsoNet.util.metadata import MetaData,Label,Item
from spIsoNet.util.dict2attr import idx2list
import logging
from spIsoNet.preprocessing.img_processing import normalize
import mrcfile
import numpy as np

def predict(args):

    logger = logging.getLogger('predict')
    if args.log_level == "debug":
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
    else:
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
    logging.info('\n\n######Isonet starts predicting######\n')
    if args.gpuID is None:
        raise ValueError("Please provide gpuID")
    args.gpuID = str(args.gpuID)
    args.ngpus = len(list(set(args.gpuID.split(','))))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID
    logger.info('percentile:{}'.format(args.normalize_percentile))
    logger.info('gpuID:{}'.format(args.gpuID))

    from spIsoNet.models.network import Net
    network = Net()
    network.load(args.model)

    if args.batch_size is None:
        args.batch_size = 4 * args.ngpus
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    md = MetaData()
    md.read(args.star_file)
    if not 'rlnCorrectedTomoName' in md.getLabels():
        md.addLabels('rlnCorrectedTomoName')
        for it in md:
            md._setItemValue(it,Label('rlnCorrectedTomoName'),None)
    args.tomo_idx = idx2list(args.tomo_idx)

    for it in md:
        if args.tomo_idx is None or str(it.rlnIndex) in args.tomo_idx:
            if args.use_deconv_tomo and "rlnDeconvTomoName" in md.getLabels() and it.rlnDeconvTomoName not in [None,'None']:
                tomo_file = it.rlnDeconvTomoName
            else:
                tomo_file = it.rlnMicrographName
            tomo_root_name = os.path.splitext(os.path.basename(tomo_file))[0]
            if os.path.isfile(tomo_file):
                

                with mrcfile.open(tomo_file) as mrcData:
                    real_data = mrcData.data.astype(np.float32)*-1
                    voxelsize = mrcData.voxel_size
                    real_data = normalize(real_data,percentile=args.normalize_percentile)
                

                tomo_out_name = '{}/{}_corrected.mrc'.format(args.output_dir,tomo_root_name)

                outData = network.predict_map(real_data, args.output_dir, cube_size = args.cube_size, crop_size= args.crop_size)
                outData = normalize(outData,percentile=args.normalize_percentile)
                with mrcfile.new(tomo_out_name, overwrite=True) as output_mrc:
                    output_mrc.set_data(-outData)
                    output_mrc.voxel_size = voxelsize
                #network.predict_tomo(args,tomo_file,output_file=tomo_out_name)
                md._setItemValue(it,Label('rlnCorrectedTomoName'),tomo_out_name)
        md.write(args.star_file)


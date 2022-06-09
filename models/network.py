from .unet import Unet
import torch
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import RichProgressBar
from .data_sequence import get_datasets, Predict_sets
import mrcfile
from IsoNet.preprocessing.img_processing import normalize
import numpy as np
import logging
from rich.progress import track
from IsoNet.util.toTile import reform3D


class Net:
    def __init__(self, gpuId = [0,1,2,3]):
        self.model = Unet()
        self.gpuId = gpuId
        self.batch_size = len(gpuId)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)
    
    def save(self, path):
        state = self.model.state_dict()
        torch.save(state, path)

    def train(self, data_path):

        train_dataset, val_dataset = get_datasets(data_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,persistent_workers=True,
                                                num_workers=self.batch_size, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,persistent_workers=True,
                                                pin_memory=True, num_workers=self.batch_size)
        self.model.train()
        trainer = pl.Trainer(
            gpus=self.gpuId,
            max_epochs=3,
            strategy = 'dp',
            #enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
            callbacks=RichProgressBar(),
            num_sanity_val_steps=0
        )
        trainer.fit(self.model, train_loader, val_loader)

    def predict(self, mrc_list, result_dir, iter_count, normalize_percentile = True):    

        bench_dataset = Predict_sets(mrc_list)
        bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=4, num_workers=1)

        model = torch.nn.DataParallel(self.model.cuda())
        model.eval()

        predicted = []
        with torch.no_grad():
            for _, val_data in enumerate(bench_loader):
                res=model(val_data).cpu().detach().numpy().astype(np.float32)
                for item in res:
                    it = item.squeeze(0)
                    predicted.append(it)
        
        for i,mrc in enumerate(mrc_list):
            root_name = mrc.split('/')[-1].split('.')[0]
            outData = normalize(predicted[i], percentile = normalize_percentile)
            with mrcfile.new('{}/{}_iter{:0>2d}.mrc'.format(result_dir, root_name, iter_count-1), overwrite=True) as output_mrc:
                output_mrc.set_data(-outData)
    
    def predict_tomo(self, args, one_tomo, output_file=None):
    #predict one tomogram in mrc format INPUT: mrc_file string OUTPUT: output_file(str) or <root_name>_corrected.mrc

        root_name = one_tomo.split('/')[-1].split('.')[0]

        if output_file is None:
            if os.path.isdir(args.output_file):
                output_file = args.output_file+'/'+root_name+'_corrected.mrc'
            else:
                output_file = root_name+'_corrected.mrc'

        logging.info('predicting:{}'.format(root_name))

        with mrcfile.open(one_tomo) as mrcData:
            real_data = mrcData.data.astype(np.float32)*-1
            voxelsize = mrcData.voxel_size

        real_data = normalize(real_data,percentile=args.normalize_percentile)
        data=np.expand_dims(real_data,axis=-1)
        reform_ins = reform3D(data)
        data = reform_ins.pad_and_crop_new(args.cube_size,args.crop_size)

        N = args.batch_size
        num_patches = data.shape[0]
        if num_patches%N == 0:
            append_number = 0
        else:
            append_number = N - num_patches%N
        data = np.append(data, data[0:append_number], axis = 0)
        num_big_batch = data.shape[0]//N
        outData = np.zeros(data.shape)

        logging.info("total batches: {}".format(num_big_batch))


        model = torch.nn.DataParallel(self.model.cuda())
        model.eval()
        with torch.no_grad():
            for i in track(range(num_big_batch), description="Processing..."):
                in_data = torch.from_numpy(np.transpose(data[i*N:(i+1)*N],(0,4,1,2,3)))
                outData[i*N:(i+1)*N] = np.transpose( model(in_data).cpu().detach().numpy().astype(np.float32), (0,2,3,4,1) )

        outData = outData[0:num_patches]

        outData=reform_ins.restore_from_cubes_new(outData.reshape(outData.shape[0:-1]), args.cube_size, args.crop_size)

        outData = normalize(outData,percentile=args.normalize_percentile)
        with mrcfile.new(output_file, overwrite=True) as output_mrc:
            output_mrc.set_data(-outData)
            output_mrc.voxel_size = voxelsize

        logging.info('Done predicting')
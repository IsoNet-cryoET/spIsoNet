from .unet import Unet
import torch
import pytorch_lightning as pl
import os
#from pytorch_lightning.callbacks import RichProgressBar
from .data_sequence import get_datasets, Predict_sets
import mrcfile
from IsoNet.preprocessing.img_processing import normalize
import numpy as np
import logging
#from rich.progress import track
from IsoNet.util.toTile import reform3D
import sys
from tqdm import tqdm
from IsoNet.preprocessing.simulate import apply_wedge_dcube
from IsoNet.preprocessing.simulate import apply_wedge

class Net:
    def __init__(self, metrics=None):
    #    pass

    #def initialize(self):
        self.model = Unet(metrics=metrics)
        #self.model.half()
        # self.model = self.model.to(memory_format=torch.channels_last)
        #print(self.model)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint)

    def load_jit(self, path):
        #Using the TorchScript format, you will be able to load the exported model and run inference without defining the model class.
        self.model = torch.jit.load(path)
    
    def save(self, path):
        state = self.model.state_dict()
        torch.save(state, path)
    def save_jit(self, path):
        model_scripted = torch.jit.script(self.model) # Export to TorchScript
        model_scripted.save(path) # Save

    def train(self, data_path, gpuID=[0,1,2,3], learning_rate=3e-4, batch_size=None, epochs = 10, steps_per_epoch=200, acc_grad =False):
        self.model.learning_rate = learning_rate

        train_batches = int(steps_per_epoch*0.9)
        val_batches = steps_per_epoch - train_batches
        if acc_grad:
            logging.info("use accumulate gradient to reduce GPU memory consumption")
            batch_size = batch_size//2
            acc_batches = 2
            train_batches = train_batches * 2
            val_batches = val_batches * 2
        else:
            acc_batches = 1

        train_dataset, val_dataset = get_datasets(data_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,persistent_workers=True,
                                                num_workers=batch_size, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,persistent_workers=True,
                                                pin_memory=True, num_workers=batch_size, drop_last=True)

        self.model.train()

        trainer = pl.Trainer(
            accumulate_grad_batches=acc_batches,
            accelerator='gpu',
            precision=16,
            #devices=gpuID,
            num_nodes=1,
            max_epochs=epochs,
            limit_train_batches = train_batches,
            limit_val_batches = val_batches,
            strategy = 'dp',
            enable_progress_bar=True,
            logger=False,
            enable_checkpointing=False,
            #callbacks=EpochProgressBar(),
            num_sanity_val_steps=0
        )
        trainer.fit(self.model, train_loader, val_loader)        
        return  self.model.metrics

    def predict(self, mrc_list, result_dir, iter_count, mw3d=None):    

        bench_dataset = Predict_sets(mrc_list)
        bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=4, num_workers=1)

        model = torch.nn.DataParallel(self.model.cuda())
        model.eval()

        predicted = []
        with torch.no_grad():
            for _, val_data in enumerate(bench_loader):
                    res = model(val_data) 
                    miu = res.cpu().detach().numpy().astype(np.float32)
                    for item in miu:
                        it = item.squeeze(0)
                        predicted.append(it)
        for i,mrc in enumerate(mrc_list):
            root_name = mrc.split('/')[-1].split('.')[0]

            #outData = normalize(predicted[i], percentile = normalize_percentile)
            file_name = '{}/{}_iter{:0>2d}.mrc'.format(result_dir, root_name, iter_count-1)
 
            output_data = predicted[i]
            if mw3d is not None:
                with mrcfile.open(mrc, 'r') as origional_mrc:
                    input_data= origional_mrc.data
                output_data = apply_wedge(output_data, mw3d=mw3d, ld1=0, ld2=1) + input_data#+ apply_wedge(input_data, mw3d=mw3d, ld1=1, ld2=0) 


            with mrcfile.new(file_name, overwrite=True) as output_mrc:
                output_mrc.set_data(output_data)
    
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
            real_data = mrcData.data.astype(np.float32)
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
            for i in tqdm(range(num_big_batch), file=sys.stdout):#track(range(num_big_batch), description="Processing..."):
                in_data = torch.from_numpy(np.transpose(data[i*N:(i+1)*N],(0,4,1,2,3)))
                output = model(in_data)
                out_tmp = output.cpu().detach().numpy().astype(np.float32)
                out_tmp = apply_wedge_dcube(out_tmp, None, mw3d="fouriermask.mrc",ld1=0, ld2=1)
                out_tmp = np.transpose(out_tmp, (0,2,3,4,1) )
                outData[i*N:(i+1)*N] = out_tmp  + data[i*N:(i+1)*N]

        outData = outData[0:num_patches]

        outData=reform_ins.restore_from_cubes_new(outData.reshape(outData.shape[0:-1]), args.cube_size, args.crop_size)

        #outData = normalize(outData,percentile=args.normalize_percentile)
        with mrcfile.new(output_file, overwrite=True) as output_mrc:
            output_mrc.set_data(outData.astype(np.float32))
            output_mrc.voxel_size = voxelsize

        logging.info('Done predicting')
    
    def predict_map(self, halfmap,halfmap_origional,fsc3d_full, fsc3d, output_file, cube_size = 64, crop_size=96, batch_size = 4, voxel_size = 1.1):
    #predict one tomogram in mrc format INPUT: mrc_file string OUTPUT: output_file(str) or <root_name>_corrected.mrc


        logging.info('Inference')

        real_data = halfmap
        data=np.expand_dims(real_data,axis=-1)
        reform_ins = reform3D(data)
        data = reform_ins.pad_and_crop_new(cube_size,crop_size)

        N = batch_size
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
            for i in tqdm(range(num_big_batch), file=sys.stdout):#track(range(num_big_batch), description="Processing..."):
                in_data = torch.from_numpy(np.transpose(data[i*N:(i+1)*N],(0,4,1,2,3)))
                #print(in_data)
                output = model(in_data)
                out_tmp = output.cpu().detach().numpy().astype(np.float32)
                #out_tmp = apply_wedge_dcube(out_tmp, mw3d=fsc3d,ld1=0, ld2=1)
                out_tmp = np.transpose(out_tmp, (0,2,3,4,1) )

                #out_data_tmp = np.transpose(data[i*N:(i+1)*N], (0,4,1,2,3))
                #out_data_tmp = apply_wedge_dcube(out_data_tmp, mw3d=fsc3d,ld1=1, ld2=0)
                #out_data_tmp = np.transpose(out_data_tmp, (0,2,3,4,1) )


                outData[i*N:(i+1)*N] = out_tmp#  + out_data_tmp

        outData = outData[0:num_patches]

        outData=reform_ins.restore_from_cubes_new(outData.reshape(outData.shape[0:-1]), cube_size, crop_size)
        print(np.std(outData))
        #outData = apply_wedge(normalize(outData),mw3d=fsc3d_full, ld1=0, ld2=1)
        
        outData = apply_wedge(outData,mw3d=fsc3d_full, ld1=0, ld2=1)
        print(np.std(outData))
        outData += halfmap_origional# apply_wedge(normalize(halfmap),mw3d=fsc3d_full, ld1=1, ld2=0) #0.5*real_data#
        print(np.std(outData))
        print(np.std(real_data))

        #outData = normalize(outData,percentile=args.normalize_percentile)
        with mrcfile.new(output_file, overwrite=True) as output_mrc:
            output_mrc.set_data(outData.astype(np.float32))
            output_mrc.voxel_size = voxel_size


        logging.info('Done predicting')

#!/usr/bin/env python3
import fire
import logging
import os, sys, traceback
from spIsoNet.util.dict2attr import Arg,check_parse,idx2list
from fire import core
from spIsoNet.util.metadata import MetaData,Label,Item

class ISONET:
    """
    ISONET: Train on tomograms and restore missing-wedge\n
    for detail discription, run one of the following commands:


    spisonet.py fsc3d -h
    spisonet.py map_refine -h
    """


    def refine(self, 
                   input: str,
                   aniso_file: str = None, 
                   mask: str=None, 

                   gpuID: str=None, 
                   alpha: float=1,
                   ncpus: int=16, 
                   output_dir: str="isonet_maps",
                   pretrained_model: str=None,

                   epochs: int=50,
                   n_subvolume: int=1000, 
                   cube_size: int=64,
                   predict_crop_size: int=80,
                   batch_size: int=None, 
                   acc_batches: int=1,
                   learning_rate: float=3e-4
                   ):

        """
        \ntrain neural network to correct preffered orientation\n
        spisonet.py map_refine half.mrc FSC3D.mrc mask.mrc [--gpuID] [--ncpus] [--output_dir] [--fsc_file]...
        :param input: Input name
        :param mask: Filename of a user-provided mask
        :param gpuID: The ID of gpu to be used during the training.
        :param ncpus: Number of cpu.
        :param output_dir: The name of directory to save output maps
        :param fsc_file: 3DFSC file if not set, isonet will generate one.
        :param epochs: Number of epochs for each iteration. This value can be increase (maybe to 10) to get (maybe) better result.
        :param n_subvolume: Number of subvolumes 
        :param predict_crop_size: The size of subvolumes, should be larger then the cube_size
        :param cube_size: Size of cubes for training, should be divisible by 16, e.g. 32, 64, 80.
        :param batch_size: Size of the minibatch. If None, batch_size will be the max(2 * number_of_gpu,4). batch_size should be divisible by the number of gpu.
        :param acc_batches: If this value is set to 2 (or more), accumulate gradiant will be used to save memory consumption.  
        :param learning_rate: learning rate. Default learning rate is 3e-4 while previous spIsoNet tomography used 3e-4 as learning rate
        """
        #TODO
        #mixed precision does not work for torch.FFT
        mixed_precision = False

        from spIsoNet.util.utils import mkfolder
        from spIsoNet.preprocessing.img_processing import normalize
        from spIsoNet.bin.map_refine import map_refine
        from spIsoNet.util.utils import process_gpuID
        from multiprocessing import cpu_count
        import mrcfile
        import numpy as np

        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])   
        
        if gpuID is None:
            import torch
            gpu_list = list(range(torch.cuda.device_count()))
            gpuID=','.join(map(str, gpu_list))
            print("using all GPUs in this node: %s" %gpuID)  

        ngpus, gpuID, gpuID_list = process_gpuID(gpuID)

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=gpuID

        if batch_size is None:
            if ngpus == 1:
                batch_size = 4
            else:
                batch_size = 2 * len(gpuID_list)

        cpu_system = cpu_count()
        if cpu_system < ncpus:
            logging.info("requested number of cpus is more than the number of the cpu cores in the system")
            logging.info(f"setting ncpus to {cpu_system}")
            ncpus = cpu_system

        mkfolder(output_dir,remove=False)

        output_base = input.split('/')[-1]
        output_base = output_base.split('.')[:-1]
        output_base = "".join(output_base)

        with mrcfile.open(input, 'r') as mrc:
            half_map = normalize(mrc.data,percentile=False)
            voxel_size = mrc.voxel_size.x
            if voxel_size == 0:
                voxel_size = 1
        logging.info("voxel_size {}".format(voxel_size))

        if mask is None:
            mask_vol = np.ones(half_map.shape, dtype = np.float32)
            logging.warning("No mask is provided, please consider providing a soft mask")
        else:
            with mrcfile.open(mask, 'r') as mrc:
                mask_vol = mrc.data
        if aniso_file is None:
            logging.warning("No fsc3d is provided. Only denoising")
            fsc3d = np.ones(half_map.shape, dtype = np.float32)
        else:
            with mrcfile.open(aniso_file, 'r') as mrc:
                fsc3d = mrc.data

        map_refine(half_map, mask_vol, fsc3d, alpha = alpha,  voxel_size=voxel_size, output_dir=output_dir, 
                   output_base=output_base, mixed_precision=mixed_precision, epochs = epochs,
                   n_subvolume=n_subvolume, cube_size=cube_size, pretrained_model=pretrained_model,
                   batch_size = batch_size, acc_batches = acc_batches,predict_crop_size=predict_crop_size,gpuID=gpuID, learning_rate=learning_rate)
        
        logging.info("Finished")

    def refine_n2n(self, 
                   h1: str,
                   h2: str,
                   aniso_file: str = None, 
                   mask: str=None, 

                   gpuID: str=None, 
                   alpha: float=1,
                   beta: float=0.5,
                   ncpus: int=16, 
                   output_dir: str="isonet_maps",
                   pretrained_model: str=None,
                   limit_res: str="None",

                   ref_map: str="None",
                   ref_resolution: float=10,

                   epochs: int=50,
                   n_subvolume: int=1000, 
                   cube_size: int=64,
                   predict_crop_size: int=80,
                   batch_size: int=None, 
                   acc_batches: int=1,
                   learning_rate: float=3e-4
                   ):

        """
        \ntrain neural network to correct preffered orientation\n
        spisonet.py map_refine half.mrc FSC3D.mrc mask.mrc [--gpuID] [--ncpus] [--output_dir] [--fsc_file]...
        :param input: Input name
        :param mask: Filename of a user-provided mask
        :param gpuID: The ID of gpu to be used during the training.
        :param ncpus: Number of cpu.
        :param output_dir: The name of directory to save output maps
        :param fsc_file: 3DFSC file if not set, isonet will generate one.
        :param epochs: Number of epochs for each iteration. This value can be increase (maybe to 10) to get (maybe) better result.
        :param n_subvolume: Number of subvolumes 
        :param predict_crop_size: The size of subvolumes, should be larger then the cube_size
        :param cube_size: Size of cubes for training, should be divisible by 16, e.g. 32, 64, 80.
        :param batch_size: Size of the minibatch. If None, batch_size will be the max(2 * number_of_gpu,4). batch_size should be divisible by the number of gpu.
        :param acc_batches: If this value is set to 2 (or more), accumulate gradiant will be used to save memory consumption.  
        :param learning_rate: learning rate. Default learning rate is 3e-4 while previous spIsoNet tomography used 3e-4 as learning rate
        """
        #TODO
        #mixed precision does not work for torch.FFT
        mixed_precision = False

        from spIsoNet.util.utils import mkfolder
        from spIsoNet.preprocessing.img_processing import normalize
        from spIsoNet.bin.map_refine import map_refine_n2n
        from spIsoNet.util.utils import process_gpuID
        from multiprocessing import cpu_count
        import mrcfile
        import numpy as np

        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])   
        
        if gpuID is None:
            import torch
            gpu_list = list(range(torch.cuda.device_count()))
            gpuID=','.join(map(str, gpu_list))
            print("using all GPUs in this node: %s" %gpuID)  

        ngpus, gpuID, gpuID_list = process_gpuID(gpuID)

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=gpuID

        if batch_size is None:
            if ngpus == 1:
                batch_size = 4
            else:
                batch_size = 2 * len(gpuID_list)

        cpu_system = cpu_count()
        if cpu_system < ncpus:
            logging.info("requested number of cpus is more than the number of the cpu cores in the system")
            logging.info(f"setting ncpus to {cpu_system}")
            ncpus = cpu_system

        mkfolder(output_dir,remove=False)

        output_base1 = h1.split('/')[-1]
        output_base1 = output_base1.split('.')[:-1]
        output_base1 = "".join(output_base1)

        output_base2 = h2.split('/')[-1]
        output_base2 = output_base2.split('.')[:-1]
        output_base2 = "".join(output_base2)

        with mrcfile.open(h1, 'r') as mrc:
            halfmap1 = normalize(mrc.data,percentile=False)
            voxel_size = mrc.voxel_size.x
            if voxel_size == 0:
                voxel_size = 1
        with mrcfile.open(h2, 'r') as mrc:
            halfmap2 = normalize(mrc.data,percentile=False)

        logging.info("voxel_size {}".format(voxel_size))

        if mask is None:
            mask_vol = np.ones(halfmap1.shape, dtype = np.float32)
            logging.warning("No mask is provided, please consider providing a soft mask")
        else:
            with mrcfile.open(mask, 'r') as mrc:
                mask_vol = mrc.data

        if aniso_file is None:
            logging.warning("No fsc3d is provided. Only denoising")
            fsc3d = np.ones(halfmap1.shape, dtype = np.float32)
        else:
            with mrcfile.open(aniso_file, 'r') as mrc:
                fsc3d = mrc.data

        if limit_res in ["None", None]:
            #from spIsoNet.util.FSC import recommended_resolution
            limit_res = None #recommended_resolution(fsc3d, voxel_size, threshold = 0.143)
        else:
            limit_res = float(limit_res)

        if ref_map not in ['None',None]:
            logging.info(f"Incoorporating low resolution information of the reference {ref_map}\n\
                         until the --ref_resolution {ref_resolution}")
            from spIsoNet.util.FSC import combine_map_F
            halfmap1 = combine_map_F(ref_map,halfmap1,ref_resolution,voxel_size,mask_data=mask)
            halfmap2 = combine_map_F(ref_map,halfmap2,ref_resolution,voxel_size,mask_data=mask)
            
        map_refine_n2n(halfmap1,halfmap2, mask_vol, fsc3d, alpha = alpha,beta=beta,  voxel_size=voxel_size, output_dir=output_dir, 
                   output_base1=output_base1, output_base2=output_base2, mixed_precision=mixed_precision, epochs = epochs,
                   n_subvolume=n_subvolume, cube_size=cube_size, pretrained_model=pretrained_model,
                   batch_size = batch_size, acc_batches = acc_batches,predict_crop_size=predict_crop_size,gpuID=gpuID, learning_rate=learning_rate, limit_res= limit_res)
        if limit_res is not None:
            logging.info("combining")
            self.combine_map(f"{output_dir}/corrected_{output_base1}_filtered.mrc",h1, out_map=f"{output_dir}/corrected_{output_base1}.mrc",limit_res=limit_res,mask_file= mask)
            self.combine_map(f"{output_dir}/corrected_{output_base2}_filtered.mrc",h2, out_map=f"{output_dir}/corrected_{output_base2}.mrc",limit_res=limit_res,mask_file= mask)

        logging.info("Finished")

    def whitening(self, 
                    h1: str,
                    o: str = "whitening.mrc",
                    mask: str=None, 
                    high_res: float=3,
                    low_res: float=10,
                    ):
        import numpy as np
        import mrcfile
        from numpy.fft import fftshift, fftn, ifftn

        with mrcfile.open(h1,'r') as mrc:
            input_map = mrc.data
            nz,ny,nx = input_map.shape
            voxel_size = mrc.voxel_size.x
            if voxel_size == 0:
                voxel_size = 1
            logging.info("voxel_size",voxel_size)

        if mask is not None:
            with mrcfile.open(mask,'r') as mrc:
                mask = mrc.data
            input_map_masked = input_map * mask
        else:
            input_map_masked = input_map

        limit_r_low = int(voxel_size * nz / low_res)
        limit_r_high = int(voxel_size * nz / high_res)

        # power spectrum
        f1 = fftshift(fftn(input_map_masked))
        ret = (np.real(np.multiply(f1,np.conj(f1)))**0.5).astype(np.float32)

        #vet whitening filter
        r = np.arange(nz)-nz//2
        [Z,Y,X] = np.meshgrid(r,r,r)
        index = np.round(np.sqrt(Z**2+Y**2+X**2))

        F_curve = np.zeros(nz//2)
        F_map = np.zeros_like(ret)
        for i in range(nz//2):
            F_curve[i] = np.average(ret[index==i])

        eps = 1e-4
        
        for i in range(nz//2):
            if i > limit_r_low:
                if i < limit_r_high:
                    F_map[index==i] = F_curve[limit_r_low]/(F_curve[i]+eps)
                else:
                    F_map[index==i] = 1
            else:
                F_map[index==i] = 1

        # apply filter
        F_input = fftn(input_map)
        out = ifftn(F_input*fftshift(F_map))
        out =  np.real(out).astype(np.float32)

        # with mrcfile.new("F.mrc", overwrite=True) as mrc:
        #     mrc.set_data(F_map)

        with mrcfile.new(o, overwrite=True) as mrc:
            mrc.set_data(out)
            mrc.voxel_size = voxel_size

    def combine_map(self, 
                    low_map: str, 
                    high_map: str, 
                    out_map:str,
                    threshold_res: float,
                    mask_file: str=None):
        import numpy as np
        import mrcfile
        with mrcfile.open(low_map,'r') as mrc:
            low_data = mrc.data
            voxel_size = mrc.voxel_size.x
            if voxel_size == 0:
                voxel_size = 1
            logging.info(f"voxel_size {voxel_size}")

        with mrcfile.open(high_map,'r') as mrc:
            high_data = mrc.data

        with mrcfile.open(mask_file,'r') as mrc:
            mask = mrc.data
    
        from spIsoNet.util.FSC import combine_map_F
        out_data = combine_map_F(low_data, high_data, threshold_res, voxel_size, mask_data=mask)

        with mrcfile.new(out_map,overwrite=True) as mrc:
            mrc.set_data(out_data)
            mrc.voxel_size = voxel_size
        


    def fsc3d(self, 
                   h: str,
                   h2: str, 
                   mask: str=None, 
                   o: str="FSC3D.mrc",
                   ncpus: int=16, 
                   limit_res: float=None, 
                   cone_sampling_angle: float=10,
                   keep_highres: bool = False
                   ):

        """
        \ntrain neural network to correct preffered orientation\n
        spisonet.py map_refine half1.mrc half2.mrc mask.mrc [--gpuID] [--ncpus] [--output_dir] [--fsc_file]...
        :param h: Input name of half1
        :param h2: Input name of half2
        :param mask: Filename of a user-provided mask
        :param ncpus: Number of cpu.
        :param limit_res: The resolution limit for recovery, default is the resolution of the map.
        :param fsc_file: 3DFSC file if not set, isonet will generate one.
        :param cone_sampling_angle: Angle for 3D fsc sampling for spIsoNet generated 3DFSC. spIsoNet default is 10 degrees, the default for official 3DFSC is 20 degrees.
        """
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])   

        from spIsoNet.preprocessing.img_processing import normalize
        import numpy as np
        from multiprocessing import cpu_count
        import mrcfile

        from spIsoNet.util.FSC import get_FSC_map, ThreeD_FSC, recommended_resolution

        cpu_system = cpu_count()
        if cpu_system < ncpus:
            logging.info("requested number of cpus is more than the number of the cpu cores in the system")
            logging.info(f"setting ncpus to {cpu_system}")
            ncpus = cpu_system

        with mrcfile.open(h, 'r') as mrc:
            half1 = normalize(mrc.data,percentile=False)
            voxel_size = mrc.voxel_size.x
            if voxel_size == 0:
                voxel_size = 1

        with mrcfile.open(h2, 'r') as mrc:
            half2 = normalize(mrc.data,percentile=False)


        if mask is None:
            mask_vol = np.ones(half1.shape, dtype = np.float32)
            logging.warning("No mask is provided, please consider providing a soft mask")
        else:
            with mrcfile.open(mask, 'r') as mrc:
                mask_vol = mrc.data

        FSC_map = get_FSC_map([half1, half2], mask_vol)
        if limit_res is None:
            limit_res = recommended_resolution(FSC_map, voxel_size, threshold=0.143)
            logging.info("Global resolution at FSC={} is {}".format(0.143, limit_res))

        limit_r = int( (2.*voxel_size) / limit_res * (half1.shape[0]/2.) + 1)
        logging.info("Limit resolution to {} for spIsoNet 3D FSC calculation. You can also tune this paramerter with --limit_res .".format(limit_res))

        logging.info("calculating fast 3DFSC, this will take few minutes")
        fsc3d = ThreeD_FSC(FSC_map, limit_r,angle=float(cone_sampling_angle), n_processes=ncpus)
        if keep_highres:
            from spIsoNet.util.FSC import get_sphere
            fsc3d = np.maximum(1-get_sphere(limit_r-2,fsc3d.shape[0]),fsc3d)
        with mrcfile.new(o, overwrite=True) as mrc:
            mrc.set_data(fsc3d.astype(np.float32))
        logging.info("voxel_size {}".format(voxel_size))

         
         
    def fsd3d(self, 
                   star_file: str, 
                   map_dim: int,
                   apix: float=1.0,
                   o: str="FSD3D.mrc",
                   low_res: float=100,
                   high_res: float=1, 
                   number_subset: float=10000,
                   grid_size: float=64,
                   sym: str = "c1"
                   ):

        """
        \nFourier shell density, reimpliment from cryoEF, relies on relion star file and also relion installation.\n
        spisonet.py map_refine half1.mrc half2.mrc mask.mrc [--gpuID] [--ncpus] [--output_dir] [--fsc_file]...
        :param h: Input name of half1
        :param h2: Input name of half2
        :param mask: Filename of a user-provided mask
        :param ncpus: Number of cpu.
        :param limit_res: The resolution limit for recovery, default is the resolution of the map.
        :param fsc_file: 3DFSC file if not set, isonet will generate one.
        :param cone_sampling_angle: Angle for 3D fsc sampling for spIsoNet generated 3DFSC. spIsoNet default is 10 degrees, the default for official 3DFSC is 20 degrees.
        """
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])   
        import numpy as np
        import mrcfile
        from subprocess import check_output

        s = f"thetacol=`grep _rlnAngleTilt {star_file} | awk '{{print $2}}' | sed 's/#//'`;\
        phicol=`grep _rlnAngleRot {star_file} | awk '{{print $2}}' | sed 's/#//'`;\
        cat {star_file} | grep @ | awk -v thetacolvar=${{thetacol}} -v phicolvar=${{phicol}} '{{if (NF>2) print $thetacolvar, $phicolvar}}' > theta_phi_angles.dat"
        check_output(s, shell=True)

        coordinates = np.loadtxt("theta_phi_angles.dat",dtype=np.float32)
        index = np.random.choice(coordinates.shape[0], number_subset)#, replace=True)
        coordinates = coordinates[index]

        def generate_grid(coordinates, grid_size):
            # Create a 3D grid for calculation
            x = np.linspace(-1, 1, grid_size).astype(np.float32)
            y = np.linspace(-1, 1, grid_size).astype(np.float32)
            z = np.linspace(-1, 1, grid_size).astype(np.float32)
            x, y, z = np.meshgrid(x, y, z)

            # Initialize the matrix to store the sum of circles
            circle_matrix = np.zeros_like(x)

            # Define the distance threshold
            threshold = 5**0.5/grid_size
            coord = np.radians(coordinates)
            cos_coord = np.cos(coord)
            sin_coord = np.sin(coord)

            # This need to think carefully
            surface_points = np.stack([sin_coord[:,0]*cos_coord[:,1],sin_coord[:,0]*sin_coord[:,1],cos_coord[:,0]])
            #surface_points = np.stack([cos_coord[:,0],sin_coord[:,0]*sin_coord[:,1],sin_coord[:,0]*cos_coord[:,1]])
            #print(surface_points)
            pixels = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

            distances = np.abs(np.matmul(pixels,surface_points))
            circle = np.where(distances <= threshold, 1, 0)
            circle = np.sum(circle, axis=1)
            circle_matrix =  circle.reshape(grid_size, grid_size, grid_size)
            circle_matrix =  np.transpose(circle_matrix,[2,0,1])
            return circle_matrix
        
        out_mat = generate_grid(coordinates[:1000],grid_size)
        for i in range(1,10):
            out_mat += generate_grid(coordinates[i*1000:(i+1)*1000],grid_size)
        out_mat = out_mat / number_subset

        with mrcfile.new(o, overwrite=True) as mrc:
            mrc.set_data(out_mat.astype(np.float32))

        s = f"relion_image_handler --i {o} --o {o} --sym {sym}"
        check_output(s,shell=True)

        with mrcfile.open(o,'r') as mrc:
            input_map = mrc.data
            nz,ny,nx = input_map.shape
            voxel_size = mrc.voxel_size.x
            if voxel_size == 0:
                voxel_size = 1

        #apix_small = apix * map_dim / grid_size
        r = np.arange(nz)-nz//2
        limit_r_low = int(apix * nz / low_res)
        limit_r_high = int(apix * nz / high_res)


        [Z,Y,X] = np.meshgrid(r,r,r)
        index = np.round(np.sqrt(Z**2+Y**2+X**2))

        F_map = np.zeros_like(input_map)
        eps = 1e-4

        for i in range(nz//2):
            if i > limit_r_low:
                if i < limit_r_high:
                    F_map[index==i] = 1.1/np.max(input_map[index==i])
                else:
                    F_map[index==i] = 0
            else:
                F_map[index==i] = 1

        out_map = F_map*input_map
        for i in range(nz//2):
            if i > limit_r_low:
                if i > limit_r_high:
                    out_map[index==i] = 0
            else:
                out_map[index==i] = 1

        out_map[out_map>1] = 1
        import skimage
        out_map = skimage.transform.resize(out_map, [map_dim,map_dim,map_dim])
        with mrcfile.new(o, overwrite=True) as mrc:
            mrc.set_data(out_map)
            mrc.voxel_size = voxel_size


    def angular_whiten(self, in_name,out_name,resolution_initial, limit_resolution):
        import mrcfile
        from numpy.fft import fftn,fftshift,ifftn
        from spIsoNet.util.FSC import apply_F_filter
        import numpy as np
        import skimage


        with mrcfile.open(in_name) as mrc:
            in_map = mrc.data.copy()
            voxel_size = mrc.voxel_size.x

        F_map = fftn(in_map)
        shifted_F_map = fftshift(F_map)
        F_power = np.real(np.multiply(shifted_F_map,np.conj(shifted_F_map)))**0.5
        F_power = F_power.astype(np.float32)
        nz = 64
        downsampled_F_map = skimage.transform.resize(F_power, [nz,nz,nz])

        low_r = nz * voxel_size / resolution_initial
        high_r = nz * voxel_size / limit_resolution
        print(low_r)
        print(high_r)

        x, y, z = np.meshgrid(np.arange(nz), np.arange(nz), np.arange(nz))

        direction_vectors = np.stack([x - nz // 2, y - nz // 2, z - nz // 2], axis=-1)
        direction_vectors = direction_vectors.reshape((nz**3,3))

        d = np.linalg.norm(direction_vectors, axis=-1)
        condition = np.logical_and((d > low_r), (d < high_r))

        distances = d[condition]
        direction_vectors = direction_vectors[condition]

        normalized_vectors = direction_vectors / distances[:,np.newaxis]
        normalized_vectors = normalized_vectors.astype(np.float32)
        normalized_matrix = np.matmul(normalized_vectors, np.transpose(normalized_vectors))

        half_angle_rad = np.radians(5)
        half_angle_cos = np.cos(half_angle_rad)
        normalized_matrix = (np.abs(normalized_matrix) > half_angle_cos).astype(np.float32)

        sum_matrix = np.sum(normalized_matrix, axis = -1)

        input_flatterned_matrix = downsampled_F_map.reshape((nz**3,1))[condition]

        out_values = np.matmul(normalized_matrix, input_flatterned_matrix).squeeze()/sum_matrix

        out_matrix = np.zeros((nz**3,), dtype = np.float32)
        out_matrix[condition] = 1/out_values
        #out_matrix[d<=low_r] = np.max(out_matrix[d<=(low_r+1)])
        out_matrix = out_matrix.reshape((nz,nz,nz))
        # with mrcfile.new("tmp.mrc", overwrite=True) as mrc:
        #     mrc.set_data(out_matrix.astype(np.float32))
        map_dim = in_map.shape[0]
        out_matrix = skimage.transform.resize(out_matrix, [map_dim,map_dim,map_dim])

        # with mrcfile.new("tmp.mrc", overwrite=True) as mrc:
        #     mrc.set_data(out_matrix.astype(np.float32))

        transformed_data = np.real(ifftn(F_map*fftshift(out_matrix))).astype(np.float32)
        transformed_data =  (transformed_data-np.mean(transformed_data))/np.std(transformed_data)
        transformed_data =   transformed_data*np.std(in_map) + np.mean(in_map)
        reverse_filter = (out_matrix<0.0000001).astype(int)
        in_map_filtered = apply_F_filter(in_map,reverse_filter)
        with mrcfile.new(out_name, overwrite=True) as mrc:
            mrc.set_data((transformed_data+in_map_filtered).astype(np.float32))
            mrc.voxel_size = tuple([voxel_size]*3) 

    '''
    def map_refine_multi(self, half1_file, half2_file, mask_file, fsc_file, limit_res, output_dir="isonet_maps", gpuID=0, n_subvolume=50, crop_size=96, cube_size=64, weighting=False):
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
            ,datefmt="%H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
        half1_list = half1_file.split(',')
        print(half1_list)
        half2_list = half2_file.split(',')
        mask_list = mask_file.split(',')
        import mrcfile
        half1 = []
        half2 = []
        mask = []
        for half1_file in half1_list:
            with mrcfile.open(half1_file, 'r') as mrc:
                half1.append(mrc.data)
                voxel_size = mrc.voxel_size.x
        for half2_file in half2_list:
            with mrcfile.open(half2_file, 'r') as mrc:
                half2.append(mrc.data)
        for mask_file in mask_list:
            with mrcfile.open(mask_file, 'r') as mrc:
                mask.append(mrc.data)
        with mrcfile.open(fsc_file, 'r') as mrc:
            fsc3d = mrc.data
        logging.info("voxel_size {}".format(voxel_size))
        from spIsoNet.bin.map_refine import map_refine_multi
        from spIsoNet.util.utils import mkfolder
        mkfolder(output_dir)
        logging.info("processing half map1")
        map_refine_multi(half1, mask, fsc3d, voxel_size=voxel_size, limit_res = limit_res, output_dir = output_dir, output_base="half1", weighting = weighting, n_subvolume = n_subvolume, cube_size = cube_size, crop_size = crop_size)
        logging.info("processing half map2")
        map_refine_multi(half2, mask, fsc3d, voxel_size=voxel_size, limit_res = limit_res, output_dir = output_dir, output_base="half2", weighting = weighting, n_subvolume = n_subvolume, cube_size = cube_size, crop_size = crop_size)
        logging.info("Two independent half maps are saved in {}. Please use other software for postprocessing and try difference B factors".format(output_dir))
    '''

    # def predict(self, star_file: str, model: str, output_dir: str='./corrected_tomos', gpuID: str = None, cube_size:int=64,
    # crop_size:int=96,use_deconv_tomo=True, batch_size:int=None,normalize_percentile: bool=True,log_level: str="info", tomo_idx=None):
    #     """
    #     \nPredict tomograms using trained model\n
    #     spisonet.py predict star_file model [--gpuID] [--output_dir] [--cube_size] [--crop_size] [--batch_size] [--tomo_idx]
    #     :param star_file: star for tomograms.
    #     :param output_dir: file_name of output predicted tomograms
    #     :param model: path to trained network model .h5
    #     :param gpuID: (0,1,2,3) The gpuID to used during the training. e.g 0,1,2,3.
    #     :param cube_size: (64) The tomogram is divided into cubes to predict due to the memory limitation of GPUs.
    #     :param crop_size: (96) The side-length of cubes cropping from tomogram in an overlapping patch strategy, make this value larger if you see the patchy artifacts
    #     :param batch_size: The batch size of the cubes grouped into for network predicting, the default parameter is four times number of gpu
    #     :param normalize_percentile: (True) if normalize the tomograms by percentile. Should be the same with that in refine parameter.
    #     :param log_level: ("debug") level of message to be displayed, could be 'info' or 'debug'
    #     :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16
    #     :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead.
    #     :raises: AttributeError, KeyError
    #     """
    #     d = locals()
    #     d_args = Arg(d)
    #     from spIsoNet.bin.predict import predict

    #     if d_args.log_level == "debug":
    #         logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
    #         datefmt="%m-%d %H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])
    #     else:
    #         logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
    #         datefmt="%m-%d %H:%M:%S",level=logging.INFO,handlers=[logging.StreamHandler(sys.stdout)])
    #     try:
    #         predict(d_args)
    #     except:
    #         error_text = traceback.format_exc()
    #         f =open('log.txt','a+')
    #         f.write(error_text)
    #         f.close()
    #         logging.error(error_text)
    
    # def resize(self, star_file:str, apix: float=15, out_folder="tomograms_resized"):
    #     '''
    #     This function rescale the tomograms to a given pixelsize
    #     '''
    #     md = MetaData()
    #     md.read(star_file)
        
    #     from scipy.ndimage import zoom
    #     import mrcfile
    #     if not os.path.isdir(out_folder):
    #         os.makedirs(out_folder)
    #     for item in md._data:
    #         ori_apix = item.rlnPixelSize
    #         tomo_name = item.rlnMicrographName
    #         zoom_factor = float(ori_apix)/apix
    #         new_tomo_name = "{}/{}".format(out_folder,os.path.basename(tomo_name))
    #         with mrcfile.open(tomo_name) as mrc:
    #             data = mrc.data
    #         print("scaling: {}".format(tomo_name))
    #         new_data = zoom(data, zoom_factor,order=3, prefilter=False)
    #         #new_data = rescale(data, zoom_factor,order=3, anti_aliasing = True)
    #         #new_data = new_data.astype(np.float32)

    #         with mrcfile.new(new_tomo_name,overwrite=True) as mrc:
    #             mrc.set_data(new_data)
    #             mrc.voxel_size = apix

    #         item.rlnPixelSize = apix
    #         print(new_tomo_name)
    #         item.rlnMicrographName = new_tomo_name
    #         print(item.rlnMicrographName)
    #     md.write(os.path.splitext(star_file)[0] + "_resized.star")
    #     print("scale_finished")

    def check(self):
        logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',
        datefmt="%m-%d %H:%M:%S",level=logging.DEBUG,handlers=[logging.StreamHandler(sys.stdout)])

        from spIsoNet.bin.predict import predict
        from spIsoNet.bin.refine import run
        import skimage
        import PyQt5
        import tqdm
        logging.info('spIsoNet --version 1.0 alpha installed')
        logging.info(f"checking gpu speed")
        from spIsoNet.bin.verify import verify
        fp16, fp32 = verify()
        logging.info(f"time for mixed/half precsion and single precision are {fp16} and {fp32}. ")
        logging.info(f"The first number should be much smaller than the second one, if not please check whether cudnn, cuda, and pytorch versions match.")

    def gui(self):
        """
        \nGraphic User Interface\n
        """
        import spIsoNet.gui.Isonet_star_app as app
        app.main()

def Display(lines, out):
    text = "\n".join(lines) + "\n"
    out.write(text)

def pool_process(p_func,chunks_list,ncpu):
    from multiprocessing import Pool
    with Pool(ncpu,maxtasksperchild=1000) as p:
        # results = p.map(partial_func,chunks_gpu_num_list,chunksize=1)
        results = list(p.map(p_func,chunks_list))
    # return results

def main():
    core.Display = Display
    # logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',datefmt="%m-%d %H:%M:%S",level=logging.INFO)
    if len(sys.argv) > 1:
        check_parse(sys.argv[1:])
    fire.Fire(ISONET)


if __name__ == "__main__":
    exit(main())

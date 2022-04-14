**Instructions**
0. prerequisite
	a. follow the instructions of cyclegan in README_cyclegan.md and make sure torch and other dependencies are installed
	b. install nngraph (https://github.com/torch/nngraph) and matio (https://github.com/soumith/matio-ffi.torch)

1. torch part
	a. download dataset
		cd DeepToF_release/src/GAN
		bash datasets/download_dataset.sh mpi_correction_corrnormamp2depth_albedoaug_zpass
	Note: you may first download just a subset of the dataset for quick evaluation
		bash datasets/download_dataset.sh mpi_correction_corrnormamp2depth_albedoaug_zpass_test

	b. download pretrained model
		bash pretrained_models/download_model.sh my_resnet_lite_my_imageGAN_128

	c. run sample test code
		bash job_example.sh

	d. results will be saved to ./results as .mat files

2. matlab part
	load('results/my_resnet_lite_my_imageGAN_128_pretrained/latest_test/images/fake_B/000001.mat') %results are stored as .mat files under [results_folder]/fake_B
	x = x*10 %scale from [0,1] back to distance in meters
	imagesc(squeeze(x(1,:,:))) %display tofnet output


**Change log (src/GAN, based on https://github.com/junyanz/CycleGAN/tree/be7db60aa57478c37feb90ef9b9fa096ad090369)**

/data:
	added support for .mat files
	relaxed the requirement that input must have three input/output channels
	added other image processing functions for correlation inputs, such as amplitude normalization, and flip, rotation, noise…

/dataset:
	modified download_dataset.sh, can be ran from the parent folder

/models:
	added our own generators G and discriminators D

/pretrained_models:
	modified download_model.sh

/utils:
	added support for correlation/phase images pre-processing


**Contact**
Shuochen Su (shuochsu@cs.ubc.ca)
08/02/2018
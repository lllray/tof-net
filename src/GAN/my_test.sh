# # training code
# DATA_ROOT=./datasets/mpi_correction_corrnormamp2depth_albedoaug_zpass niter=50 save_epoch_freq=200 mat=1 input_nc=5 output_nc=1 \
# 	model=pix2pix align_data=1 name=my_resnet_lite_my_imageGAN_128 which_model_netG=my_resnet_lite which_model_netD=my_imageGAN_128 \
# 	use_GAN=1 lambda_A=10 norm=none \
# 	noise=1 lr=0.00005 resize_or_crop=resize_and_crop_ratio_rand mask_nan=1 tv_strength=1e-4 tv_weight_scheme=1 tv_weight_scale=1 \
# 	manualSeed=1 inv_lambda=1 display_port=1234 th train.lua;

# testing code (on real tintin test set)
DATA_ROOT=./datasets/royal_itof_test mat=1 input_nc=5 output_nc=1 \
	model=pix2pix align_data=1 name=new_gan_128 which_model_netG=my_resnet_lite which_model_netD=my_imageGAN_128 \
        use_GAN=1 lambda_A=1 norm=none \
	noise=0 results_dir=./results/ tv_strength=1e-4 tv_weight_scheme=1 tv_weight_scale=1 \
	display_port=1234 phase=test_0429_1 th test.lua;	

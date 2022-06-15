# deep-tof

0. prerequisite
	a. follow the instructions of cyclegan in README_cyclegan.md and make sure torch and other dependencies are installed
	b. install nngraph (https://github.com/torch/nngraph) and matio (https://github.com/soumith/matio-ffi.torch)

1. torch part
	a. test

        cp -r /mnt/6t_1/lx/itof_datasets/model_dir/new_gan_128 ./checkpoints

        sh my_test.sh

	b. train

        sh my_train.sh

2. matlab part

        run CompareResults.m with matlab



--------------------------------------------------------------------------------
-- Configure options
--------------------------------------------------------------------------------

local options = {}
-- options for train
local opt_train = {
   DATA_ROOT = '',         -- path to images (should have subfolders 'train', 'val', etc)
   manualSeed = 0,         -- 0: not applying manualSeed; 1, applying manualSeed
   batchSize = 1,          -- # images in batch
   loadSize = 143,         -- scale images to this size
   fineSize = 128,         -- then crop to this size
   normalizeAmp = 0,       -- apply image-wise amplitude normalization
   ngf = 64,               -- # of gen filters in first conv layer
   ndf = 64,               -- # of discrim filters in first conv layer
   mat = 0,                -- mat=0 for loading/saving in image format, mat=1 for .mat files (higher precision)
   -- input_type = 'corr',    -- 'corr' | 'phase', if latter, apply util.corr2phase to imA and assert input_nc=2
   input_nc = 3,           -- # of input image channels
   output_nc = 3,          -- # of output image channels
   niter = 100,            -- # of iter at starting learning rate
   niter_decay = 100,      -- # of iter to linearly decay learning rate to zero
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- # of examples per epoch. math.huge for full dataset
   flip = 1,               -- if flip the images for data argumentation
   rotate = 1,             -- if rotate the images for data argumentation
   noise = 0,              -- add a bit noise to imB (imA) to improve GAN stability. 0: no noise; 1: imA+imB; 2: imA; 3: imB
   mask_nan = 0,           -- whether to mask nan pixels in fakeB, if 1, nan pixels won't count
   display_id = 10,        -- display window id.
   display_winsize = 128,  -- display window size
   display_port = 1234,    -- display server port
   display_freq = 25,      -- display the current results every display_freq iterations
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = '',              -- name of the experiment, should generally be passed on the command line
   which_direction = 'AtoB',    -- AtoB or BtoA
   phase = 'train',             -- train, val, test, etc
   nThreads = 4,                -- # threads for loading data
   save_epoch_freq = 1,         -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = 5000,     -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   print_freq = 50,             -- print the debug information every print_freq iterations
   save_display_freq = 2500,    -- save the current display of results every save_display_freq_iterations
   continue_train = 0,          -- if continue training, load the latest model: 1: true, 0: false
   continue_from_batch = 1,     -- if continue training, specify which epoch to start with
   serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
   checkpoints_dir = './checkpoints', -- models are saved here
   cache_dir = './cache',             -- cache files are saved here
   cudnn = 1,                         -- set to 0 to not use cudnn
   use_GAN = 1,                       -- use GAN or not
   which_model_netD = 'basic',        -- selects model to use for netD
   which_model_netG = 'resnet_6blocks',   -- selects model to use for netG
   criterionL1 = 'AbsCriterion',  -- criterion for L1 loss
   tv_strength = 0,               -- total variation smoothness of the output
   tv_weight_scheme = 0,          -- methods to apply weights in the tv term. 0: no weights, just vanila tv; 1: using gradient of amplitude image (eq. 8); 2: use instance normalization instead of batch norm
   tv_weight_scale = 1,           -- amplify the weight source (e.g. amplitude map) before feeding into tv+amp layer
   norm = 'instance',             -- batch, instance or no normalization
   n_layers_D = 3,                -- only used if which_model_netD=='n_layers'
   content_loss = 'pixel',        -- content loss type: pixel, vgg
   layer_name = 'pixel',          -- layer used in content loss (e.g. relu4_2)
   lambda_A = 10.0,               -- weight for cycle loss (A -> B -> A)
   lambda_B = 10.0,               -- weight for cycle loss (B -> A -> B)
   inv_lambda = 0,                -- 0: mul lambda to G's loss/grad; 1: mul 1/lambda to D's loss/grad
   model = 'cycle_gan',           -- which mode to run. 'cycle_gan', 'pix2pix', 'bigan', 'content_gan'
   use_lsgan = 1,                 -- if 1, use least square GAN, if 0, use vanilla GAN
   align_data = 0,                -- if > 0, use the dataloader for where the images are aligned
   pool_size = 50,                -- the size of image buffer that stores previously generated images
   resize_or_crop = 'resize_and_crop',  -- resizing/cropping strategy: resize_and_crop | crop | scale_width | scale_height
   identity = 0,                  -- use identity mapping. Setting opt.identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set opt.identity = 0.1
   use_optnet = 0,                -- use optnet to save GPU memory during test
}

-- options for test
local opt_test = {
  DATA_ROOT = '',           -- path to images (should have subfolders 'train', 'val', etc)
  loadSize = 128,           -- scale images to this size
  fineSize = 240,           --  then crop to this size
  normalizeAmp = 0,         -- apply image-wise amplitude normalization
  flip = 0,                 -- horizontal mirroring data augmentation
  rotate = 0,               -- if rotate the images for data argumentation
  noise = 0,                -- add a bit noise to imB (imA) to improve GAN stability
  display = 1,              -- display samples while training. 0 = false
  display_id = 200,         -- display window id.
  gpu = 1,                  -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  how_many = 'all',         -- how many test images to run (set to all to run on every image found in the data/phase folder)
  phase = 'test',           -- train, val, test, etc
  aspect_ratio = 1.0,       -- aspect ratio of result images
  norm = 'instance',        -- batchnorm or isntance norm
  name = '',                -- name of experiment, selects which model to run, should generally should be passed on command line
  input_nc = 3,             -- #  of input image channels
  output_nc = 3,            -- #  of output image channels
  tv_strength = 0,          -- total variation smoothness of the output
  tv_weight_scheme = 0,     -- methods to apply weights in the tv term. 0: no weights, just vanila tv; 1: using gradient of amplitude image (eq. 8); 2: use instance normalization instead of batch norm
  serial_batches = 1,       -- if 1, takes images in order to make batches, otherwise takes them randomly
  cudnn = 1,                -- set to 0 to not use cudnn (untested)
  checkpoints_dir = './checkpoints', -- loads models from here
  cache_dir = './cache',             -- cache files are saved here
  results_dir='./results/',          -- saves results here
  which_epoch = 'latest',            -- which epoch to test? set to 'latest' to use latest cached model
  model = 'cycle_gan',               -- which mode to run. 'cycle_gan', 'pix2pix', 'bigan', 'content_gan'; to use pretrained model, select `one_direction_test`
  align_data = 0,                    -- if > 0, use the dataloader for pix2pix
  which_direction = 'AtoB',          -- AtoB or BtoA
  resize_or_crop = 'scale_height',   -- resizing/cropping strategy: resize_and_crop | crop | scale_width | scale_height
}

--------------------------------------------------------------------------------
-- util functions
--------------------------------------------------------------------------------
function options.clone(opt)
  local copy = {}
  for orig_key, orig_value in pairs(opt) do
    copy[orig_key] = orig_value
  end
  return copy
end

function options.parse_options(mode)
  if mode == 'train' then
    opt = opt_train
    opt.test = 0
  elseif mode == 'test' then
    opt = opt_test
    opt.test = 1
  else
    print("Invalid option [" .. mode .. "]")
    return nil
  end

  -- one-line argument parser. parses enviroment variables to override the defaults
  for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
  if mode == 'test' then
    opt.nThreads = 1
    opt.continue_train = 1
    opt.batchSize = 1  -- test code only supports batchSize=1
  end

  -- print by keys
  keyset = {}
  for k,v in pairs(opt) do
    table.insert(keyset, k)
  end
  table.sort(keyset)
  print("------------------- Options -------------------")
  for i,k in ipairs(keyset) do
    print(('%+25s: %s'):format(k, opt[k]))
  end
  print("-----------------------------------------------")

  -- save opt to checkpoints
  paths.mkdir(opt.checkpoints_dir)
  paths.mkdir(paths.concat(opt.checkpoints_dir, opt.name))
  opt.visual_dir = paths.concat(opt.checkpoints_dir, opt.name, 'visuals')
  paths.mkdir(opt.visual_dir)
  -- save opt to the disk
  fd = io.open(paths.concat(opt.checkpoints_dir, opt.name, 'opt_' .. mode .. '.txt'), 'w')
  for i,k in ipairs(keyset) do
    fd:write(("%+25s: %s\n"):format(k, opt[k]))
  end
  fd:close()

  return opt
end


return options

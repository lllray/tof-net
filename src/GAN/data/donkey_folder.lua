
--[[
    This data loader is a modified version of the one from dcgan.torch
    (see https://github.com/soumith/dcgan.torch/blob/master/data/donkey_folder.lua).
    Copyright (c) 2016, Deepak Pathak [See LICENSE file for details]
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.
    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
local matio = require 'matio'
paths.dofile('dataset.lua')
-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- Check for existence of opt.data
if opt.DATA_ROOT then
  opt.data = paths.concat(opt.DATA_ROOT, opt.phase)
else
  print(os.getenv('DATA_ROOT'))
  opt.data = paths.concat(os.getenv('DATA_ROOT'), opt.phase)
end

if not paths.dirp(opt.data) then
    error('Did not find directory: ' .. opt.data)
end

-- a cache file of the training metadata (if doesnt exist, will be created)
local cache_prefix = opt.data:gsub('/', '_')
os.execute(('mkdir -p %s'):format(opt.cache_dir))
local trainCache = paths.concat(opt.cache_dir, cache_prefix .. '_trainCache.t7')

--------------------------------------------------------------------------------------------
local input_nc = opt.input_nc -- input channels
local output_nc = opt.output_nc -- output channels, assume that output_nc <= input_nc
local loadSize   = {input_nc, opt.loadSize}
local sampleSize = {input_nc, opt.fineSize}

local function loadImage(path)
  local input
  if opt.mat == 0 then
    input = image.load(path, input_nc, 'float')
  else
    input = matio.load(path,'im_pair'):float():contiguous()
    if input:dim() == 2 then
      input = input:view(1,input:size(1),input:size(2))
    end
  end
  local h = input:size(2)
  local w = input:size(3)

  local imA = image.crop(input, 0, 0, w/2, h)
  local imB = image.crop(input, w/2, 0, w, h)
  if opt.resize_or_crop == 'resize_and_crop' then
    imA = image.scale(imA, loadSize[2], loadSize[2])
    imB = image.scale(imB, loadSize[2], loadSize[2])
  elseif opt.resize_or_crop == 'resize_and_crop_ratio' then
    imA = image.scale(imA, loadSize[2]/h*w/2, loadSize[2]) -- keep the aspect ratio
    imB = image.scale(imB, loadSize[2]/h*w/2, loadSize[2])
  elseif opt.resize_or_crop == 'resize_and_crop_ratio_rand' then
    local loadSizeH = torch.uniform(loadSize[2],h)
    imA = image.scale(imA, loadSizeH/h*w/2, loadSizeH) -- keep the aspect ratio and perturb loadSize
    imB = image.scale(imB, loadSizeH/h*w/2, loadSizeH)
  end
  imB = imB:narrow(1,1,output_nc)

  if input_nc==3 and output_nc==3 then
    local perm = torch.LongTensor{3, 2, 1}
    imA = imA:index(1, perm)
    imB = imB:index(1, perm)
  end
  if opt.normalizeAmp>0 then
    local util = require 'util/util'
    imA = util.normalize(imA)
    -- imB = util.normalize(imB)
  end
  imA[imA:ne(imA)] = 0.5
  imB[imB:ne(imB)] = 0 -- handling nan values, corr should be 0.5 in [0,1], depth should be 0 in [0,1]
  imA[imA:gt(1)] = 0.5
  imB[imB:gt(1)] = 0 -- handling inf values
  imA = imA:mul(2):add(-1)
  imB = imB:mul(2):add(-1)
  --print(('imA max: %.4f; imA min: %.4f; imB max: %.4f; imB min: %.4f'):format(imA:max(),imA:min(),imB:max(),imB:min()))
  assert(imA:max()<=1,"A: badly scaled inputs")
  assert(imA:min()>=-1,"A: badly scaled inputs")
  assert(imB:max()<=1,"B: badly scaled inputs")
  assert(imB:min()>=-1,"B: badly scaled inputs")

  -- imA:mul(0.9) -- shrink a bit to avoid saturation

  local oW = sampleSize[2]
  local oH = sampleSize[2]
  local iH = imA:size(2)
  local iW = imA:size(3)

  if opt.resize_or_crop == 'resize_and_crop' or opt.resize_or_crop == 'resize_and_crop_ratio' or opt.resize_or_crop == 'resize_and_crop_ratio_rand' then
    if iH~=oH then
      h1 = math.ceil(torch.uniform(1e-2, iH-oH))
    end

    if iW~=oW then
      w1 = math.ceil(torch.uniform(1e-2, iW-oW))
    end
    if iH ~= oH or iW ~= oW then
      imA = image.crop(imA, w1, h1, w1 + oW, h1 + oH)
      imB = image.crop(imB, w1, h1, w1 + oW, h1 + oH)
    end
  elseif (opt.resize_or_crop == 'crop') then
    local w = math.min(math.min(oH, iH),iW)
    w = math.floor(w/4)*4
    local x = math.floor(torch.uniform(0, iW - w))
    local y = math.floor(torch.uniform(0, iH - w))
    imA = image.crop(imA, x, y, x+w, y+w)
    imB = image.crop(imB, x, y, x+w, y+w)
  elseif opt.resize_or_crop == 'scale_width' then
    w = oW
    h = torch.floor(iH * oW/iW)
    imA = image.scale(imA, w, h)
    imB = image.scale(imB, w, h)
  elseif opt.resize_or_crop == 'scale_height' then
    h = oH
    w = torch.floor(iW * oH / iH)
    imA = image.scale(imA, w, h)
    imB = image.scale(imB, w, h)
  end
  
  if opt.flip == 1 then
    local is_hflip = torch.uniform()
    local is_vflip = torch.uniform()
    if is_hflip < 0.5 then
        imA = image.hflip(imA)
        imB = image.hflip(imB)
    end
    if is_vflip < 0.5 then
        imA = image.vflip(imA)
        imB = image.vflip(imB)
    end
  end
  if opt.rotate == 1 then
    local is_rot = torch.uniform()
    if is_rot >= 0.25 and is_rot < 0.5 then
        imA = image.rotate(imA,math.pi/2)
        imB = image.rotate(imB,math.pi/2)
    elseif is_rot >= 0.5 and is_rot < 0.75 then
        imA = image.rotate(imA,math.pi)
        imB = image.rotate(imB,math.pi)
    elseif is_rot >= 0.75 then
        imA = image.rotate(imA,math.pi*3/2)
        imB = image.rotate(imB,math.pi*3/2)
    end
  end
  if opt.noise > 0 then
    -- add a bit noise to GT depth (hack for better gan)
    local noise_sigma = torch.rand(1)*0.01 --[0,0.01)
    local pnoise = torch.Tensor(imB:size()):zero()
    if opt.noise == 1 or opt.noise == 3 then
      pnoise:normal(0,noise_sigma[1])
      imB = torch.clamp(imB+pnoise,-1,1)
    end
    -- also to imA
    if opt.noise == 1 or opt.noise == 2 then
      noise_sigma = torch.rand(1)*0.005 --[0,0.005)
      pnoise = torch.Tensor(imA:size()):zero()
      pnoise:normal(0,noise_sigma[1])
      imA = torch.clamp(imA+pnoise,-1,1)
    end
  end

  local concatenated = torch.cat(imA,imB,1)

  return concatenated
end


local function loadSingleImage(path)
    local im
    if opt.mat == 0 then
      im = image.load(path, input_nc, 'float')
    else
      im = matio.load(path, 'im'):float():contiguous()
    end
    if opt.resize_or_crop == 'resize_and_crop' then
      im = image.scale(im, loadSize[2], loadSize[2])
    end
    if input_nc == 3 then
      local perm = torch.LongTensor{3, 2, 1}
      im = im:index(1, perm)--:mul(256.0): brg, rgb
    end
    if opt.normalizeAmp>0 then
      local util = require 'util/util'
      im = util.normalize(im)
    end
    im[im:ne(im)] = 0 -- handling nan values
    im[im:gt(1)] = 0 -- handling inf values
    im = im:mul(2):add(-1)
    assert(im:max()<=1,"A: badly scaled inputs")
    assert(im:min()>=-1,"A: badly scaled inputs")

    local oW = sampleSize[2]
    local oH = sampleSize[2]
    local iH = im:size(2)
    local iW = im:size(3)
    if (opt.resize_or_crop == 'resize_and_crop' ) then
      local h1, w1 = 0, 0
      if iH~=oH then
        h1 = math.ceil(torch.uniform(1e-2, iH-oH))
      end
      if iW~=oW then
        w1 = math.ceil(torch.uniform(1e-2, iW-oW))
      end
      if iH ~= oH or iW ~= oW then
        im = image.crop(im, w1, h1, w1 + oW, h1 + oH)
      end
    elseif (opt.resize_or_crop == 'combined') then
      local sH = math.min(math.ceil(oH * torch.uniform(1+1e-2, 2.0-1e-2)), iH-1e-2)
      local sW = math.min(math.ceil(oW * torch.uniform(1+1e-2, 2.0-1e-2)), iW-1e-2)
      local h1 = math.ceil(torch.uniform(1e-2, iH-sH))
      local w1 = math.ceil(torch.uniform(1e-2, iW-sW))
      im = image.crop(im, w1, h1, w1 + sW, h1 + sH)
      im = image.scale(im, oW, oH)
    elseif (opt.resize_or_crop == 'crop') then
      local w = math.min(math.min(oH, iH),iW)
      w = math.floor(w/4)*4
      local x = math.floor(torch.uniform(0, iW - w))
      local y = math.floor(torch.uniform(0, iH - w))
      im = image.crop(im, x, y, x+w, y+w)
    elseif (opt.resize_or_crop == 'scale_width') then
      w = oW
      h = torch.floor(iH * oW/iW)
      im = image.scale(im, w, h)
    elseif (opt.resize_or_crop == 'scale_height') then
      h = oH
      w = torch.floor(iW * oH / iH)
      im = image.scale(im, w, h)
    end

    if opt.flip == 1 and torch.uniform() > 0.5 then
        im = image.hflip(im)
    end
    if opt.flip == 1 then
      local is_hflip = torch.uniform()
      local is_vflip = torch.uniform()
      if is_hflip < 0.5 then
        im = image.hflip(im)
      end
      if is_vflip < 0.5 then
        im = image.vflip(im)
      end
    end
    if opt.rotate == 1 then
      local is_rot = torch.uniform()
      if is_rot >= 0.25 and is_rot < 0.5 then
        im = image.rotate(im,math.pi/2)
      elseif is_rot >= 0.5 and is_rot < 0.75 then
        im = image.rotate(im,math.pi)
      elseif is_rot >= 0.75 then
        im = image.rotate(im,math.pi*3/2)
      end
    end

  return im

end

-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook_singleimage = function(self, path)
   collectgarbage()
  --  print('load single image')
   local im = loadSingleImage(path)
   return im
end

-- function that loads images that have juxtaposition
-- of two images from two domains
local trainHook_doubleimage = function(self, path)
  -- print('load double image')
  collectgarbage()

  local im = loadImage(path)
  return im
end


if opt.align_data > 0 then
  sample_nc = input_nc + output_nc
  trainHook = trainHook_doubleimage
else
  sample_nc = input_nc
  trainHook = trainHook_singleimage
end

trainLoader = dataLoader{
    paths = {opt.data},
    loadSize = {input_nc, loadSize[2], loadSize[2]},
    sampleSize = {sample_nc, sampleSize[2], sampleSize[2]},
    split = 100,
    serial_batches = opt.serial_batches,
    verbose = true
 }

trainLoader.sampleHookTrain = trainHook
collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")
end

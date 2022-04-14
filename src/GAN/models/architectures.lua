require 'nngraph'
require './TotalVariation'


----------------------------------------------------------------------------
local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0.01) -- goodfellow deeplearningbook, for norm=none only
  elseif name:find('Normalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end


normalization = nil

function set_normalization(norm)
if norm == 'instance' then
  require 'util.InstanceNormalization'
  print('use InstanceNormalization')
  normalization = nn.InstanceNormalization
elseif norm == 'batch' then
  print('use SpatialBatchNormalization')
  normalization = nn.SpatialBatchNormalization
elseif norm == 'none' then
  print('use identity mapping')
  normalization = nn.Identity
end
end

function defineG(input_nc, output_nc, ngf, which_model_netG, tv_strength, tv_weight_scheme, tv_weight_scale, nz, arch)
  local netG = nil
  local tv = tv_strength or 0
  local tv_weight_sch = tv_weight_scheme or 0
  if tv_weight_sch == 2 then
    require 'util.InstanceNormalization'
  end
  if     which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
  elseif which_model_netG == "unet128" then netG = defineG_unet128(input_nc, output_nc, ngf)
  elseif which_model_netG == "unet256" then netG = defineG_unet256(input_nc, output_nc, ngf)
  elseif which_model_netG == "resnet_6blocks" then netG = defineG_resnet_6blocks(input_nc, output_nc, ngf)
  elseif which_model_netG == "resnet_9blocks" then netG = defineG_resnet_9blocks(input_nc, output_nc, ngf)
  elseif which_model_netG == "my_resnet" then netG = defineG_my_resnet(input_nc, output_nc, ngf, tv, tv_weight_sch, tv_weight_scale)
  elseif which_model_netG == "my_resnet_9blocks" then netG = defineG_my_resnet_9blocks(input_nc, output_nc, ngf, tv, tv_weight_sch, tv_weight_scale)
  elseif which_model_netG == "my_resnet_lite" then netG = defineG_my_resnet_lite(input_nc, output_nc, ngf, tv, tv_weight_sch, tv_weight_scale)
  elseif which_model_netG == "my_resnet_lite_noskip" then netG = defineG_my_resnet_lite_noskip(input_nc, output_nc, ngf, tv, tv_weight_sch, tv_weight_scale)
  else error("unsupported netG model")
    end
  netG:apply(weights_init)

  return netG
end

function defineD(input_nc, ndf, which_model_netD, n_layers_D, use_sigmoid, tv_weight_scheme)
  local netD = nil
  local tv_weight_sch = tv_weight_scheme or 0
  if     which_model_netD == "basic" then netD = defineD_basic(input_nc, ndf, use_sigmoid)
  elseif which_model_netD == "imageGAN" then netD = defineD_imageGAN(input_nc, ndf, use_sigmoid)
  elseif which_model_netD == "n_layers" then netD = defineD_n_layers(input_nc, ndf, n_layers_D, use_sigmoid)
  elseif which_model_netD == "my_imageGAN_128" then netD = defineD_my_imageGAN_128(input_nc, ndf, use_sigmoid, tv_weight_sch)
  elseif which_model_netD == "my_imageGAN_192" then netD = defineD_my_imageGAN_192(input_nc, ndf, use_sigmoid, tv_weight_sch)
  else error("unsupported netD model")
  end
  netD:apply(weights_init)

  return netD
end


function defineG_encoder_decoder(input_nc, output_nc, ngf)
    -- input is (nc) x 256 x 256
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - normalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - normalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- normalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1

    local d1 = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d2 = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d3 = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d4 = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d5 = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1) - normalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d6 = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1) - normalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d7 = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) - normalization(ngf)
    -- input is (ngf) x128 x 128
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    local o1 = d8 - nn.Tanh()

    local netG = nn.gModule({e1},{o1})
    return netG
end


function defineG_unet128(input_nc, output_nc, ngf)
    local netG = nil
    -- input is (nc) x 128 x 128
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 64 x 64
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - normalization(ngf * 2)
    -- input is (ngf * 2) x 32 x 32
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - normalization(ngf * 4)
    -- input is (ngf * 4) x 16 x 16
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- normalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1

    local d1_ = e7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e6} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e5} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e4} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - normalization(ngf * 4)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e3} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - normalization(ngf * 2)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e2} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - normalization(ngf)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e1} - nn.JoinTable(2)

    local d7 = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 128 x 128

    local o1 = d7 - nn.Tanh()
    local netG = nn.gModule({e1},{o1})
    return netG
end


function defineG_unet256(input_nc, output_nc, ngf)
    local netG = nil
    -- input is (nc) x 256 x 256
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - normalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - normalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- - normalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1

    local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e7} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - normalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - normalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - normalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e2} - nn.JoinTable(2)
    local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - normalization(ngf)
    -- input is (ngf) x128 x 128
    local d7 = {d7_,e1} - nn.JoinTable(2)
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256

    local o1 = d8 - nn.Tanh()
    local netG = nn.gModule({e1},{o1})
    return netG
end

--------------------------------------------------------------------------------
-- Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
--------------------------------------------------------------------------------

local function build_conv_block(dim, padding_type)
  local conv_block = nn.Sequential()
  local p = 0
  if padding_type == 'reflect' then
    conv_block:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  elseif padding_type == 'replicate' then
    conv_block:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
  elseif padding_type == 'zero' then
    p = 1
  end
  conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
  conv_block:add(normalization(dim))
  conv_block:add(nn.ReLU(true))
  if padding_type == 'reflect' then
    conv_block:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  elseif padding_type == 'replicate' then
    conv_block:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
  end
  conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
  conv_block:add(normalization(dim))
  return conv_block
end


local function build_res_block(dim, padding_type)
  local conv_block = build_conv_block(dim, padding_type)
  local res_block = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(conv_block)
  concat:add(nn.Identity())
  
  res_block:add(concat):add(nn.CAddTable())
  return res_block
end

function defineG_resnet_6blocks(input_nc, output_nc, ngf)
  padding_type = 'reflect'
  local ks = 3
  local netG = nil
  local f = 7
  local p = (f - 1) / 2
  local data = -nn.Identity()
  local e1 = data - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(input_nc, ngf, f, f, 1, 1) - normalization(ngf) - nn.ReLU(true)
  local e2 = e1 - nn.SpatialConvolution(ngf, ngf*2, ks, ks, 2, 2, 1, 1) - normalization(ngf*2) - nn.ReLU(true)
  local e3 = e2 - nn.SpatialConvolution(ngf*2, ngf*4, ks, ks, 2, 2, 1, 1) - normalization(ngf*4) - nn.ReLU(true)
  local d1 = e3 - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
  - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
  local d2 = d1 - nn.SpatialFullConvolution(ngf*4, ngf*2, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf*2) - nn.ReLU(true)
  local d3 = d2 - nn.SpatialFullConvolution(ngf*2, ngf, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf) - nn.ReLU(true)
  local d4 = d3 - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(ngf, output_nc, f, f, 1, 1) - nn.Tanh()
  netG = nn.gModule({data},{d4})
  return netG
end

function defineG_resnet_9blocks(input_nc, output_nc, ngf)
  padding_type = 'reflect'
  local ks = 3
  local netG = nil
  local f = 7
  local p = (f - 1) / 2
  local data = -nn.Identity()
  local e1 = data - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(input_nc, ngf, f, f, 1, 1) - normalization(ngf) - nn.ReLU(true)
  local e2 = e1 - nn.SpatialConvolution(ngf, ngf*2, ks, ks, 2, 2, 1, 1) - normalization(ngf*2) - nn.ReLU(true)
  local e3 = e2 - nn.SpatialConvolution(ngf*2, ngf*4, ks, ks, 2, 2, 1, 1) - normalization(ngf*4) - nn.ReLU(true)
  local d1 = e3 - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
  - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
 - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
  local d2 = d1 - nn.SpatialFullConvolution(ngf*4, ngf*2, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf*2) - nn.ReLU(true)
  local d3 = d2 - nn.SpatialFullConvolution(ngf*2, ngf, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf) - nn.ReLU(true)
  local d4 = d3 - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(ngf, output_nc, f, f, 1, 1) - nn.Tanh()
  netG = nn.gModule({data},{d4})
  return netG
end


function defineD_imageGAN(input_nc, ndf, use_sigmoid)
    local netD = nn.Sequential()

    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc, ndf, 4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 128 x 128
    netD:add(nn.SpatialConvolution(ndf, ndf * 2, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 64 x 64
    netD:add(nn.SpatialConvolution(ndf * 2, ndf*4, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*4) x 32 x 32
    netD:add(nn.SpatialConvolution(ndf * 4, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 16 x 16
    netD:add(nn.SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 8 x 8
    netD:add(nn.SpatialConvolution(ndf * 8, ndf * 8, 4, 4, 2, 2, 1, 1))
    netD:add(nn.SpatialBatchNormalization(ndf * 8)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*8) x 4 x 4
    netD:add(nn.SpatialConvolution(ndf * 8, 1, 4, 4, 2, 2, 1, 1))
    -- state size: 1 x 1 x 1
    if use_sigmoid then
      netD:add(nn.Sigmoid())
    end

	return netD
end


function defineD_basic(input_nc, ndf, use_sigmoid)
    n_layers = 3
    return defineD_n_layers(input_nc, ndf, n_layers, use_sigmoid)
end

-- rf=1
function defineD_pixelGAN(input_nc, ndf, use_sigmoid)

    local netD = nn.Sequential()

    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc, ndf, 1, 1, 1, 1, 0, 0))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 256 x 256
    netD:add(nn.SpatialConvolution(ndf, ndf * 2, 1, 1, 1, 1, 0, 0))
    netD:add(normalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 256 x 256
    netD:add(nn.SpatialConvolution(ndf * 2, 1, 1, 1, 1, 1, 0, 0))
    -- state size: 1 x 256 x 256
    if use_sigmoid then
      netD:add(nn.Sigmoid())
    -- state size: 1 x 30 x 30
    end

    return netD
end

-- if n=0, then use pixelGAN (rf=1)
-- else rf is 16 if n=1
--            34 if n=2
--            70 if n=3
--            142 if n=4
--            286 if n=5
--            574 if n=6
function defineD_n_layers(input_nc, ndf, n_layers, use_sigmoid, kw, dropout_ratio)

  if dropout_ratio == nil then
    dropout_ratio = 0.0
  end

  if kw == nil then
	kw = 4
  end
  padw = math.ceil((kw-1)/2)

    if n_layers==0 then
        return defineD_pixelGAN(input_nc, ndf, use_sigmoid)
    else

        local netD = nn.Sequential()

        -- input is (nc) x 256 x 256
        -- print('input_nc', input_nc)
        netD:add(nn.SpatialConvolution(input_nc, ndf, kw, kw, 2, 2, padw, padw))
        netD:add(nn.LeakyReLU(0.2, true))

        local nf_mult = 1
        local nf_mult_prev = 1
        for n = 1, n_layers-1 do
            nf_mult_prev = nf_mult
            nf_mult = math.min(2^n,8)
            netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, kw, kw, 2, 2, padw,padw))
            netD:add(normalization(ndf * nf_mult)):add(nn.Dropout(dropout_ratio))
            netD:add(nn.LeakyReLU(0.2, true))
        end

        -- state size: (ndf*M) x N x N
        nf_mult_prev = nf_mult
        nf_mult = math.min(2^n_layers,8)
        netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, kw, kw, 1, 1, padw, padw))
        netD:add(normalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
        -- state size: (ndf*M*2) x (N-1) x (N-1)
        netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, kw, kw, 1, 1, padw,padw))
        -- state size: 1 x (N-2) x (N-2)
        if use_sigmoid then
          netD:add(nn.Sigmoid())
        end
        -- state size: 1 x (N-2) x (N-2)
        return netD
    end
end


------ Shuochen's networks --------

function defineG_my_resnet_9blocks(input_nc, output_nc, ngf, tv_strength, tv_weight_sch, tv_weight_scale)
  padding_type = 'reflect'
  local ks = 3
  local netG = nil
  local f = 7
  local p = (f - 1) / 2
  local data = -nn.Identity()
  local data_, amp_
  if tv_weight_sch==0 then
    data_ = data --just corr/phase
  else
    data_ = data - nn.Narrow(2,1,input_nc-1)
    amp_ = data - nn.Narrow(2,input_nc,1) --last channel is the amp for tv weighting
    input_nc = input_nc-1
  end
  local e1 = data_ - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(input_nc, ngf, f, f, 1, 1) - normalization(ngf) - nn.ReLU(true)
  local e2 = e1 - nn.SpatialConvolution(ngf, ngf*2, ks, ks, 2, 2, 1, 1) - normalization(ngf*2) - nn.ReLU(true)
  local e3 = e2 - nn.SpatialConvolution(ngf*2, ngf*4, ks, ks, 2, 2, 1, 1) - normalization(ngf*4) - nn.ReLU(true)
  local d1 = e3 - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
  - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
 - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
  local d2 = d1 - nn.SpatialFullConvolution(ngf*4, ngf*2, 4, 4, 2, 2, 1, 1) - normalization(ngf*2) - nn.ReLU(true)
  local d3 = d2 - nn.SpatialFullConvolution(ngf*2, ngf, 4, 4, 2, 2, 1, 1) - normalization(ngf) - nn.ReLU(true)
  local d4 = d3 - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(ngf, output_nc, f, f, 1, 1) - nn.Tanh()
  local out
  if tv_weight_sch==0 then
    out = d4 - nn.TotalVariation(tv_strength, tv_weight_sch, tv_weight_scale)
  else
    out = {d4,amp_} - nn.JoinTable(2) - nn.TotalVariation(tv_strength, tv_weight_sch, tv_weight_scale)
  end
  netG = nn.gModule({data},{out})
  return netG
end


function defineG_my_resnet(input_nc, output_nc, ngf, tv_strength, tv_weight_sch, tv_weight_scale) -- 4, 1, 16
  local netG = nil
  local data = - nn.Identity()
  local data_, amp_
  if tv_weight_sch==0 then
    data_ = data --just corr/phase
  else
    data_ = data - nn.Narrow(2,1,input_nc-1)
    amp_ = data - nn.Narrow(2,input_nc,1) --last channel is the amp for tv weighting
    input_nc = input_nc-1
  end
  local F0 = data_ - nn.SpatialReplicationPadding(2,2,2,2) - nn.SpatialConvolution(input_nc,ngf,5,5,1,1,0,0) - normalization(ngf) - nn.ReLU(true)
  local F01 = F0 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf,ngf,3,3,1,1,0,0) - normalization(ngf) - nn.ReLU(true)
  local F02 = F01  - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf,ngf,3,3,1,1,0,0) - normalization(ngf) - nn.ReLU(true)
  local D1 = F02 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf,ngf*2,3,3,2,2,0,0) - normalization(ngf*2) - nn.ReLU(true)
  local F1 = D1 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf*2,ngf*2,3,3,1,1,0,0) - normalization(ngf*2) - nn.ReLU(true)
  local F2 = F1 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf*2,ngf*2,3,3,1,1,0,0) - normalization(ngf*2) - nn.ReLU(true)
  local D2 = F2 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf*2,ngf*4,3,3,2,2,0,0) - normalization(ngf*4) - nn.ReLU(true)
  local F3 = D2 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf*4,ngf*4,3,3,1,1,0,0) - normalization(ngf*4) - nn.ReLU(true)
  local F4 = F3 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf*4,ngf*4,3,3,1,1,0,0) - normalization(ngf*4) - nn.ReLU(true)
  local F5 = F4 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf*4,ngf*4,3,3,1,1,0,0) - normalization(ngf*4) - nn.ReLU(true)
  local D3 = F5 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf*4,ngf*8,3,3,2,2,0,0) - normalization(ngf*8) - nn.ReLU(true)
  local F8 = D3 - build_res_block(ngf*8, 'replicate') - build_res_block(ngf*8, 'replicate') - build_res_block(ngf*8, 'replicate') - build_res_block(ngf*8, 'replicate')
  local U1 = F8 - nn.SpatialFullConvolution(ngf*8,ngf*4,4,4,2,2,1,1) - normalization(ngf*4)
  local S1 = {F5,U1} - nn.CAddTable() - nn.ReLU(true)
  local F9 = S1 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf*4,ngf*4,3,3,1,1,0,0) - normalization(ngf*4) - nn.ReLU(true)
  local F10 = F9 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf*4,ngf*4,3,3,1,1,0,0) - normalization(ngf*4) - nn.ReLU(true)
  local F11 = F10 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf*4,ngf*4,3,3,1,1,0,0) - normalization(ngf*4) - nn.ReLU(true)
  local U2 = F11 - nn.SpatialFullConvolution(ngf*4,ngf*2,4,4,2,2,1,1) - normalization(ngf*2)
  local S2 = {F2,U2} - nn.CAddTable() - nn.ReLU(true)
  local F12 = S2 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf*2,ngf*2,3,3,1,1,0,0) - normalization(ngf*2) - nn.ReLU(true)
  local F13 = F12 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf*2,ngf*2,3,3,1,1,0,0) - normalization(ngf*2) - nn.ReLU(true)
  local U3 = F13 - nn.SpatialFullConvolution(ngf*2,ngf,4,4,2,2,1,1) - normalization(ngf)
  local S3 = {F02,U3} - nn.CAddTable() - nn.ReLU(true)
  local F14 = S3 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf,ngf,3,3,1,1,0,0) - normalization(ngf) - nn.ReLU(true)
  local F15 = F14 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf,ngf,3,3,1,1,0,0) - normalization(ngf) - nn.ReLU(true)
  local F16 = F15 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf,output_nc,3,3,1,1,0,0) - nn.Tanh()
  local out
  if tv_weight_sch==0 then
    out = F16 - nn.TotalVariation(tv_strength, tv_weight_sch, tv_weight_scale)
  else
    out = {F16,amp_} - nn.JoinTable(2) - nn.TotalVariation(tv_strength, tv_weight_sch, tv_weight_scale)
  end
  netG = nn.gModule({data},{out})
  return netG
end


function defineG_my_resnet_lite(input_nc, output_nc, ngf, tv_strength, tv_weight_sch, tv_weight_scale) -- 4, 1, 16
  local netG = nil
  local data = - nn.Identity()
  local data_, amp_
  if tv_weight_sch==0 then
    data_ = data --just corr/phase
  else
    data_ = data - nn.Narrow(2,1,input_nc-1)
    amp_ = data - nn.Narrow(2,input_nc,1) --last channel is the amp for tv weighting
    input_nc = input_nc-1
  end
  local F1 = data_ - nn.SpatialReplicationPadding(3,3,3,3) - nn.SpatialConvolution(input_nc,ngf,7,7,1,1,0,0) - normalization(ngf) - nn.ReLU(true)
  local F2 = F1 - nn.SpatialConvolution(ngf,ngf,3,3,1,1,1,1) - normalization(ngf) - nn.ReLU(true)
  local D1 = F2 - nn.SpatialConvolution(ngf,ngf*2,3,3,2,2,1,1) - normalization(ngf*2) - nn.ReLU(true)
  local F3 = D1 - nn.SpatialConvolution(ngf*2,ngf*2,3,3,1,1,1,1) - normalization(ngf*2) - nn.ReLU(true)
  local D2 = F3 - nn.SpatialConvolution(ngf*2,ngf*4,3,3,2,2,1,1) - normalization(ngf*4) - nn.ReLU(true)
  local F4 = D2 - build_res_block(ngf*4, 'replicate') - build_res_block(ngf*4, 'replicate') - build_res_block(ngf*4, 'replicate')
                - build_res_block(ngf*4, 'replicate') - build_res_block(ngf*4, 'replicate') - build_res_block(ngf*4, 'replicate') 
                - build_res_block(ngf*4, 'replicate') - build_res_block(ngf*4, 'replicate') - build_res_block(ngf*4, 'replicate')
  local U1 = F4 - nn.SpatialFullConvolution(ngf*4,ngf*2,4,4,2,2,1,1) - normalization(ngf*2)
  local S1 = {F3,U1} - nn.CAddTable() - nn.ReLU(true)
  local F5 = S1 - nn.SpatialConvolution(ngf*2,ngf*2,3,3,1,1,1,1) - normalization(ngf*2) - nn.ReLU(true)
  local U2 = F5 - nn.SpatialFullConvolution(ngf*2,ngf,4,4,2,2,1,1) - normalization(ngf)
  local S2 = {F2,U2} - nn.CAddTable() - nn.ReLU(true)
  local F6 = S2 - nn.SpatialConvolution(ngf,ngf,3,3,1,1,1,1) - normalization(ngf) - nn.ReLU(true)
  local F7 = F6 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf,output_nc,3,3,1,1,0,0) - nn.Tanh()
  local out
  if tv_weight_sch==0 then
    out = F7 - nn.TotalVariation(tv_strength, tv_weight_sch, tv_weight_scale)
  elseif tv_weight_sch==1 then
    out = {F7,amp_} - nn.JoinTable(2) - nn.TotalVariation(tv_strength, tv_weight_sch, tv_weight_scale)
  elseif tv_weight_sch==2 then -- works terribly, network cannot seperate contribution from corr and amp... changed from sigmoid to exp (110217)
    local amp_refined = amp_ - nn.SpatialConvolution(1,  ngf,3,3,1,1,1,1) - nn.InstanceNormalization(ngf) - nn.ReLU(true)
                             - nn.SpatialConvolution(ngf,ngf,3,3,1,1,1,1) - nn.InstanceNormalization(ngf) - nn.ReLU(true)
                             - nn.SpatialConvolution(ngf,output_nc,3,3,1,1,1,1)
                             - nn.Exp()
    out = {F7,amp_refined} - nn.CMulTable() - nn.TotalVariation(tv_strength, 0)
  end
  netG = nn.gModule({data},{out})
  return netG
end


function defineG_my_resnet_lite_noskip(input_nc, output_nc, ngf, tv_strength, tv_weight_sch, tv_weight_scale) -- 4, 1, 16
  local netG = nil
  local data = - nn.Identity()
  local data_, amp_
  if tv_weight_sch==0 then
    data_ = data --just corr/phase
  else
    data_ = data - nn.Narrow(2,1,input_nc-1)
    amp_ = data - nn.Narrow(2,input_nc,1) --last channel is the amp for tv weighting
    input_nc = input_nc-1
  end
  local F1 = data_ - nn.SpatialReplicationPadding(3,3,3,3) - nn.SpatialConvolution(input_nc,ngf,7,7,1,1,0,0) - normalization(ngf) - nn.ReLU(true)
  local F2 = F1 - nn.SpatialConvolution(ngf,ngf,3,3,1,1,1,1) - normalization(ngf) - nn.ReLU(true)
  local D1 = F2 - nn.SpatialConvolution(ngf,ngf*2,3,3,2,2,1,1) - normalization(ngf*2) - nn.ReLU(true)
  local F3 = D1 - nn.SpatialConvolution(ngf*2,ngf*2,3,3,1,1,1,1) - normalization(ngf*2) - nn.ReLU(true)
  local D2 = F3 - nn.SpatialConvolution(ngf*2,ngf*4,3,3,2,2,1,1) - normalization(ngf*4) - nn.ReLU(true)
  local F4 = D2 - build_res_block(ngf*4, 'replicate') - build_res_block(ngf*4, 'replicate') - build_res_block(ngf*4, 'replicate')
                - build_res_block(ngf*4, 'replicate') - build_res_block(ngf*4, 'replicate') - build_res_block(ngf*4, 'replicate') 
                - build_res_block(ngf*4, 'replicate') - build_res_block(ngf*4, 'replicate') - build_res_block(ngf*4, 'replicate')
  local U1 = F4 - nn.SpatialFullConvolution(ngf*4,ngf*2,4,4,2,2,1,1) - normalization(ngf*2) - nn.ReLU(true)
  local F5 = U1 - nn.SpatialConvolution(ngf*2,ngf*2,3,3,1,1,1,1) - normalization(ngf*2) - nn.ReLU(true)
  local U2 = F5 - nn.SpatialFullConvolution(ngf*2,ngf,4,4,2,2,1,1) - normalization(ngf) - nn.ReLU(true)
  local F6 = U2 - nn.SpatialConvolution(ngf,ngf,3,3,1,1,1,1) - normalization(ngf) - nn.ReLU(true)
  local F7 = F6 - nn.SpatialReplicationPadding(1, 1, 1, 1) - nn.SpatialConvolution(ngf,output_nc,3,3,1,1,0,0) - nn.Tanh()
  local out
  if tv_weight_sch==0 then
    out = F7 - nn.TotalVariation(tv_strength, tv_weight_sch, tv_weight_scale)
  else
    out = {F7,amp_} - nn.JoinTable(2) - nn.TotalVariation(tv_strength, tv_weight_sch, tv_weight_scale)
  end
  netG = nn.gModule({data},{out})
  return netG
end


function defineD_my_imageGAN_128(input_nc, ndf, use_sigmoid, tv_weight_sch)
  local netD = nil
  local data = -nn.Identity()
  local data_
  if tv_weight_sch==0 then
    data_ = data --just corr/phase
  else
    data_ = data - nn.Narrow(2,1,input_nc-1)
    input_nc = input_nc-1
  end
  -- input is (nc) x 128 x 128
  local d1 = data_ - nn.SpatialConvolution(input_nc, ndf, 5, 5, 2, 2, 2, 2) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf) x 64 x 64
  local d2 = d1 - nn.SpatialConvolution(ndf, ndf * 2, 5, 5, 2, 2, 2, 2) - normalization(ndf * 2) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf*2) x 32 x 32
  local d3 = d2 - nn.SpatialConvolution(ndf * 2, ndf*4, 5, 5, 2, 2, 2, 2) - normalization(ndf * 4) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf*4) x 16 x 16
  local d4 = d3 - nn.SpatialConvolution(ndf * 4, ndf * 8, 5, 5, 2, 2, 2, 2) - normalization(ndf * 8) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf*8) x 8 x 8
  local d5 = d4 - nn.SpatialConvolution(ndf * 8, ndf * 8, 5, 5, 2, 2, 2, 2) - normalization(ndf * 8) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf*8) x 4 x 4
  local d6 = d5 - nn.SpatialConvolution(ndf * 8, ndf * 8, 5, 5, 2, 2, 2, 2) - normalization(ndf * 8) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf*8) x 2 x 2
  local d7 = d6 - nn.SpatialConvolution(ndf * 8, ndf * 16, 5, 5, 2, 2, 2, 2) - normalization(ndf * 16) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf*16) x 1 x 1
  local d8 = d7 - nn.SpatialConvolution(ndf * 16, 1, 1, 1)
  -- state size: 1 x 1 x 1
  if use_sigmoid then
    local d9 = d8 - nn.Sigmoid()
    netD = nn.gModule({data},{d9})
  else
    netD = nn.gModule({data},{d8})
  end
  return netD
end


function defineD_my_imageGAN_192(input_nc, ndf, use_sigmoid, tv_weight_sch)
  local netD = nil
  local data = -nn.Identity()
  local data_ = nil
  if tv_weight_sch==0 then
    data_ = data --just corr/phase
  else
    data_ = data - nn.Narrow(2,1,input_nc-1)
    input_nc = input_nc-1
  end
  -- input is (nc) x 192 x 192
  local d1 = data_ - nn.SpatialConvolution(input_nc, ndf, 5, 5, 2, 2, 2, 2) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf) x 96 x 96
  local d2 = d1 - nn.SpatialConvolution(ndf, ndf * 2, 5, 5, 2, 2, 2, 2) - normalization(ndf * 2) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf*2) x 48 x 48
  local d3 = d2 - nn.SpatialConvolution(ndf * 2, ndf*4, 5, 5, 2, 2, 2, 2) - normalization(ndf * 4) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf*4) x 24 x 24
  local d4 = d3 - nn.SpatialConvolution(ndf * 4, ndf * 8, 5, 5, 2, 2, 2, 2) - normalization(ndf * 8) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf*8) x 12 x 12
  local d5 = d4 - nn.SpatialConvolution(ndf * 8, ndf * 8, 5, 5, 2, 2, 2, 2) - normalization(ndf * 8) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf*8) x 6 x 6
  local d6 = d5 - nn.SpatialConvolution(ndf * 8, ndf * 8, 5, 5, 2, 2, 2, 2) - normalization(ndf * 8) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf*8) x 3 x 3
  local d7 = d6 - nn.SpatialConvolution(ndf * 8, ndf * 16, 3, 3, 1, 1, 0, 0) - normalization(ndf * 16) - nn.LeakyReLU(0.2, true)
  -- state size: (ndf*16) x 1 x 1
  local d8 = d7 - nn.SpatialConvolution(ndf * 16, 1, 1, 1)
  -- state size: 1 x 1 x 1
  if use_sigmoid then
    local d9 = d8 - nn.Sigmoid()
    netD = nn.gModule({data},{d9})
  else
    netD = nn.gModule({data},{d8})
  end
  return netD
end

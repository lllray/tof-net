FILE=$1

echo "Note: available models are my_resnet_lite_my_imageGAN_128"

echo "Specified [$FILE]"

mkdir -p ./checkpoints/${FILE}_pretrained
URL=http://www.cs.ubc.ca/labs/imager/tr/2018/DeepEnd2EndToFImaging/models/$FILE.t7
MODEL_FILE=./checkpoints/${FILE}_pretrained/latest_net_G.t7
wget -N $URL -O $MODEL_FILE

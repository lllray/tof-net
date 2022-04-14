FILE=$1

if [[ $FILE != "mpi_correction_corrnormamp2depth_albedoaug_zpass" && $FILE != "mpi_correction_corrnormamp2depth_albedoaug_zpass_test" ]]; then
    echo "Available datasets are: mpi_correction_corrnormamp2depth_albedoaug_zpass, mpi_correction_corrnormamp2depth_albedoaug_zpass_test"
    exit 1
fi

URL=http://www.cs.ubc.ca/labs/imager/tr/2018/DeepEnd2EndToFImaging/datasets/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE

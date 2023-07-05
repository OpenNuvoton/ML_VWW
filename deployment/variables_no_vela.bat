set IMAGE_SRC_DIR=..\samples
set IMAGE_SRC_WIDTH=320
set IMAGE_SRC_HEIGHT=320

::set LABEL_SRC_FILE=..\labels\labels_mobilenet_v2_1.0_224.txt
::set GEN_LABEL_FILE_NAME=Labels

set MODEL_SRC_DIR=..\workspace\vww_v2_96_035\tflite_model
set MODEL_SRC_FILE=vww_mobilenetv2_0.35_96_96_int8quant.tflite
::The vela OPTIMISE_FILE should be SRC_FILE_NAME + _vela

set TEMPLATES_DIR=Tool\tflite2cpp\templates

set GEN_SRC_DIR=..\workspace\vww_v2_96_035\tflite_model\c
set GEN_INC_DIR=generated\include











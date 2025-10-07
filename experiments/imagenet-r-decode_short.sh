# bash experiments/imagenet-r.sh
# experiment settings
DATASET=ImageNet_R
N_CLASS=200

# save directory
OUTDIR=outputs/${DATASET}/5-task

# hard coded inputs
GPUID='0'
CONFIG=configs/imnet-r_decode_short.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

# CODA-P
#
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight - with updated code, now can be 0!
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type decode --learner_name Decode \
    --hyper_param 128 128 \
    --log_dir ${OUTDIR}/decode
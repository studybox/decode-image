# bash experiments/cifar-100.sh
# experiment settings
DATASET=5-datasets
N_CLASS=50

# save directory
OUTDIR=outputs/${DATASET}/10-task

# hard coded inputs
GPUID='0'
CONFIG=configs/5dataset_decode.yaml
CONFIG_FT=configs/5dataset_ft.yaml
REPEAT=1
OVERWRITE=0

###############################################################

# process inputs
mkdir -p $OUTDIR

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type decode --learner_name Decode \
    --hyper_param 128 128 \
    --log_dir ${OUTDIR}/decode_0.0005

# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 30 20 -1 \
#     --log_dir ${OUTDIR}/l2p++

# # FT
# python -u run.py --config $CONFIG_FT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type default --learner_name FinetunePlus \
#     --log_dir ${OUTDIR}/ft++

# # FT++
# python -u run.py --config $CONFIG_FT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type default --learner_name NormalNN \
#     --log_dir ${OUTDIR}/ft

# # Offline
# python -u run.py --config $CONFIG_FT --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type default --learner_name NormalNN --upper_bound_flag \
#     --log_dir ${OUTDIR}/offline


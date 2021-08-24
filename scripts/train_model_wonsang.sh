cd ..
#CUDA_VISIBLE_DEVICES=5 python transformer_discrete_main.py \
CUDA_VISIBLE_DEVICES=0 python two_stream_transformer_discrete_main.py \
--cuda \
--train-discrete-json-file="./data/new_split/pose_train_new.json" \
--val-discrete-json-file="./data/new_split/pose_test_new.json" \
--discrete-pose3d-folder="./data/keypoints3d_npy_files" \
--mfcc-beat-json-folder="./data/final_audio_json" \
--log-dir="./tensorboard/block_20s_discrete300_transformer/lr1e-4_bs32_feats10_dout10" \
--batch-size=32 \
--lr=1e-4 \
--n-dec-layers=4 \
--n-head=4 \
--d-model=128 \
--d-k=128 \
--d-v=128 \
--epochs=2000 \
--lr-steps 800 1600 \
--max-timesteps=2878 \
--num-cls=300 \
--feats-dim=10 \
--d-out=10 \
--multi-stream \
--add-mfcc \
--add-beat \
#--pose3d-folder="./data/keypoints3d_npy_files" \
#--train-json-file="./data/new_split/pose_train_new.json" \
#--val-json-file="./data/new_split/pose_test_new.json" \
2>&1 |tee logs/train_discrete300_transformer_decoder_block_20s_lr1e-4_bs32_feats10_dout10.log
#--checkpoint-folder="checkpoint/block_20s_discrete300_transformer/lr1e-4_bs32_feats10_dout10" \
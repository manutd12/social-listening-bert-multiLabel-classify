export BERT_BASE_DIR=model/english_uncased_L-12_H-768_A-12
export XNLI_DIR=data

CUDA_VISIBLE_DEVICES=5 python bert/run_classifier_train_multiLabel.py \
  --task_name=social_listening \
  --do_train=False \
  --do_calThresholds=False \
  --do_eval=True \
  --data_dir=$XNLI_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=256 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=10.0 \
  --output_dir=bert/my_output_16batch_allLabel_sampleThresholdData_label_smooth/

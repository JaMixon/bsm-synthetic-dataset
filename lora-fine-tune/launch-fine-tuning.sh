export TRAIN_DATA= # enter path to training data folder
export OUTPUT_DIR= # enter path to output folder 
export MODEL_NAME="stabilityai/stable-diffusion-2" 

accelerate launch diffusers/examples/text_to_image/train_text_to_image_lora.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--dataset_name=$TRAIN_DATA --dataloader_num_workers=8 \
--resolution=768 --allow_tf32 \
--train_batch_size=1 --gradient_accumulation_steps=4 \
--max_train_steps=15000 --learning_rate=1e-04 \
--max_grad_norm=1 --lr_scheduler="cosine" \
--lr_warmup_steps=0 --output_dir=${OUTPUT_DIR} \
--report_to=wandb --checkpointing_steps=500 \
--validation_prompt="Bone surface modification with meat." \
--seed=939 
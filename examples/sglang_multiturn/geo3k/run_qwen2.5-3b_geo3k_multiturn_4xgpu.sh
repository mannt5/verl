# run on 4xH100
# make sure your current working directory is the root of the project

set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='geo3k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=256 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    '+actor_rollout_ref.model.override_config.chat_template="{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{%- if tools %}{{- \"<|im_start|>system\\n\" }}{%- if messages[0][\"role\"] == \"system\" %}{{- messages[0][\"content\"] }}{%- else %}{{- \"You are a helpful assistant.\" }}{%- endif %}{{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}{%- for tool in tools %}{{- \"\\n\" }}{{- tool | tojson }}{%- endfor %}{{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}{% for message in messages %}{% if message[\"role\"] != \"system\" or loop.first == false %}{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}<|im_start|>{{ message[\"role\"] }}\n{% if message[\"content\"] is string %}{{ message[\"content\"] }}<|im_end|>\n{% else %}{% for content in message[\"content\"] %}{% if content[\"type\"] == \"image\" or \"image\" in content or \"image_url\" in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content[\"type\"] == \"video\" or \"video\" in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif \"text\" in content %}{{ content[\"text\"] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{%- elif message.role == \"assistant\" %}{{- \"<|im_start|>\" + message.role }}{%- if message.content %}{{- \"\\n\" + message.content }}{%- endif %}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- \"\\n<tool_call>\\n{\"name\": \"\" }}{{- tool_call.name }}{{- \"\", \"arguments\": \" }}{{- tool_call.arguments | tojson }}{{- \"}\\n</tool_call>\" }}{%- endfor %}{{- \"<|im_end|>\\n\" }}{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- \"<|im_start|>user\" }}{%- endif %}{{- \"\\n<tool_response>\\n\" }}{% if message[\"content\"] is string %}{{ message.content }}{% else %}{% for content in message[\"content\"] %}{% if content[\"type\"] == \"image\" or \"image\" in content or \"image_url\" in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content[\"type\"] == \"video\" or \"video\" in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content[\"type\"] == \"text\" or \"text\" in content %}{{ content[\"text\"] }}{% endif %}{% endfor %}{% endif %}{{- \"\\n</tool_response>\" }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- \"<|im_end|>\\n\" }}{%- endif %}{%- endif %}{% endif %}{% endfor %}{%- else %}{% for message in messages %}{% if loop.first and message[\"role\"] != \"system\" %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}{%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}<|im_start|>{{ message[\"role\"] }}\n{% if message[\"content\"] is string %}{{ message[\"content\"] }}<|im_end|>\n{% else %}{% for content in message[\"content\"] %}{% if content[\"type\"] == \"image\" or \"image\" in content or \"image_url\" in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content[\"type\"] == \"video\" or \"video\" in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif \"text\" in content %}{{ content[\"text\"] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{%- elif message.role == \"assistant\" %}{{- \"<|im_start|>\" + message.role }}{%- if message.content %}{{- \"\\n\" + message.content }}{%- endif %}{%- for tool_call in message.tool_calls %}{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}{{- \"\\n<tool_call>\\n{\"name\": \"\" }}{{- tool_call.name }}{{- \"\", \"arguments\": \" }}{{- tool_call.arguments | tojson }}{{- \"}\\n</tool_call>\" }}{%- endfor %}{{- \"<|im_end|>\\n\" }}{%- elif message.role == \"tool\" %}{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}{{- \"<|im_start|>user\" }}{%- endif %}{{- \"\\n<tool_response>\\n\" }}{% if message[\"content\"] is string %}{{ message.content }}{% else %}{% for content in message[\"content\"] %}{% if content[\"type\"] == \"image\" or \"image\" in content or \"image_url\" in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content[\"type\"] == \"video\" or \"video\" in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif content[\"type\"] == \"text\" or \"text\" in content %}{{ content[\"text\"] }}{% endif %}{% endfor %}{% endif %}{{- \"\\n</tool_response>\" }}{%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}{{- \"<|im_end|>\\n\" }}{%- endif %}{%- endif %}{% endfor %}{%- endif %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"' \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='geo3k_async_rl' \
    trainer.experiment_name='qwen2.5-3b_function_rm-geo3k-async-sgl-multi-w-tool-verify-n16-4cards' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=15 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    critic.ppo_max_token_len_per_gpu=8192 \
    critic.forward_max_token_len_per_gpu=8192 \
    data.train_files=$HOME/data/geo3k/train.parquet \
    data.val_files=$HOME/data/geo3k/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/geo3k_tool_config.yaml" \
    $@
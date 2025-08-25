# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
import logging
import os
from typing import Any
from functools import partial


import psutil
import torch
import torch.distributed
from codetiming import Timer

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
)
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.profiler import DistProfiler, DistProfilerExtension
from verl.workers.engine import EngineRegistry
import verl.workers.engine.config as engine_cfg
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class ActorWorker(Worker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config, role, **kwargs):
        Worker.__init__(self)
        # TODO(haibin.lin):
        # As of now the type of config is DictConfig, if we assign config.profiler with ProfilerConfig,
        # it will actually convert the ProfilerConfig dataclass back to a DictConfig.
        # We can still use ProfilerConfig for testing purpose (tests/utils/test_nvtx_profile.py)
        # as they provides DictConfig-like interface
        # The benefit of creating the dataclass config is to perform validation during __post_init__
        self.profile_option = kwargs.get("profile_option", None)
        profiler_config = omega_conf_to_dataclass(config.get("profiler"))
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, option=self.profile_option)
        )

        import torch.distributed
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        # if strategy is fsdp, offload_config should be None

        self.role = role
        assert self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]

        self.config = config
        assert not self.config.model.use_fused_kernels and not self.config.actor.entropy_checkpointing, \
            "fused kernels and entropy checkpointing are not supported in the new worker implementation yet"

        engine_config = self.create_engine_config(self.config)
        self.engine = EngineRegistry.new(self.config.actor.strategy, engine_config)
    

    def create_engine_config(self, config):
        # ModelConfig Setup
        model_config = engine_cfg.ModelConfig(
            path=config.model.path,
            module_type="causal_lm",
            lora_rank=config.model.lora_rank,
            lora_alpha=config.model.lora_alpha,
            target_modules=config.model.target_modules,
            trust_remote_code=config.model.trust_remote_code,
            use_shm=config.model.use_shm,
            external_lib=config.model.external_lib,
            override_config=config.model.get("override_config", {}),
            custom_chat_template=config.model.get("custom_chat_template", None),
            tokenizer_path=None,
            use_liger=config.model.get("use_liger", False),
            use_fused_kernels=config.model.get("use_fused_kernels", False),
            fused_kernel_options=config.model.get("fused_kernel_options", None),
        )

        # OptimConfig Setup
        optim_config = config.actor.optim
        lr_scheduler_style = optim_config.get("warmup_style", "constant")
        lr_scheduler_args = None
        if lr_scheduler_style == "consine":
            lr_scheduler_args = {
                "min_lr_ratio": optim_config.get("min_lr_ratio", 0.0),
                "num_cycles": optim_config.get("num_cycles", 0.5),
            }

        optim_config = engine_cfg.OptimConfig(
            grad_clip=config.actor.grad_clip,  
            betas=optim_config.get("betas", (0.9, 0.999)),    
            weight_decay=optim_config.get("weight_decay", 1e-2),  
            lr=optim_config.lr,  
            lr_warmup_steps=optim_config.get("lr_warmup_steps", -1),    
            lr_warmup_steps_ratio=optim_config.lr_warmup_steps_ratio,
            lr_scheduler_style=lr_scheduler_style,
            lr_scheduler_args=lr_scheduler_args,
        )

        # SystemConfig Setup
        fsdp_config = config.actor.fsdp_config

        system_config = engine_cfg.SystemConfig(
            fsdp_size=fsdp_config.fsdp_size,
            model_dtype=fsdp_config.get("model_dtype", "fp32"),
            param_offload=fsdp_config.param_offload,
            optimizer_offload=fsdp_config.optimizer_offload,
            wrap_policy=fsdp_config.wrap_policy,
            reshard_after_forward=fsdp_config.reshard_after_forward,
            offload_policy=fsdp_config.offload_policy,
            forward_prefetch=fsdp_config.forward_prefetch,
            ulysses_sequence_parallel_size=config.actor.ulysses_sequence_parallel_size,
            enable_gradient_checkpointing=config.model.enable_gradient_checkpointing,
            enable_activation_offload=config.model.enable_activation_offload,
            use_remove_padding=config.model.use_remove_padding,
            mixed_precision=fsdp_config.get("mixed_precision", None),
            use_orig_params=fsdp_config.get("use_orig_params", False),
        )

        # CheckpointConfig Setup
        ckpt_config = engine_cfg.CheckpointConfig(
            save_contents=config.actor.checkpoint.save_contents,
            load_contents=config.actor.checkpoint.load_contents,
            async_save=config.actor.checkpoint.async_save,
        )

        train_mini_batch_size = config.actor.ppo_mini_batch_size * config.rollout.n

        engine_config = engine_cfg.EngineConfig(
            strategy=config.actor.strategy,
            model=model_config,
            optim=optim_config,
            system=system_config,
            checkpoint=ckpt_config,
            total_training_steps=config.actor.optim.get("total_training_steps", -1),
            use_dynamic_bsz=config.actor.use_dynamic_bsz,
            train_mini_batch_size=train_mini_batch_size,
            train_micro_batch_size_per_gpu=config.actor.ppo_micro_batch_size_per_gpu,
            train_max_token_len_per_gpu=config.actor.ppo_max_token_len_per_gpu,
            infer_micro_batch_size_per_gpu=config.rollout.log_prob_micro_batch_size_per_gpu,
            infer_max_token_len_per_gpu=config.rollout.log_prob_max_token_len_per_gpu,
        )

        return engine_config


    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self.engine.initialize()

    # TODO: temporary
    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        self.engine.send_params()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        return self.engine.get_params_meta_info()

    def _post_fn_log_prob(self, micro_batch, preds):
        # do not support not calculate entropy
        response_length = micro_batch["responses"].size(-1)
        temperature = self.config.rollout.temperature

        logits = preds                                                      # (bsz, seqlen, vocab_size)
        logits.div_(temperature)
        logits = logits[:, -response_length - 1 : -1, :]                    # (bsz, response_length, vocab_size)
        log_probs = logprobs_from_logits(logits,
                                         micro_batch["responses"],
                                         inplace_backward=False)            # (bsz, response_length)
        entropy = verl_F.entropy_from_logits(logits)                        # (bsz, response_length)

        outputs = {
            "log_probs": log_probs.clone().detach(),
            "entropy": entropy.clone().detach(),
        }

        return (log_probs, entropy), outputs

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        # TODO: adapter_ctx support for lora model

        # Support all hardwares
        data = data.to(get_device_id())

        with self.engine.eval_mode():
            data = self.engine.shard_data(data=data)
            output = self.engine.forward_step(data, post_fn=self._post_fn_log_prob)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output["log_probs"], "entropys": output["entropy"]},
                meta_info={"temperature": self.config.rollout.temperature},
            )

            output = self.engine.unshard_data(data=output)
        output = output.to("cpu")

        return output

    
    def loss_fn(
        self, batch: DataProto, vpreds: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        response_mask = batch["response_mask"]
        old_log_prob = batch["old_log_probs"]
        advantages = batch["advantages"]
        micro_batch_metrics = {}

        entropy_coeff = self.config.actor.entropy_coeff
        loss_agg_mode = self.config.actor.loss_agg_mode

        (log_prob, entropy), _ = self._post_fn_log_prob(batch, vpreds)

        loss_mode = self.config.actor.policy_loss.get("loss_mode", "vanilla")
        # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=self.config.actor,
        )

        if entropy_coeff != 0:
            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
            # compute policy loss
            policy_loss = pg_loss - entropy_loss * entropy_coeff
        else:
            policy_loss = pg_loss

        if self.config.actor.use_kl_loss:
            ref_log_prob = batch["ref_log_prob"]
            # compute kl loss
            kld = kl_penalty(
                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.actor.kl_loss_type
            )
            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

            policy_loss = policy_loss + kl_loss * self.config.actor.kl_loss_coef
            micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
            micro_batch_metrics["actor/kl_coef"] = self.config.actor.kl_loss_coef

        micro_batch_metrics.update({
            "actor/pg_loss": pg_loss.detach().item(),
            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
            "actor/ppo_kl": ppo_kl.detach().item(),
            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),    
        })
        return policy_loss, micro_batch_metrics


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        metrics = {}
        # Support all hardwares
        data = data.to(get_device_id())
        # perform forward computation
        with self.engine.train_mode():
            data = self.engine.shard_data(data=data)

            with Timer(name="update_policy", logger=None) as timer:
                select_keys = [
                    "responses",
                    "response_mask",
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "old_log_probs",
                    "advantages",
                ]

                if self.config.actor.use_kl_loss:
                    select_keys.append("ref_log_prob")

                has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
                non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
                data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

                # calculate mini batch size
                mini_bsz = self.config.actor.ppo_mini_batch_size * self.config.rollout.n
                mini_bsz = mini_bsz // self.engine.get_data_parallel_size()
                # Split to make minibatch iterator for updating the actor
                # See PPO paper for details. https://arxiv.org/abs/1707.06347
                mini_batches = data.split(mini_bsz)

                metrics = {}
                for epoch in range(self.config.actor.ppo_epochs):
                    for batch_idx, mini_batch in enumerate(mini_batches):
                        mini_batch_metrics = self.engine.train_step(mini_batch, self.loss_fn)
                        # renaming metrics for actor specific
                        mini_batch_metrics["actor/grad_norm"] = mini_batch_metrics.pop("grad_norm")
                        append_to_dict(metrics, mini_batch_metrics)
            delta_time = timer.last

            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.engine.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.engine.lr_scheduler_step()[0]
            metrics["actor/lr"] = lr

            output = DataProto(batch=None, meta_info={"metrics": metrics})
            output = self.engine.unshard_data(data=output)

        output = output.to("cpu")
        raise ValueError
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        raise NotImplementedError

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        raise NotImplementedError

import math
from tqdm import tqdm
from pathlib import Path
import traceback
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from lmr.checkpointing import Checkpointing
from lmr.ddp import setup_ddp, cleanup_ddp, initialize_model_ddp, unwrap_model, initialize_samplers_ddp
from lmr.utils.logger import Logger
from lmr.utils.parsing import int_to_formatted_string

class Trainer:
    def __init__(self, training_config, model, tokenizer, splits, checkpointing, samplers=None, device=None):
        
        self.training_config = training_config
        self.model = model
        self.tokenizer = tokenizer
        self.splits = splits
        self.checkpointing = checkpointing
        self.samplers = samplers
        self.device = device

        self.autocast_dtype = getattr(torch, self.training_config.precision)
        
        self.use_ddp = False
        self.rank = 0
        self.world_size = 1
    
    # We keep this separate to keep the DDP and non-DDP logic well organized
    def _setup_training(self):
            
        self.train_dataloader = self._get_dataloader("train")
        self.validation_dataloader = self._get_dataloader("validation")
        
        if self.training_config.use_grad_accum and self.training_config.grad_accum_steps == "auto":
                tokens_per_model_step = self.training_config.batch_size * self.model.config.max_seq_len * self.world_size
                self.grad_accum_steps = self.training_config.tokens_per_step // tokens_per_model_step
        elif self.training_config.use_grad_accum:
            self.grad_accum_steps = self.training_config.grad_accum_steps
        else:
            self.grad_accum_steps = 1
            
        self.steps_per_epoch = len(self.train_dataloader) // self.grad_accum_steps # May lose a few batches when using grad_accum, but ensures consistent updates
        
        self.tokens_per_batch = self.training_config.batch_size * self.model.config.max_seq_len
        self.tokens_per_step = self.grad_accum_steps * self.tokens_per_batch * self.world_size
        self.tokens_per_epoch = self.tokens_per_step * self.steps_per_epoch

        self.checkpointing.load_model_states("recent")
        
        self.device = torch.device(f"cuda:{self.rank}")
        
        if self.training_config.compile:
            self.model = torch.compile(self.model, mode=self.training_config.compile_mode)
        
        self.model.to(self.device)
        
        if self.use_ddp:
            self.model = initialize_model_ddp(self.model, self.rank)
            
        self.model.train()

        self._initialize_optimizer()
        self._initialize_scheduler()
        self._initialize_scaler() # Currently Unsupported
        
        self.checkpointing.load_training_states("recent")
        
    def _initialize_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.lr,
            betas=self.training_config.betas,
            weight_decay=self.training_config.weight_decay
        )
        self.checkpointing.optimizer = self.optimizer
    
    def _initialize_scheduler(self):
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=self.steps_per_epoch * self.training_config.max_epochs
        )
        self.checkpointing.scheduler = self.scheduler
    
    def _initialize_scaler(self):
        if self.autocast_dtype != torch.float16:
            self.scaler = None
            return
        self.scaler = GradScaler("cuda")
        self.checkpointing.scaler = self.scaler
        
    def _get_dataloader(self, split_name):
        num_workers = 1 if split_name == "validation" else self.training_config.num_workers - 1
        return DataLoader(
            self.splits[split_name],
            batch_size=self.training_config.batch_size,
            num_workers=num_workers, 
            shuffle=(split_name == "train" and self.samplers is None),
            sampler=None if self.samplers is None else self.samplers[split_name],
            pin_memory=False,
            drop_last=True
        )
        
    def _calculate_training_tokens(self, epoch, step):
        return epoch * self.tokens_per_epoch + step * self.tokens_per_step

    def _step_loss(self, batch):
        batch = batch.to(self.device, non_blocking=True)
        input_tokens = batch[:, :-1]
        target_tokens = batch[:, 1:]
        
        with autocast(device_type="cuda", dtype=self.autocast_dtype):
            logits = self.model(input_tokens)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1),
                ignore_index=self.tokenizer.pad_token_id,
                reduction='mean'
                )
            
        return loss
        
    def _validate(self):
        self.model.eval()
        loss_sum = torch.tensor(0.0, device=self.device)
        count = torch.tensor(0, device=self.device, dtype=torch.long)
        with torch.no_grad():
            for batch in self.validation_dataloader:
                loss = self._step_loss(batch).detach()
                # Ensures consistency when validating across devices (when batch number might be different)
                loss_sum += loss * self.tokens_per_batch
                count += self.tokens_per_batch

        self._reduce(loss_sum)
        self._reduce(count)
        
        self.model.train()
        return (loss_sum / count).item()
        
    def _log_training_msg(self, resume=False):
        
        model = unwrap_model(self.model)
        
        total_params, embed_params, non_embed_params = model.count_parameters()
        
        param_msg = f"{int_to_formatted_string(total_params)} Total, {int_to_formatted_string(embed_params)} Embed, {int_to_formatted_string(non_embed_params)} Non-Embed"
        
        if resume:
            model_msg = f"Resuming model '{model.full_name}' [{param_msg}]"
        else:
            model_msg = f"Training Model '{model.full_name}' [{param_msg}] from scratch"
            
        training_msg = f"Using training={self.training_config.training_name}, device={self.device} [total devices: {self.world_size}], dtype={str(self.autocast_dtype)}, grad_scaling={self.scaler is not None}"

        if self.training_config.compile:
            training_msg += f", and compile_mode={self.training_config.compile_mode}"
        
        grad_msg = f"Using grad_accum={self.grad_accum_steps}, batch_size={self.training_config.batch_size}, grad_accum_tokens={self.tokens_per_step}"
        
        Logger.log(model_msg)        
        Logger.log(training_msg)
        Logger.log(grad_msg)

    def _ddp_barrier(self):
        if self.use_ddp:
            torch.distributed.barrier()

    def _reduce(self, item):
        if self.use_ddp:
            torch.distributed.all_reduce(item, op=torch.distributed.ReduceOp.SUM)
        
    def _train(self):
        
        self._setup_training()

        start_epoch = self.checkpointing.epoch
        start_step = self.checkpointing.step
        tokens_trained = self.checkpointing.tokens_trained

        resume = start_step != 0
        
        self._log_training_msg(resume=resume)

        if resume:
            tokens_trained = self._calculate_training_tokens(start_epoch, start_step)
            tokens_trained_formatted = int_to_formatted_string(tokens_trained)
            Logger.log(f"Resuming from epoch {start_epoch} step {start_step} ({tokens_trained_formatted} tokens)")

        mr_step_loss = self.checkpointing.train_loss
        mr_validation_loss = self.checkpointing.val_loss
        mr_validation_ppl = float("inf") if mr_validation_loss is None or mr_validation_loss > 100 else math.exp(self.checkpointing.val_loss)

        for epoch in range(start_epoch, self.training_config.max_epochs):
            
            if self.train_dataloader.sampler is not None and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)
        
            pbar = tqdm(total=self.steps_per_epoch, desc=f"Epoch {epoch}") if self.rank == 0 else None
            step = 0
            step_loss = 0.0

            for micro_step, batch in enumerate(self.train_dataloader):

                step = micro_step // self.grad_accum_steps
                
                if step >= self.steps_per_epoch:
                    break

                if resume and step < start_step:
                    if pbar is not None: pbar.update(1)
                    continue
                elif resume:
                    resume = False
                    
                is_update_step = ((micro_step + 1) % self.grad_accum_steps == 0)
                    
                sync_ctx = self.model.no_sync() if (self.use_ddp and not is_update_step) else nullcontext()
                
                with sync_ctx:
                    loss = self._step_loss(batch)
                    loss = loss / self.grad_accum_steps
                    step_loss += loss.item()
                    loss.backward()

                # During micro-steps, no updating/saving is done
                if not is_update_step:
                    continue
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

                tokens_trained = self._calculate_training_tokens(epoch, step + 1)
                tokens_trained_formatted = int_to_formatted_string(tokens_trained)
                
                mr_step_loss = step_loss
                step_loss = 0.0
                
                # Validation Checkpointing
                if self.training_config.validation_steps is not None and (step + 1) % self.training_config.validation_steps == 0:
                    self._ddp_barrier()
                    mr_validation_loss = self._validate()
                    mr_validation_ppl = math.exp(mr_validation_loss)
                    if self.rank == 0: self.checkpointing.save_checkpoint(epoch, step + 1, mr_step_loss, mr_validation_loss, tokens_trained)
                    self._ddp_barrier()
                    
                if pbar is not None: 
                    pbar.set_postfix(train_loss=f"{mr_step_loss:.3f}", val_loss=f"{mr_validation_loss:.3f}", val_ppl=f"{mr_validation_ppl:.3f}", tokens=tokens_trained_formatted)
                    pbar.update(1)
                
            # Per-Epoch Checkpointing
            tokens_trained = self._calculate_training_tokens(epoch + 1, 0)
            tokens_trained_formatted = int_to_formatted_string(tokens_trained)

            self._ddp_barrier()
            mr_validation_loss = self._validate()
            mr_validation_ppl = math.exp(mr_validation_loss)
            if self.rank == 0: self.checkpointing.save_checkpoint(epoch + 1, None, mr_step_loss, mr_validation_loss, tokens_trained)
            self._ddp_barrier()
            
            Logger.log(f"Epoch {epoch + 1} | Train Loss: {mr_step_loss:.3f} | Val Loss: {mr_validation_loss:.3f} | Val PPL: {mr_validation_ppl:.3f} | Tokens Trained: {tokens_trained_formatted}")

    def _train_ddp(self):
        self.use_ddp = True
        self.rank, self.world_size = setup_ddp()
        try:
            self.samplers = initialize_samplers_ddp(self.splits, self.rank, self.world_size)
            self._train()
        except Exception as e:
            print(f"[Rank {self.rank}] Exception occurred:")
            traceback.print_exc()
        finally:
            cleanup_ddp()

    def train(self):
        if self.training_config.use_ddp:
            self._train_ddp()
        else:
            self._train()
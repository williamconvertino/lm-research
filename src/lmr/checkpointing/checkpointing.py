import time
import torch
import re
from pathlib import Path
from glob import glob

from lmr.utils.logger import Logger
from lmr.ddp import unwrap_model

class Checkpointing:
    
    def __init__(self, model, checkpoint_dir, optimizer=None, scheduler=None, scaler=None, map_device="cpu"):
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.map_device = map_device
        
        # State fields
        self.epoch = 0
        self.step = 0
        self.train_loss = float("inf")
        self.val_loss = float("inf")
        self.tokens_trained = 0

        # DDP flags
        self.use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.is_main_process = not self.use_ddp or torch.distributed.get_rank() == 0

        self.checkpoint_dir = Path(checkpoint_dir)
        
        if not self.is_main_process:
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.checkpoint_dir / "_checkpoint_log.tsv"
        self._create_log()

        self.best_val_loss = self._get_best_val_loss()

    # ---------- ddp ----------

    def _barrier(self):
        if self.use_ddp:
            torch.distributed.barrier()

    # ---------- logging + util ----------
    
    def _remove_old(self, pattern):
        if not self.is_main_process:
            return
        for f in glob(str(self.checkpoint_dir / pattern)):
            try:
                Path(f).unlink()
            except FileNotFoundError:
                pass

    def _get_best_val_loss(self):
        best_ckpts = glob(str(self.checkpoint_dir / "best_epoch*_val=*.pt"))
        best_val = float("inf")
        for ckpt_path in best_ckpts:
            match = re.search(r"val=([0-9.]+)\.pt", ckpt_path)
            if match:
                try:
                    best_val = min(best_val, float(match.group(1)))
                except ValueError:
                    pass
        return best_val

    def _checkpoint_filename(self, epoch, step=None, val_loss=None, prefix=None):
        
        name = f"epoch_{epoch:03d}"
        
        if prefix is not None:
            name = f"{prefix}_{name}"
        if step is not None:
            name += f"_step_{step:09d}"
        if val_loss is not None:
            name += f"_val={val_loss:.4f}"
        
        name += ".pt"
        
        return name

    def _create_log(self):
        if self.log_path.exists():
            return
        header = "Time\tCheckpoint_Type\tEpoch\tStep\tTrain_Loss\tVal_Loss\tTokens_Trained\n"
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(header)

    def _update_log(self, kind):
        
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        epoch_str = f"{self.epoch:03d}"
        step_str = f"{self.step:09d}"
        
        line = (
            f"{ts}\t{kind}\t{epoch_str}\t{step_str}\t"
            f"{'' if self.train_loss is None else self.train_loss}\t"
            f"{'' if self.val_loss is None else self.val_loss}\t"
            f"{'' if self.tokens_trained is None else int(self.tokens_trained)}\n"
        )
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    # ---------- state saving ----------

    def _save_state(self, checkpoint_filename, include_training_states=False):
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        model = unwrap_model(self.model)
        checkpoint = {
            "model": model.state_dict(),
            "epoch": self.epoch,
            "step": self.step,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "tokens_trained": self.tokens_trained,
        }
        if include_training_states:
            if self.optimizer is not None:
                checkpoint["optimizer"] = self.optimizer.state_dict()
            if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
                checkpoint["scheduler"] = self.scheduler.state_dict()
            if self.scaler is not None and hasattr(self.scaler, "state_dict"):
                checkpoint["scaler"] = self.scaler.state_dict()
        torch.save(checkpoint, checkpoint_path)

    def _update_state(self, epoch, step=0, train_loss=None, val_loss=None, tokens_trained=None):
        self.epoch = int(epoch)
        self.step = int(step) if step is not None else 0
        if train_loss is not None:
            self.train_loss = train_loss
        if val_loss is not None:
            self.val_loss = val_loss
        if tokens_trained is not None:
            self.tokens_trained = tokens_trained

    def _save_best(self):
        self._remove_old("best_epoch*_step*_val=*.pt")
        filename = self._checkpoint_filename(self.epoch, self.step, self.val_loss, prefix="best")
        self._save_state(filename, include_training_states=False)
        self._update_log("best")
        
    def _save_recent(self):
        self._remove_old("recent_epoch*_step*_val=*.pt")
        filename = self._checkpoint_filename(self.epoch, self.step, self.val_loss, prefix="recent")
        self._save_state(filename, include_training_states=True)
        self._update_log("recent")

    def _save_epoch(self):
        filename = self._checkpoint_filename(self.epoch, None, self.val_loss)
        self._save_state(filename, include_training_states=False)
        self._update_log("epoch")
    
    def _save_step(self):
        filename = self._checkpoint_filename(self.epoch, self.step, self.val_loss)
        self._save_state(filename, include_training_states=False)
        self._update_log("step")

    # ---------- checkpoint saving ----------

    def save_checkpoint(self, epoch, step=None, train_loss=None, val_loss=None, tokens_trained=None):
        
        if self.is_main_process:
    
            self._update_state(epoch, step=step, train_loss=train_loss, val_loss=val_loss, tokens_trained=tokens_trained)

            self._save_recent()

            if step is None:
                self.step = 0
                self._save_epoch()
            else:
                self._save_step()

            if (self.val_loss is not None) and (self.val_loss < self.best_val_loss):
                self.best_val_loss = self.val_loss
                self._save_best()
        
        self._barrier()

    # ---------- loading ----------

    def _load_training_states(self, state):
        if self.optimizer is not None and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler is not None and "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])

        self.epoch = int(state.get("epoch", 0))
        self.step = int(state.get("step", 0))
        self.train_loss = state.get("train_loss", None)
        self.val_loss = state.get("val_loss", None)
        self.tokens_trained = state.get("tokens_trained", 0)

    def _load_model_states(self, state):        
        unwrap_model(self.model).load_state_dict(state["model"])

    def _load_checkpoint_states(self, checkpoint_type, load_model_states=True, load_training_states=False):
        
        assert load_model_states or load_training_states, "One of load_model_states or load_training_states must be True"
        
        if checkpoint_type == "best":
            candidates = sorted(glob(str(self.checkpoint_dir / "best_epoch*_val=*.pt")))
            if not candidates:
                Logger.log("No best checkpoint found")
                return
            ckpt_path = Path(candidates[-1])

        elif checkpoint_type == "recent":
            candidates = sorted(glob(str(self.checkpoint_dir / "recent_epoch*_val=*.pt")))
            if not candidates:
                Logger.log("No recent checkpoint found")
                return
            ckpt_path = Path(candidates[-1])

        elif checkpoint_type.startswith("epoch_"):
            pattern = str(self.checkpoint_dir / f"{checkpoint_type}_val=*.pt")
            candidates = sorted(glob(pattern))
            if candidates:
                ckpt_path = Path(candidates[-1])
            else:
                Logger.log(f"No checkpoint found for {checkpoint_type}")
                return

        else:
            Logger.log(f"Invalid mode: {checkpoint_type}")
            return

        state = torch.load(str(ckpt_path), map_location=self.map_device, weights_only=False)
        
        if load_model_states:
            self._load_model_states(state)    
            Logger.log(f"Loaded model checkpoint from {ckpt_path}")
        if load_training_states:
            self._load_training_states(state)
            Logger.log(f"Loaded training states from {ckpt_path}")
        
    def load_model_states(self, checkpoint_type="recent"):
        self._load_checkpoint_states(checkpoint_type, load_model_states=True, load_training_states=False)
        self._barrier()

    def load_training_states(self, checkpoint_type="recent"):
        self._load_checkpoint_states(checkpoint_type, load_model_states=False, load_training_states=True)
        self._barrier()
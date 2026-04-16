#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Attention DPO Tuner
GUI-утилита для обучения Phase Attention LoRA адаптеров методом DPO
"""

import os
import sys
import json
import threading
import queue
import traceback
import inspect
from datetime import datetime
from collections import OrderedDict

# Monkey patching для совместимости с некоторыми моделями
import transformers
import transformers.models.auto.modeling_auto as modeling_auto

if not hasattr(modeling_auto, "MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES"):
    modeling_auto.MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict()
if not hasattr(modeling_auto, "MODEL_FOR_VISION_2_SEQ_MAPPING"):
    modeling_auto.MODEL_FOR_VISION_2_SEQ_MAPPING = OrderedDict()

import torch
import customtkinter as ctk
from tkinter import filedialog, messagebox
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, set_seed,
    BitsAndBytesConfig, TrainingArguments, TrainerCallback, TrainerControl, TrainerState
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import DPOTrainer

try:
    from trl import DPOConfig
    HAS_DPO_CONFIG = True
except ImportError:
    HAS_DPO_CONFIG = False


# ==============================================================================
# PHASE ATTENTION LAYER
# ==============================================================================

class PhaseAttentionHybrid(torch.nn.Module):
    """
    Кастомный attention-слой с фазовым кодированием.
    LoRA обучает только phase_q и phase_k.
    """
    
    def __init__(self, base_attn, config):
        super().__init__()
        self.base_attn = base_attn
        self.config = config

        self.num_heads = getattr(config, "num_attention_heads", getattr(config, "num_heads", 16))
        self.hidden_size = getattr(config, "hidden_size", 2048)
        self.head_dim = getattr(base_attn, "head_dim", self.hidden_size // self.num_heads)

        self.phase_q = torch.nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.phase_k = torch.nn.Linear(self.hidden_size, self.num_heads, bias=False)

        torch.nn.init.zeros_(self.phase_q.weight)
        torch.nn.init.zeros_(self.phase_k.weight)

    def forward(self, *args, **kwargs):
        hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states")

        base_out = self.base_attn(*args, **kwargs)
        if isinstance(base_out, tuple):
            attn_output, past = base_out[0], base_out[1:]
        else:
            attn_output, past = base_out, None

        phase_q = self.phase_q(hidden_states)
        phase_k = self.phase_k(hidden_states)

        p_q = phase_q.unsqueeze(2)
        p_k = phase_k.unsqueeze(1)

        phase_factor = torch.cos(p_q - p_k)
        phase_mod = phase_factor.mean(dim=(2, 3)).unsqueeze(-1)

        attn_output = attn_output * (1.0 + 0.1 * phase_mod)

        return (attn_output, *past) if past else attn_output


# ==============================================================================
# STOP CALLBACK
# ==============================================================================

class StopEventCallback(TrainerCallback):
    """Останавливает обучение по сигналу."""
    
    def __init__(self, stop_event):
        self.stop_event = stop_event

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_event.is_set():
            control.should_training_stop = True
        return control


# ==============================================================================
# TRAINING ENGINE
# ==============================================================================

class TrainerEngine:
    """Ядро обучения DPO с Phase Attention."""
    
    def __init__(self, log_queue, stop_event, done_callback=None):
        self.log_q = log_queue
        self.stop_event = stop_event
        self.done_callback = done_callback

    def log(self, msg):
        self.log_q.put(msg)

    def run(self, cfg):
        try:
            set_seed(42)
            self.log("🚀 Инициализация...")

            # Проверки
            if not cfg['model_path']:
                self.log("❌ Выберите папку с моделью")
                return
            if not cfg['data_path']:
                self.log("❌ Выберите файл датасета")
                return
            if not torch.cuda.is_available():
                self.log("❌ CUDA не найдена. Установите PyTorch с CUDA")
                return

            self.log(f"✅ GPU: {torch.cuda.get_device_name(0)}")

            if not os.path.isfile(cfg['data_path']):
                self.log("❌ Файл датасета не найден")
                return
            if not os.path.isdir(cfg['model_path']):
                self.log("❌ Папка модели не найдена")
                return

            # Токенайзер
            self.tokenizer = AutoTokenizer.from_pretrained(
                cfg['model_path'],
                trust_remote_code=True,
                use_fast=False
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

            # Модель (4-bit)
            self.log("📦 Загрузка модели (4-bit NF4)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            model = AutoModelForCausalLM.from_pretrained(
                cfg['model_path'],
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True
            )

            model.config.use_cache = False
            model.config.pad_token_id = self.tokenizer.pad_token_id

            # Phase Attention patching
            self.log("🔧 Внедрение Phase Attention...")
            patched = 0
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                for layer in model.model.layers:
                    if hasattr(layer, 'self_attn') and layer.self_attn is not None:
                        if 'Phase' not in layer.self_attn.__class__.__name__:
                            layer.self_attn = PhaseAttentionHybrid(layer.self_attn, model.config)
                            patched += 1

            self.log(f"✅ Слоёв пропатчено: {patched}")

            if patched == 0:
                self.log("❌ Не удалось пропатчить ни один слой")
                return

            # Подготовка к обучению
            model = prepare_model_for_kbit_training(model)
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

            lora_cfg = LoraConfig(
                r=cfg['lora_r'],
                lora_alpha=cfg['lora_alpha'],
                target_modules=["phase_q", "phase_k"],
                lora_dropout=cfg['lora_dropout'],
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_cfg)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.log(f"📊 Обучаемых параметров: {trainable:,}")

            if trainable == 0:
                self.log("❌ Нет обучаемых параметров")
                return

            # Датасет
            self.log("📖 Загрузка датасета...")
            data = {"prompt": [], "chosen": [], "rejected": []}
            with open(cfg['data_path'], 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if all(k in item for k in ["prompt", "chosen", "rejected"]):
                            data["prompt"].append(f"### Instruction:\n{item['prompt']}\n### Response:\n")
                            data["chosen"].append(item['chosen'])
                            data["rejected"].append(item['rejected'])
                    except:
                        pass

            if len(data["prompt"]) == 0:
                self.log("❌ Датасет пуст")
                return

            dataset = Dataset.from_dict(data).train_test_split(test_size=0.05)
            self.log(f"✅ Пар: {len(dataset['train'])} train, {len(dataset['test'])} test")

            # Аргументы обучения
            base_args = {
                "output_dir": cfg['output_dir'],
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": cfg['grad_accum'],
                "num_train_epochs": cfg['epochs'],
                "learning_rate": cfg['lr'],
                "bf16": True,
                "optim": "paged_adamw_8bit",
                "gradient_checkpointing": True,
                "logging_steps": 5,
                "save_steps": 50,
                "eval_steps": 50,
                "report_to": "none",
                "remove_unused_columns": False,
                "max_grad_norm": 1.0,
                "dataloader_pin_memory": False,
            }

            train_sig = inspect.signature(TrainingArguments.__init__).parameters
            if "gradient_checkpointing_kwargs" in train_sig:
                base_args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

            if "eval_strategy" in train_sig:
                base_args["eval_strategy"] = "steps"
            else:
                base_args["evaluation_strategy"] = "steps"

            if HAS_DPO_CONFIG:
                dpo_sig = inspect.signature(DPOConfig.__init__).parameters
                if "beta" in dpo_sig:
                    base_args["beta"] = 0.1
                if "max_length" in dpo_sig:
                    base_args["max_length"] = cfg['max_seq_len']
                if "max_prompt_length" in dpo_sig:
                    base_args["max_prompt_length"] = cfg['max_seq_len'] // 2
                valid_args = {k: v for k, v in base_args.items() if k in dpo_sig}
                training_args = DPOConfig(**valid_args)
            else:
                valid_args = {k: v for k, v in base_args.items() if k in train_sig}
                training_args = TrainingArguments(**valid_args)

            # Trainer
            trainer_params = inspect.signature(DPOTrainer.__init__).parameters
            trainer_kwargs = {
                "model": model,
                "ref_model": None,
                "args": training_args,
                "train_dataset": dataset['train'],
                "eval_dataset": dataset['test'],
            }

            if "processing_class" in trainer_params:
                trainer_kwargs["processing_class"] = self.tokenizer
            else:
                trainer_kwargs["tokenizer"] = self.tokenizer

            if not HAS_DPO_CONFIG:
                if "beta" in trainer_params:
                    trainer_kwargs["beta"] = 0.1
                if "max_length" in trainer_params:
                    trainer_kwargs["max_length"] = cfg['max_seq_len']
                if "max_prompt_length" in trainer_params:
                    trainer_kwargs["max_prompt_length"] = cfg['max_seq_len'] // 2
                if "label_pad_token_id" in trainer_params:
                    trainer_kwargs["label_pad_token_id"] = -100

            trainer = DPOTrainer(**trainer_kwargs)
            trainer.add_callback(StopEventCallback(self.stop_event))

            # Обучение
            self.log("🔥 Начало обучения...")
            trainer.train()

            if self.stop_event.is_set():
                self.log("⏹️ Остановлено пользователем")
            else:
                self.log("💾 Сохранение...")
                trainer.model.save_pretrained(os.path.join(cfg['output_dir'], "phase_dpo_adapter"))
                self.tokenizer.save_pretrained(cfg['output_dir'])
                self.log("✅ Готово!")

        except Exception as e:
            self.log(f"❌ Ошибка: {str(e)}")
            self.log(traceback.format_exc())
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.log("🧹 VRAM очищена")
            if self.done_callback:
                self.done_callback()


# ==============================================================================
# GUI
# ==============================================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class TrainerGUI(ctk.CTk):
    
    def __init__(self):
        super().__init__()
        self.title("Phase DPO Tuner")
        self.geometry("800x700")
        self.resizable(False, False)

        self.stop_event = threading.Event()
        self.log_queue = queue.Queue()
        self.is_running = False

        self._build_ui()
        self._poll_log()
        self._poll_vram()

    def _build_ui(self):
        # Заголовок
        ctk.CTkLabel(
            self,
            text="🌌 Phase DPO Tuner",
            font=ctk.CTkFont(size=22, weight="bold")
        ).pack(pady=(15, 10))

        # Блок путей
        paths_frame = ctk.CTkFrame(self)
        paths_frame.pack(fill="x", padx=20, pady=5)

        ctk.CTkLabel(
            paths_frame,
            text="📁 Файлы",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=5)

        self.vars = {
            "model_path": ctk.StringVar(value=""),
            "data_path": ctk.StringVar(value=""),
            "output_dir": ctk.StringVar(value=f"./adapter_{datetime.now().strftime('%H%M')}")
        }

        labels = [
            ("model_path", "Модель:"),
            ("data_path", "Датасет:"),
            ("output_dir", "Вывод:")
        ]
        
        for i, (key, label) in enumerate(labels):
            ctk.CTkLabel(paths_frame, text=label).grid(row=i+1, column=0, sticky="e", padx=10, pady=5)
            entry = ctk.CTkEntry(paths_frame, textvariable=self.vars[key], width=480, show="•" if key != "output_dir" else "")
            entry.grid(row=i+1, column=1, sticky="w", padx=5, pady=5)
            if key != "output_dir":
                ctk.CTkButton(
                    paths_frame,
                    text="📂",
                    width=40,
                    command=lambda k=key: self._browse(k)
                ).grid(row=i+1, column=2, padx=5, pady=5)

        # Блок настроек
        settings_frame = ctk.CTkFrame(self)
        settings_frame.pack(fill="x", padx=20, pady=15)

        ctk.CTkLabel(
            settings_frame,
            text="⚙️ Параметры",
            font=ctk.CTkFont(weight="bold")
        ).grid(row=0, column=0, columnspan=8, sticky="w", padx=10, pady=5)

        self.cfg_vars = {
            "max_seq_len": ctk.IntVar(value=256),
            "epochs": ctk.IntVar(value=3),
            "grad_accum": ctk.IntVar(value=8),
            "lora_r": ctk.IntVar(value=16),
            "lora_alpha": ctk.IntVar(value=32),
            "lora_dropout": ctk.DoubleVar(value=0.05),
            "lr": ctk.DoubleVar(value=5e-5),
        }

        for i, (k, v) in enumerate(self.cfg_vars.items()):
            r, c = divmod(i, 4)
            ctk.CTkLabel(settings_frame, text=k).grid(row=r+1, column=c*2, sticky="e", padx=5, pady=5)
            ctk.CTkEntry(settings_frame, textvariable=v, width=80).grid(row=r+1, column=c*2+1, sticky="w", padx=5, pady=5)

        # VRAM индикатор
        self.vram_lbl = ctk.CTkLabel(
            self,
            text="VRAM: --",
            text_color="#00FF00",
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.vram_lbl.pack(anchor="e", padx=20)

        # Кнопки
        btns_frame = ctk.CTkFrame(self, fg_color="transparent")
        btns_frame.pack(fill="x", padx=20, pady=5)

        self.start_btn = ctk.CTkButton(
            btns_frame,
            text="▶️ СТАРТ",
            fg_color="#28a745",
            hover_color="#218838",
            font=ctk.CTkFont(weight="bold"),
            height=40,
            command=self._start
        )
        self.start_btn.pack(side="left", expand=True, fill="x", padx=5)

        self.stop_btn = ctk.CTkButton(
            btns_frame,
            text="🛑 СТОП",
            fg_color="#dc3545",
            hover_color="#c82333",
            font=ctk.CTkFont(weight="bold"),
            height=40,
            state="disabled",
            command=self._stop
        )
        self.stop_btn.pack(side="right", expand=True, fill="x", padx=5)

        # Лог
        self.log_box = ctk.CTkTextbox(
            self,
            state="disabled",
            font=ctk.CTkFont(family="Consolas", size=11),
            wrap="word"
        )
        self.log_box.pack(fill="both", expand=True, padx=20, pady=(5, 20))

        self.log("Готово к работе")
        self.log("")

    def _browse(self, key):
        if key == "model_path":
            path = filedialog.askdirectory(title="Выберите папку с моделью")
        else:
            path = filedialog.askopenfilename(
                title="Выберите датасет",
                filetypes=[("JSONL", "*.jsonl")]
            )
        if path:
            self.vars[key].set(path)

    def log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _poll_log(self):
        try:
            while True:
                self.log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(100, self._poll_log)

    def _poll_vram(self):
        if torch.cuda.is_available():
            f, t = torch.cuda.mem_get_info()
            used = (t - f) / 1024**3
            tot = t / 1024**3
            color = "#FF0000" if used > tot * 0.9 else "#FFA500" if used > tot * 0.7 else "#00FF00"
            self.vram_lbl.configure(text=f"VRAM: {used:.1f}/{tot:.1f} GB", text_color=color)
        self.after(1000, self._poll_vram)

    def _on_training_done(self):
        self.is_running = False
        self.after(0, lambda: (
            self.start_btn.configure(state="normal"),
            self.stop_btn.configure(state="disabled")
        ))

    def _start(self):
        if self.is_running:
            return

        cfg = {k: v.get() for k, v in {**self.vars, **self.cfg_vars}.items()}
        
        self.stop_event.clear()
        self.is_running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        self.log("-" * 40)
        engine = TrainerEngine(self.log_queue, self.stop_event, self._on_training_done)
        threading.Thread(target=engine.run, args=(cfg,), daemon=True).start()

    def _stop(self):
        self.stop_event.set()
        self.log("⏹️ Остановка...")
        self.stop_btn.configure(state="disabled")


if __name__ == "__main__":
    app = TrainerGUI()
    app.mainloop()

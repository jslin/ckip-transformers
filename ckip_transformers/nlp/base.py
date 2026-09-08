# -*- coding:utf-8 -*-
import warnings
from typing import List, Union, Any
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from ckip_transformers.backend import detect_best_backend


class CkipTokenClassificationBase:
    """重構後的通用 NLP Token Classification 核心基類"""

    def __init__(
        self,
        model_name: str,
        device: Union[int, str, torch.device] = "auto",
        use_fp16: bool = True,
        use_mlx_if_available: bool = True,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16

        # 1. 自動偵測最佳運算引擎
        req_dev = "mlx" if use_mlx_if_available and device == "auto" else device
        self.backend_type, self.target_device = detect_best_backend(req_dev)

        if self.backend_type == "mlx":
            print(f"[CKIP-Engine] 🚀 檢測到 Apple Silicon，啟用 MLX 原生 Metal GPU 加速！")
            self._init_mlx_model()
        else:
            print(f"[CKIP-Engine] ⚡ 使用 PyTorch 引擎 (Device: {self.target_device})")
            self._init_torch_model()

    def _init_torch_model(self):
        """初始化現代化的 PyTorch 引擎"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)

        # 混合精度設定
        if self.use_fp16 and self.target_device.type in ["cuda", "mps"]:
            if self.target_device.type == "cuda":
                self.model = self.model.half()

        self.model.to(self.target_device)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def _init_mlx_model(self):
        """初始化 MLX 引擎"""
        try:
            import mlx.core as mx
            from huggingface_hub import snapshot_download
            import json, os

            # 下載/載入模型與配置
            model_path = snapshot_download(repo_id=self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            with open(os.path.join(model_path, "config.json"), "r") as f:
                config = json.load(f)
            self.id2label = {int(k): v for k, v in config.get("id2label", {}).items()}

            # 載入自訂 MLX Bert 類別
            from bert_mlx import BertForTokenClassification as MlxBert

            self.mlx_model = MlxBert(config)
            weights_file = os.path.join(model_path, "weights.safetensors")
            if not os.path.exists(weights_file):
                weights_file = os.path.join(model_path, "model.safetensors")
            self.mlx_model.load_weights(weights_file)
            self.mlx_model.eval()
        except Exception as e:
            warnings.warn(f"MLX 初始化失敗，自動降級至 PyTorch: {e}")
            self.backend_type = "torch"
            self.target_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self._init_torch_model()

    def __call__(self, input_text: List[Any], batch_size: int = 256, max_length: int = 512, **kwargs):
        if self.backend_type == "mlx":
            return self._predict_mlx(input_text, batch_size, max_length)
        else:
            return self._predict_torch(input_text, batch_size, max_length)

    @torch.inference_mode()  # 修復 Bug: 解決推論記憶體洩漏 (VRAM OOM)
    def _predict_torch(self, input_text: List[Any], batch_size: int, max_length: int):
        results = []
        for i in range(0, len(input_text), batch_size):
            batch = input_text[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(self.target_device)

            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            # 解碼轉換
            for idx, text in enumerate(batch):
                pred_ids = predictions[idx]
                tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][idx])
                results.append(self._post_process(text, tokens, pred_ids))
        return results

    def _predict_mlx(self, input_text: List[Any], batch_size: int, max_length: int):
        import mlx.core as mx
        results = []
        for text in input_text:
            inputs = self.tokenizer(text, return_tensors="np", truncation=True, max_length=max_length)
            input_ids = mx.array(inputs["input_ids"])
            attention_mask = mx.array(inputs["attention_mask"])

            logits = self.mlx_model(input_ids=input_ids, attention_mask=attention_mask)
            mx.eval(logits)  # 強制觸發 Metal GPU 運算，防護 Lazy Evaluation 記憶體堆積

            preds = mx.argmax(logits[0], axis=-1).tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            results.append(self._post_process(text, tokens, preds))
        return results

    def _post_process(self, text, tokens, pred_ids):
        # 繼承類別將實現具體的斷詞/POS/NER重組解碼邏輯
        raise NotImplementedError
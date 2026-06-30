# Arabic NLP Pipeline

Project này triển khai pipeline phân tích yêu cầu người dùng theo kiểu decomposed multi-task cho dữ liệu AISA, hỗ trợ:

- huấn luyện mô hình bằng LoRA trên model causal LM
- chạy inference cho nhiệm vụ phát hiện hàm/tool, tên hàm và arguments
- có thể dùng OpenAI-compatible API cho bước argument extraction

## 1. Yêu cầu hệ thống

- Python 3.10+
- CUDA-compatible GPU để training
- Git

## 2. Cài đặt môi trường

```bash
cd c:\coding_space\paper\PP2.ArabicNLP2026
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_aisa.txt
```

Nếu đang dùng Windows PowerShell, có thể dùng:

```powershell
cd c:\coding_space\paper\PP2.ArabicNLP2026
.\.venv\Scripts\Activate.ps1
```

## 3. Cấu hình biến môi trường

Tạo file `.env` ở thư mục gốc với nội dung như sau:

```env
OPENAI_API_KEY=your_api_key
LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_MODEL=your_model_name
```

Nếu không dùng OpenAI cho bước argument extraction, bạn có thể bỏ qua các biến này và chạy local model.

## 4. Huấn luyện mô hình

Ví dụ huấn luyện trên model gốc:

```bash
python -m src.main --mode train --model_id <model_id> --dataset_id <dataset_id> --output_dir outputs/aisa_decomposed
```

Ví dụ với một model cụ thể:

```bash
python -m src.main --mode train --model_id google/gemma-2-2b-it --dataset_id aisa/aisa --output_dir outputs/aisa_decomposed
```

Một số tham số quan trọng:

- `--num_train_epochs`
- `--learning_rate`
- `--per_device_train_batch_size`
- `--gradient_accumulation_steps`
- `--max_length`
- `--negative_repeat`

> Training yêu cầu GPU vì code dùng CUDA và PEFT/LoRA.

## 5. Chạy inference

Sau khi đã có checkpoint hoặc sau khi train xong:

```bash
python -m src.main --mode infer --model_id <base_model_id> --checkpoint_dir outputs/aisa_decomposed --output_dir outputs/aisa_decomposed
```

Nếu muốn chạy cả train + infer:

```bash
python -m src.main --mode all --model_id <base_model_id> --dataset_id <dataset_id> --output_dir outputs/aisa_decomposed
```

## 6. Dùng OpenAI-compatible API cho argument extraction

Bật chế độ này bằng tham số:

```bash
python -m src.main --mode infer --model_id <base_model_id> --checkpoint_dir outputs/aisa_decomposed --output_dir outputs/aisa_decomposed --use_openai_args --openai_args_model <model_name>
```

## 7. Kết quả đầu ra

Sau khi chạy, thư mục `outputs/aisa_decomposed` sẽ chứa:

- `aisa_dev_submission.jsonl`
- `aisa_dev_debug.jsonl`
- `decomposed_training_config.json` (nếu train)
- `retrieval_db.jsonl` (nếu bật `--build_retrieval_db`)

## 8. Ghi chú quan trọng

- File `.env` nên được đặt ở thư mục gốc của project.
- Nếu gặp lỗi về dataset hoặc schema, hãy kiểm tra `args.dataset_id` và `args.dataset_revision`.
- Nếu chỉ muốn chạy inference trên checkpoint đã có sẵn, dùng `--mode infer` và truyền `--checkpoint_dir`.

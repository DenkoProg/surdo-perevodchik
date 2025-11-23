# Project Roadmap: Surzhyk & Dialect Normalization System

**Repository**: `surdo-perevodchik`
**Related Projects**: `diploma/unlp` (pravopysnyk), `diploma/vuyko-hutsul`

---

## 1. Project Overview

**Goal**: Build a system to automatically normalize text from 13 Ukrainian dialects and Russian-Ukrainian Surzhyk into standard literary Ukrainian.

**Approach**:

1. Generate a synthetic parallel corpus (Dialect/Surzhyk ↔ Standard Ukrainian) using RAG + LLM generation + programmatic errorification.
2. Train a baseline Seq2Seq model (mT5 or mBART) to establish performance benchmarks.
3. Fine-tune/Instruction-tune a state-of-the-art Ukrainian LLM (Lapa) with LoRA/QLoRA for production-quality results.

**Key Metrics**: BLEU, chrF++, TER, and LLM-as-a-judge scoring (fluency, adequacy, dialectal quality).

## 2. Target Dialects & Varieties

### Surzhyk

- **Russian-Ukrainian Surzhyk** (Priority: High)

### Northern Dialects

1. **Polesian** (Western, Central, Eastern Polesian consolidated)

### South-Eastern Dialects

2. Middle Dnieprian
3. Slobozhan
4. Steppe

### South-Western Dialects

5. Volynian
6. Podillian
7. Upper Dniestrian
8. Sian
9. Pokuttia-Bukovynian
10. **Hutsul** (Reference implementation available)
11. Boyko
12. Trans-Carpathian
13. Lemkian

---

## 3. Pipeline & Tasks

### Phase 1: Data Preparation & Synthetic Corpus Generation

_Objective: Create a large-scale parallel corpus (Source: Dialect/Surzhyk -> Target: Standard Ukrainian)._

#### 1.1 Project Structure Setup

- [ ] **Create Directory Structure**

  ```
  surdo-perevodchik/
  ├── data/
  │   ├── raw/                     # Raw text corpora and HF dataset exports
  │   ├── embeddings/              # FAISS indices and embeddings
  │   ├── dictionaries/            # Dialect-specific word mappings
  │   ├── parallel/                # Generated parallel corpora
  │   │   ├── surzhyk/
  │   │   ├── hutsul/
  │   │   ├── polesian/
  │   │   └── ...
  │   └── splits/                  # Train/val/test splits
  ├── prompts/                     # Linguistic rule prompts for each dialect
  │   ├── surzhyk_rules.txt
  │   ├── hutsul_rules.txt (from vuyko-hutsul)
  │   ├── polesian_rules.txt
  │   └── ...
  ├── src/
  │   ├── surdo_perevodchik/       # Python package root (package-style layout)
  │   │   ├── __init__.py
  │   │   ├── data_generation/
  │   │   │   ├── rag_pipeline.py  # Generic RAG-based generation
  │   │   │   ├── errorification.py# Programmatic error injection
  │   │   │   ├── embeddings.py    # OpenAI/local embedding functions
  │   │   │   └── quality_filter.py# LLM-based quality filtering
  │   │   ├── training/
  │   │   │   ├── train_mt5.py     # mT5/mBART training
  │   │   │   ├── train_lapa.py    # Lapa LLM fine-tuning
  │   │   │   └── data_collators.py
  │   │   ├── evaluation/
  │   │   │   ├── metrics.py       # BLEU, chrF++, TER
  │   │   │   └── llm_judge.py     # GPT-4 based scoring
  │   │   └── utils/
  │   │       ├── tokenizers.py
  │   │       └── preprocessing.py
  │   ├── notebooks/               # Exploratory analysis & prototyping
  │   ├── configs/                 # Hydra/YAML training configs
  │   ├── tests/
  │   └── scripts/                 # CLI scripts for pipeline stages
  ```

- [ ] **Dependencies Installation**

  ```bash
  # Core ML libraries
  pip install transformers==4.47.1 datasets accelerate peft
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

  # Data processing
  pip install spacy-udpipe pymorphy2 faiss-cpu scikit-learn pandas
  pip install sentencepiece protobuf

  # Evaluation & metrics
  pip install sacrebleu openai wandb

  # NLP utilities
  pip install tiktoken difflib
  ```

#### 1.2 Base Data Collection

-- [x] **Gather Standard Ukrainian Corpora**

  - [ ] Use Hugging Face datasets as base corpora

    - [x] Load `hutsul/hutsul-synthetic` `hutsul/hutsul-manually-annotated` and `grammarly/spivavtor` via `datasets.load_dataset`
    - [x] Extract the `source` column from `hutsul/hutsul-synthetic` and `hutsul/hutsul-manually-annotated` and the `target` column from `grammarly/spivavtor`.
    - [x] Combine and deduplicate these columns to produce the base standard-Ukrainian sentence pool and dialect-seed pools as needed.
  

    - [ ] Clean and tokenize using the `unlp/dataset/create_dataset.py` pattern:
      - Split into sentences with spaCy UDPipe
      - Remove duplicates, filter by length (10-150 tokens)
      - Save cleaned export in `data/raw/standard_ukrainian.txt` and keep original HF exports in `data/raw/hf_exports/`

-- [ ] **Collect Dialect-Specific Resources**
  - [x] **Hutsul**: Use HF datasets `hutsul/hutsul-synthetic` and `hutsul/hutsul-manually-annotated` directly (use their `source` column) — no regeneration required. Also reuse `vuyko-hutsul` resources (dictionary, examples) for reference and validation.
  - [ ] **Surzhyk**: Compile Russism dictionaries, calque patterns
  - [ ] **Other dialects**: Research linguistic papers, folklore collections, dialect dictionaries
  - [ ] Store in `data/dictionaries/{dialect}_dictionary.csv` (columns: `standard`, `dialect`, `pos_tag`)

#### 1.3 Embedding & RAG Infrastructure

- [ ] **Create FAISS Index for Standard Ukrainian**

  - [ ] Adapt `vuyko-hutsul/RAG/create_rag.py`:

    ```python
    # src/data_generation/embeddings.py
    from openai import OpenAI
    import faiss
    import numpy as np

    def create_embeddings(sentences, model="text-embedding-3-large", batch_size=100):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            response = client.embeddings.create(input=batch, model=model)
            embeddings.extend([e.embedding for e in response.data])
        return np.array(embeddings)

    def build_faiss_index(embeddings):
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index
    ```

  - [ ] Generate embeddings for all standard Ukrainian sentences
  - [ ] Save index: `data/embeddings/standard_ukrainian_faiss.bin`
  - [ ] Save metadata: `data/embeddings/standard_ukrainian_metadata.csv`

- [ ] **Implement Generic RAG Retrieval**
  ```python
  # src/data_generation/rag_pipeline.py
  def get_closest_sentences(index, metadata_df, query_text, top_n=5):
      query_embedding = get_embedding(query_text)
      distances, indices = index.search(np.array([query_embedding]), top_n)
      return metadata_df.iloc[indices[0]]
  ```

#### 1.4 Linguistic Rule Prompts Creation

- [ ] **Surzhyk Rules Prompt** (`prompts/surzhyk_rules.txt`)

  - [ ] Phonetic russification patterns:
    - "і" → "и", "є" → "е", "ґ" → "г"
    - Unstressed "о" → "а" (Russian akan'e)
  - [ ] Lexical calques: "повинен" → "должен", "тільки" → "только"
  - [ ] Grammatical interference:
    - Russian verb conjugations (-ть endings)
    - Case confusion (esp. instrumental)
    - Preposition substitution ("про" → "о")
  - [ ] Reference: Use `unlp/errorification/RussismErrorifier.py` patterns

- [ ] **Polesian Rules** (`prompts/polesian_rules.txt`)

  - [ ] Research northern vowel systems (akanje, jakanje)
  - [ ] Consonant devoicing patterns
  - [ ] Lexical archaisms specific to Polissya region

- [ ] **South-Western Dialects** (Boyko, Lemkian, Trans-Carpathian)

  - [ ] Vowel shifts similar to Hutsul but less extreme
  - [ ] Hungarian/Slovak/Polish loanwords
  - [ ] Mountain terminology

- [ ] **South-Eastern Dialects** (Slobozhan, Steppe)
  - [ ] Russian influence patterns (border regions)
  - [ ] Cossack-era vocabulary
  - [ ] Consonant cluster simplifications

**Format**: Each prompt should follow the structure of `vuyko-hutsul/prompts/hutsul_rules_prompt.txt`:

- Section 1: Vowel Changes (numbered rules with examples)
- Section 2: Consonant Changes
- Section 3: Grammatical Transformations
- Section 4: Word Order & Syntax
- Section 5: Lexical Substitutions

#### 1.5 Programmatic Errorification (Surzhyk Priority)

- [ ] **Adapt `unlp/errorification` Modules**

  - [ ] Extract core logic from `SurzhErrorifier.py` and `RussismErrorifier.py`
  - [ ] Create modular error injection pipeline:

    ```python
    # src/data_generation/errorification.py
    from unlp.errorification.RussismErrorifier import RussismErrofifier

    class SurzhykGenerator:
        def __init__(self):
            self.errorifier = RussismErrofifier()

        def generate_surzhyk_pairs(self, standard_sentences, error_rate=0.15):
            # Injects Russian-influenced errors programmatically
            surzhyk_sentences = self.errorifier.errorify_rus_dataset(
                standard_sentences, prob=error_rate
            )
            return list(zip(surzhyk_sentences, standard_sentences))
    ```

  - [ ] Generate 20K+ Surzhyk pairs programmatically
  - [ ] Validate with native speakers or linguistic heuristics

#### 1.6 LLM-Based Synthetic Generation

- [ ] **Implement Generation Loop** (adapt `vuyko-hutsul/RAG/generate_data.py`)

  ```python
  # src/data_generation/rag_pipeline.py
  import tiktoken

  def generate_dialect_corpus(
      standard_sentences,
      faiss_index,
      metadata_df,
      dialect_dict,
      rule_prompt_path,
      dialect_name,
      model="gpt-4o",
      batch_token_limit=20000
  ):
      """
      Generates dialect translations using RAG + LLM.

      For each standard sentence:
      1. Retrieve top-5 similar sentences from FAISS
      2. Find relevant dialect word mappings from dictionary
      3. Build prompt with rules + examples + sentence batch
      4. Send to LLM when token limit reached
      5. Parse and save results
      """
      with open(rule_prompt_path) as f:
          rules_prompt = f.read()

      client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
      encoding = tiktoken.get_encoding("cl100k_base")

      results = []
      batch_sentences = []
      batch_rag_examples = []
      batch_dict_words = []

      for sentence in tqdm(standard_sentences):
          # RAG retrieval
          similar = get_closest_sentences(faiss_index, metadata_df, sentence, top_n=3)
          rag_examples = "\n".join(similar['dialect_text'].tolist())

          # Dictionary lookup
          dict_words = find_dialect_words(sentence, dialect_dict)

          batch_sentences.append(sentence)
          batch_rag_examples.append(rag_examples)
          batch_dict_words.append(dict_words)

          # Build prompt and check token count
          prompt = build_translation_prompt(
              rules_prompt, batch_sentences, batch_rag_examples, batch_dict_words
          )
          token_count = len(encoding.encode(prompt))

          if token_count > batch_token_limit:
              # Send batch to LLM
              response = client.chat.completions.create(
                  model=model,
                  messages=[{"role": "user", "content": prompt}]
              )
              translations = parse_translations(response.choices[0].message.content)
              results.extend(zip(batch_sentences, translations))

              # Reset batch
              batch_sentences, batch_rag_examples, batch_dict_words = [], [], []

      return results
  ```

- [ ] **Generation Schedule**
  - [ ] **Hutsul**: Use existing data + generate 10K more pairs for augmentation
  - [ ] **Surzhyk**:
    - 20K programmatic (from errorification)
    - 30K LLM-generated (diverse contexts)
  - [ ] **Other dialects**: 15K pairs each (195K total for 13 dialects)
  - [ ] **Grand Total Target**: ~300K parallel sentence pairs

#### 1.7 Quality Filtering & Validation

- [ ] **Implement LLM-as-a-Judge Filter** (adapt `vuyko-hutsul/eval/eval_llm.py`)

  ```python
  # src/data_generation/quality_filter.py
  def score_translation_quality(source, generated_dialect, dialect_name):
      prompt = f"""
      Rate this {dialect_name} translation on:
      1. Fluency (1-5): Natural and grammatical?
      2. Adequacy (1-5): Preserves original meaning?
      3. Dialectal Authenticity (1-5): Proper {dialect_name} features?

      Return JSON: {{"fluency": x, "adequacy": y, "dialect": z}}

      Standard: {source}
      {dialect_name}: {generated_dialect}
      """
      response = client.chat.completions.create(
          model="gpt-4o-mini",  # Cheaper for filtering
          messages=[{"role": "user", "content": prompt}],
          temperature=0
      )
      scores = extract_json(response.choices[0].message.content)
      return scores

  def filter_corpus(parallel_pairs, min_avg_score=3.5):
      filtered = []
      for source, dialect in tqdm(parallel_pairs):
          scores = score_translation_quality(source, dialect, "Surzhyk")
          avg_score = (scores['fluency'] + scores['adequacy'] + scores['dialect']) / 3
          if avg_score >= min_avg_score:
              filtered.append((source, dialect, scores))
      return filtered
  ```

- [ ] **Filtering Criteria**:

  - [ ] Minimum average LLM score: 3.5/5
  - [ ] Remove duplicates (exact matches)
  - [ ] Remove pairs with length ratio > 2.0 (likely errors)
  - [ ] Sample 10% for manual human validation

- [ ] **Save Filtered Data**
  - Format: JSONL with metadata
    ```json
    {"source": "...", "target": "...", "dialect": "surzhyk", "scores": {...}, "method": "llm_generated"}
    ```
  - Paths: `data/parallel/{dialect}/corpus_filtered.jsonl`

### Phase 2: Baseline Model (Seq2Seq - mT5/mBART)

_Objective: Establish a baseline performance metric using encoder-decoder architectures._

#### 2.1 Data Preprocessing

- [ ] **Create Train/Val/Test Splits**

  ```python
  # src/training/data_preprocessing.py
  from sklearn.model_selection import train_test_split
  import json

  def create_splits(corpus_path, test_size=0.1, val_size=0.1):
      """
      Stratified split ensuring each dialect is represented.
      """
      with open(corpus_path) as f:
          data = [json.loads(line) for line in f]

      # Group by dialect
      dialect_groups = {}
      for item in data:
          dialect = item['dialect']
          if dialect not in dialect_groups:
              dialect_groups[dialect] = []
          dialect_groups[dialect].append(item)

      # Split each dialect proportionally
      train, val, test = [], [], []
      for dialect, items in dialect_groups.items():
          temp_train, temp_test = train_test_split(items, test_size=test_size, random_state=42)
          temp_train, temp_val = train_test_split(temp_train, test_size=val_size/(1-test_size), random_state=42)
          train.extend(temp_train)
          val.extend(temp_val)
          test.extend(temp_test)

      return train, val, test
  ```

- [ ] **Data Format Conversion**
  - [ ] Convert JSONL to Hugging Face `datasets` format
  - [ ] Create parallel text files: `source.txt` (dialect) and `target.txt` (standard)
  - [ ] Save splits to `data/splits/{train,val,test}.json`

#### 2.2 Model Selection & Setup

- [ ] **Choose Base Model**

  - Option 1: `google/mt5-base` (580M params, multilingual)
  - Option 2: `facebook/mbart-large-50` (611M params, better for Slavic languages)
  - **Recommended**: mBART (used in `unlp/seq2seq/train.py`)

- [ ] **Tokenizer Configuration**

  ```python
  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(
      "facebook/mbart-large-50",
      src_lang="uk_UA",
      tgt_lang="uk_UA"
  )
  ```

#### 2.3 Training Implementation

- [ ] **Adapt `unlp/seq2seq/train.py`**

  ```python
  # src/training/train_mt5.py
  from transformers import (
      AutoModelForSeq2SeqLM,
      AutoTokenizer,
      Seq2SeqTrainer,
      Seq2SeqTrainingArguments,
      DataCollatorForSeq2Seq
  )
  from datasets import load_dataset

  class DialectNormalizationTrainer:
      def __init__(self, model_checkpoint="facebook/mbart-large-50"):
          self.model_checkpoint = model_checkpoint
          self.tokenizer = AutoTokenizer.from_pretrained(
              model_checkpoint, src_lang="uk_UA", tgt_lang="uk_UA"
          )
          self.model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

      def preprocess_function(self, examples, max_length=128):
          inputs = examples["source"]
          targets = examples["target"]
          model_inputs = self.tokenizer(
              inputs,
              text_target=targets,
              max_length=max_length,
              truncation=True
          )
          return model_inputs

      def train(self, train_dataset, val_dataset, output_dir="./models/mbart-dialect"):
          tokenized_train = train_dataset.map(self.preprocess_function, batched=True)
          tokenized_val = val_dataset.map(self.preprocess_function, batched=True)

          data_collator = DataCollatorForSeq2Seq(
              self.tokenizer,
              model=self.model,
              padding=True
          )

          training_args = Seq2SeqTrainingArguments(
              output_dir=output_dir,
              evaluation_strategy="steps",
              eval_steps=500,
              learning_rate=2e-5,
              per_device_train_batch_size=16,
              per_device_eval_batch_size=32,
              weight_decay=0.01,
              save_total_limit=3,
              num_train_epochs=5,
              predict_with_generate=True,
              fp16=True,
              push_to_hub=False,
              logging_steps=100,
              save_steps=500,
              load_best_model_at_end=True,
              metric_for_best_model="eval_loss",
              report_to=["wandb"]
          )

          trainer = Seq2SeqTrainer(
              model=self.model,
              args=training_args,
              train_dataset=tokenized_train,
              eval_dataset=tokenized_val,
              data_collator=data_collator,
              tokenizer=self.tokenizer
          )

          trainer.train()
          trainer.save_model(output_dir)
  ```

- [ ] **Training Configuration**

  - Learning rate: 2e-5
  - Batch size: 16 (with gradient accumulation if needed)
  - Epochs: 5-10 (monitor validation loss)
  - FP16 training for speed
  - Save checkpoints every 500 steps

- [ ] **Multi-Dialect vs Single-Dialect Training**
  - [ ] **Experiment 1**: Train one model on all dialects mixed
  - [ ] **Experiment 2**: Train separate models per dialect
  - [ ] **Experiment 3**: Train with dialect ID tags (e.g., `<surzhyk>`, `<hutsul>`)

#### 2.4 Evaluation

- [ ] **Implement Automatic Metrics** (adapt `vuyko-hutsul/eval/eval.py`)

  ```python
  # src/evaluation/metrics.py
  from sacrebleu import corpus_bleu, corpus_chrf, corpus_ter
  import json

  def evaluate_model(predictions_file, references_file):
      with open(predictions_file) as f:
          predictions = [line.strip() for line in f]
      with open(references_file) as f:
          references = [[line.strip()] for line in f]

      bleu = corpus_bleu(predictions, references)
      chrf = corpus_chrf(predictions, references, word_order=2)
      ter = corpus_ter(predictions, references)

      return {
          "BLEU": bleu.score,
          "chrF++": chrf.score,
          "TER": ter.score
      }
  ```

- [ ] **Per-Dialect Breakdown**

  - [ ] Calculate metrics separately for each dialect
  - [ ] Identify which dialects are hardest to normalize
  - [ ] Save results to `results/mbart_evaluation.json`

- [ ] **Human Evaluation** (100 samples per dialect)
  - [ ] Fluency (1-5): Is the output natural Ukrainian?
  - [ ] Adequacy (1-5): Is the meaning preserved?
  - [ ] Create annotation interface (can use Label Studio or simple spreadsheet)

### Phase 3: Advanced Model (Lapa LLM Fine-Tuning)

_Objective: Achieve SOTA performance using the new Ukrainian Lapa LLM with instruction tuning._

#### 3.1 Model Research & Selection

- [ ] **Investigate Lapa LLM**

  - [ ] Check Hugging Face for official model: `lapa-ai/lapa-*` or similar
  - [ ] Verify model size, architecture (likely Llama/Mistral-based)
  - [ ] Check license and usage restrictions
  - [ ] Alternatives if Lapa unavailable: `meta-llama/Llama-3.2-3B-Instruct` fine-tuned on Ukrainian

- [ ] **Hardware Requirements**
  - [ ] GPU: NVIDIA A100 (40GB) or 4x RTX 4090 recommended
  - [ ] RAM: 64GB+ system RAM
  - [ ] Storage: 500GB for models + data
  - [ ] Consider: Google Colab Pro+, Lambda Labs, RunPod for rentals

#### 3.2 Data Formatting for Instruction Tuning

- [ ] **Convert Parallel Corpus to Instruction Format**

  ```python
  # src/training/format_instruction_data.py
  import json

  def create_instruction_dataset(parallel_corpus_path, output_path):
      """
      Converts (source, target) pairs to instruction-following format.
      """
      instructions = [
          "Переклади цей текст літературною українською мовою:",
          "Нормалізуй наступне речення до стандартної української:",
          "Виправ діалектні та суржикові форми в тексті:",
          "Перетвори цей діалектний текст на літературну мову:",
          "Переклади наступне речення на правильну українську:"
      ]

      formatted_data = []
      with open(parallel_corpus_path) as f:
          for idx, line in enumerate(f):
              item = json.loads(line)
              instruction = instructions[idx % len(instructions)]  # Vary instructions

              formatted_item = {
                  "messages": [
                      {
                          "role": "user",
                          "content": f"{instruction}\n\n{item['source']}"
                      },
                      {
                          "role": "assistant",
                          "content": item['target']
                      }
                  ],
                  "metadata": {
                      "dialect": item.get('dialect', 'unknown'),
                      "method": item.get('method', 'synthetic')
                  }
              }
              formatted_data.append(formatted_item)

      with open(output_path, 'w') as f:
          for item in formatted_data:
              f.write(json.dumps(item, ensure_ascii=False) + '\n')
  ```

- [ ] **Create Chat Template Formatting**
  ```python
  def format_example_for_lapa(example, tokenizer):
      """
      Formats example using Lapa's chat template.
      """
      messages = example["messages"]
      text = tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=False
      )
      return {"text": text}
  ```

#### 3.3 LoRA/QLoRA Configuration

- [ ] **Setup LoRA Training** (adapt `vuyko-hutsul/finetune/Mistral.ipynb`)

  ```python
  # src/training/train_lapa.py
  from transformers import (
      AutoModelForCausalLM,
      AutoTokenizer,
      Trainer,
      TrainingArguments,
      DataCollatorForLanguageModeling
  )
  from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
  from datasets import load_dataset
  import torch

  class LapaFineTuner:
      def __init__(self, model_name="lapa-ai/lapa-7b"):
          self.model_name = model_name
          self.tokenizer = AutoTokenizer.from_pretrained(model_name)
          self.tokenizer.pad_token = self.tokenizer.eos_token

          # Load model in 4-bit for QLoRA (memory efficient)
          self.model = AutoModelForCausalLM.from_pretrained(
              model_name,
              torch_dtype=torch.bfloat16,
              device_map="auto",
              load_in_4bit=True,  # QLoRA
              bnb_4bit_compute_dtype=torch.bfloat16,
              bnb_4bit_use_double_quant=True,
              bnb_4bit_quant_type="nf4"
          )
          self.model = prepare_model_for_kbit_training(self.model)

      def setup_lora(self):
          lora_config = LoraConfig(
              task_type=TaskType.CAUSAL_LM,
              r=16,                    # Rank (higher = more capacity, 8-64 typical)
              lora_alpha=32,           # Scaling factor (usually 2*r)
              lora_dropout=0.05,       # Regularization
              target_modules=[         # Which layers to adapt
                  "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                  "gate_proj", "up_proj", "down_proj"       # MLP
              ],
              bias="none",
              inference_mode=False
          )
          self.model = get_peft_model(self.model, lora_config)
          print("LoRA trainable parameters:")
          self.model.print_trainable_parameters()
          return self.model

      def train(self, train_dataset, eval_dataset, output_dir="./models/lapa-dialect-lora"):
          training_args = TrainingArguments(
              output_dir=output_dir,
              num_train_epochs=3,
              per_device_train_batch_size=1,      # Small batch for large models
              per_device_eval_batch_size=1,
              gradient_accumulation_steps=128,    # Effective batch = 128
              evaluation_strategy="steps",
              eval_steps=200,
              save_steps=200,
              learning_rate=5e-5,                 # Slightly higher for LoRA
              weight_decay=0.01,
              fp16=False,
              bf16=True,                          # bfloat16 for stability
              logging_steps=10,
              save_total_limit=3,
              load_best_model_at_end=True,
              metric_for_best_model="eval_loss",
              warmup_steps=100,
              lr_scheduler_type="cosine",
              report_to=["wandb"],
              gradient_checkpointing=True         # Reduce memory
          )

          data_collator = DataCollatorForLanguageModeling(
              tokenizer=self.tokenizer,
              mlm=False  # Causal LM (not masked)
          )

          trainer = Trainer(
              model=self.model,
              args=training_args,
              train_dataset=train_dataset,
              eval_dataset=eval_dataset,
              data_collator=data_collator
          )

          trainer.train()

          # Save LoRA adapters
          self.model.save_pretrained(output_dir)
          self.tokenizer.save_pretrained(output_dir)
  ```

- [ ] **Training Hyperparameters**
  - LoRA rank (r): 16 (balance between capacity and efficiency)
  - Learning rate: 5e-5 (higher than full fine-tuning)
  - Batch size: 1 per device with grad accumulation = 128 (effective batch 128)
  - Epochs: 3-5
  - Optimizer: AdamW with cosine decay
  - Warmup: 100 steps

#### 3.4 Training Execution

- [ ] **Single vs Multi-Task Training**

  - [ ] **Option A**: Single model for all dialects (with dialect tags in prompt)
  - [ ] **Option B**: Separate LoRA adapters per dialect (switchable)
  - [ ] **Recommended**: Option A for simplicity and cross-dialect knowledge transfer

- [ ] **Training Monitoring**

  - [ ] Use Weights & Biases (wandb) for logging
  - [ ] Track: loss, perplexity, gradient norms, learning rate
  - [ ] Save checkpoints every 200 steps
  - [ ] Run validation every 200 steps

- [ ] **Training Script**
  ```bash
  # scripts/train_lapa.sh
  python src/training/train_lapa.py \
    --model_name lapa-ai/lapa-7b \
    --train_data data/splits/train_instruction.jsonl \
    --eval_data data/splits/val_instruction.jsonl \
    --output_dir ./models/lapa-dialect-lora \
    --num_epochs 3 \
    --batch_size 1 \
    --grad_accum 128 \
    --learning_rate 5e-5 \
    --lora_r 16
  ```

#### 3.5 Inference & Generation

- [ ] **Create Inference Pipeline**

  ```python
  # src/inference/generate.py
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from peft import PeftModel

  def load_lapa_model(base_model, lora_adapter_path):
      tokenizer = AutoTokenizer.from_pretrained(base_model)
      model = AutoModelForCausalLM.from_pretrained(
          base_model,
          torch_dtype=torch.bfloat16,
          device_map="auto"
      )
      model = PeftModel.from_pretrained(model, lora_adapter_path)
      model.eval()
      return model, tokenizer

  def normalize_text(model, tokenizer, dialect_text):
      prompt = f"Переклади цей текст літературною українською мовою:\n\n{dialect_text}"
      messages = [{"role": "user", "content": prompt}]

      input_text = tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
      )
      inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

      outputs = model.generate(
          **inputs,
          max_new_tokens=512,
          temperature=0.3,         # Low temp for consistency
          top_p=0.9,
          do_sample=True,
          pad_token_id=tokenizer.eos_token_id
      )

      response = tokenizer.decode(outputs[0], skip_special_tokens=True)
      # Extract assistant response
      assistant_response = response.split("assistant\n")[-1].strip()
      return assistant_response
  ```

#### 3.6 Evaluation & Comparison

- [ ] **Run Comprehensive Evaluation**

  - [ ] Test set predictions for all dialects
  - [ ] Calculate BLEU, chrF++, TER (reuse metrics from Phase 2)
  - [ ] LLM-as-a-judge scoring (fluency, adequacy, correctness)

- [ ] **Model Comparison Table**

  ```markdown
  | Model          | BLEU | chrF++ | TER | Fluency | Adequacy | Inference Speed |
  | -------------- | ---- | ------ | --- | ------- | -------- | --------------- |
  | mBART-base     | TBD  | TBD    | TBD | TBD     | TBD      | ~50 tok/s       |
  | Lapa-7B (LoRA) | TBD  | TBD    | TBD | TBD     | TBD      | ~30 tok/s       |
  ```

- [ ] **Error Analysis**
  - [ ] Identify common failure modes per dialect
  - [ ] Compare which model handles each dialect better
  - [ ] Document edge cases and limitations

#### 3.7 Model Deployment Preparation

- [ ] **Merge LoRA Weights** (optional for deployment)

  ```python
  from peft import PeftModel

  base_model = AutoModelForCausalLM.from_pretrained("lapa-ai/lapa-7b")
  lora_model = PeftModel.from_pretrained(base_model, "./models/lapa-dialect-lora")
  merged_model = lora_model.merge_and_unload()
  merged_model.save_pretrained("./models/lapa-dialect-merged")
  ```

- [ ] **Quantization for Production** (GGUF for llama.cpp, GPTQ, etc.)
- [ ] **API Wrapper** (FastAPI endpoint for normalization service)
- [ ] **Gradio Demo Interface** for testing

---

## 4. Detailed Resource Reuse Strategy

### 4.1 From `vuyko-hutsul` Project

| Component               | File                                                    | How to Adapt                                             | Usage in Our Project                                                         |
| ----------------------- | ------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **Hutsul Rules Prompt** | `prompts/hutsul_rules_prompt.txt`                       | Direct reuse (240 lines of linguistic rules)             | Use as-is for Hutsul dialect; template for creating rules for other dialects |
| **RAG Index Creation**  | `RAG/create_rag.py`                                     | Extract `get_embedding()` and FAISS index building logic | Create embeddings for standard Ukrainian corpus; build search index          |
| **Generation Pipeline** | `RAG/generate_data.py`                                  | Generalize `generate_dialect_corpus()` function          | Core generation loop for all dialects with RAG retrieval + LLM translation   |
| **Dictionary Lookup**   | `RAG/generate_data.py` → `find_similar_words()`         | Adapt for multiple dialect dictionaries                  | Find dialect-specific word mappings during generation                        |
| **Token Counting**      | `RAG/generate_data.py` → `num_tokens_from_string()`     | Use directly with tiktoken                               | Batch prompt construction to stay under API limits                           |
| **Automatic Metrics**   | `eval/eval.py`                                          | Copy evaluation logic (BLEU/chrF++/TER)                  | Benchmark both mBART and Lapa models on test sets                            |
| **LLM Judge**           | `eval/eval_llm.py` → `build_prompt()`, `extract_json()` | Modify prompt template for quality filtering             | Filter generated corpus; score model outputs in evaluation                   |
| **Mistral Fine-tuning** | `finetune/Mistral.ipynb`                                | Extract LoRA config and training loop                    | Template for Lapa LLM fine-tuning with similar hyperparameters               |
| **Data Formatting**     | `finetune/Mistral.ipynb` → `format_example()`           | Adapt for instruction format                             | Convert parallel pairs to chat-style training examples                       |

### 4.2 From `unlp` Project

| Component                  | File/Module                                    | How to Adapt                              | Usage in Our Project                                           |
| -------------------------- | ---------------------------------------------- | ----------------------------------------- | -------------------------------------------------------------- |
| **Surzhyk Errorification** | `errorification/SurzhErrorifier.py`            | Extract `SurzhiksGenerator` class         | Generate realistic Surzhyk programmatically (20K+ pairs)       |
| **Russism Patterns**       | `errorification/RussismErrorifier.py`          | Use `antichanger()` and translation logic | Create Russism dictionary; inject Russian-influenced errors    |
| **Grammar Errors**         | `errorification/GrammarErrofifier.py`          | Review for additional error patterns      | Supplement LLM-generated data with programmatic variants       |
| **Seq2Seq Trainer**        | `seq2seq/train.py` → `PravopysnykTrainer`      | Adapt for mBART-based training            | Baseline model training (Phase 2); compare to Lapa             |
| **Data Preprocessing**     | `seq2seq/train.py` → `preprocess_function()`   | Reuse tokenization logic                  | Prepare parallel data for encoder-decoder models               |
| **Dataset Creation**       | `dataset/create_dataset.py`                    | Use sentence splitting and cleaning       | Process raw Ukrainian text into clean sentences for embeddings |
| **Sentence Splitter**      | `dataset/data_classes/FileSentenceSplitter.py` | Integrate spaCy UDPipe pipeline           | Split scraped/raw text into training sentences                 |
| **Text Cleaning**          | `dataset/data_classes/FileSentenceCleaner.py`  | Apply cleaning heuristics                 | Normalize whitespace, remove duplicates, filter by length      |

### 4.3 Linguistic Resources to Collect

| Dialect Group        | Potential Sources                                          | What to Extract                                                    |
| -------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------ |
| **Hutsul**           | Existing `vuyko-hutsul` data, Ivanchyk corpus              | Rules prompt (already done), dictionary, parallel examples         |
| **Surzhyk**          | Academic papers on Surzhyk, Russian-Ukrainian dictionaries | Phonetic rules, lexical calques, grammatical interference patterns |
| **Polesian**         | Folklore archives, dialectological atlases                 | Northern vowel shifts (akanje), archaic vocabulary                 |
| **Boyko/Lemkian**    | Carpathian ethnographic texts, church documents            | Mountain terminology, Slovak/Polish loanwords                      |
| **Slobozhan/Steppe** | Historical Cossack texts, border region literature         | Russian influence patterns, regional vocabulary                    |
| **All Dialects**     | Ukrainian Linguistic Atlas, academic dissertations         | Phonetic inventories, morphological paradigms, syntax patterns     |

---

## 5. Project Milestones & Timeline

### Milestone 1: Infrastructure & Data Collection (Weeks 1-3)

- ✅ Complete directory structure setup
- ✅ Install all dependencies
- ✅ Collect & clean 100K standard Ukrainian sentences
- ✅ Create FAISS embedding index
- ✅ Adapt RAG generation pipeline from vuyko-hutsul

**Deliverable**: Working RAG infrastructure + clean Ukrainian corpus

### Milestone 2: Surzhyk Corpus Generation (Weeks 4-5)

- ✅ Create `prompts/surzhyk_rules.txt` (linguistic rules)
- ✅ Adapt `unlp/errorification` for programmatic generation (20K pairs)
- ✅ Run LLM-based generation with RAG (30K pairs)
- ✅ Quality filtering with LLM judge (target: 40K high-quality pairs)

**Deliverable**: 40K+ Surzhyk ↔ Standard Ukrainian parallel corpus

### Milestone 3: Multi-Dialect Corpus Expansion (Weeks 6-10)

- ✅ Research & create rule prompts for 12 remaining dialects
- ✅ Collect dialect-specific dictionaries and resources
- ✅ Run generation pipeline for each dialect (15K pairs × 12 = 180K)
- ✅ Filter and validate all generated data
- ✅ Create train/val/test splits (stratified by dialect)

**Deliverable**: 250K+ multi-dialect parallel corpus

### Milestone 4: Baseline Model Training (Weeks 11-12)

- ✅ Implement mBART fine-tuning pipeline
- ✅ Train baseline model on full corpus
- ✅ Run evaluation (BLEU, chrF++, TER)
- ✅ Error analysis and dialect-specific performance breakdown

**Deliverable**: Trained mBART model + evaluation report (baseline metrics)

### Milestone 5: Lapa LLM Fine-Tuning (Weeks 13-15)

- ✅ Format data for instruction tuning
- ✅ Setup LoRA/QLoRA configuration
- ✅ Train Lapa model on multi-dialect corpus
- ✅ Hyperparameter tuning (learning rate, LoRA rank, etc.)
- ✅ Run comprehensive evaluation

**Deliverable**: Fine-tuned Lapa model with LoRA adapters

### Milestone 6: Final Evaluation & Deployment (Weeks 16-17)

- ✅ Head-to-head comparison: mBART vs Lapa
- ✅ Human evaluation (100 samples per dialect)
- ✅ Error analysis and failure mode documentation
- ✅ Create inference API and demo interface
- ✅ Write thesis chapter / technical report

**Deliverable**: Production-ready normalization system + thesis documentation

---

## 6. Success Criteria & Expected Outcomes

### Quantitative Metrics (Target Performance)

- **BLEU Score**: > 50 (good), > 60 (excellent)
- **chrF++ Score**: > 65 (good), > 75 (excellent)
- **TER Score**: < 30 (good), < 20 (excellent)
- **LLM Judge Avg**: > 4.0/5.0 on fluency and adequacy

### Per-Dialect Performance Goals

- **Surzhyk**: Highest performance (most data + programmatic generation)
- **Hutsul**: High performance (existing resources from vuyko-hutsul)
- **Other SW Dialects**: Moderate-high (linguistic similarity to Hutsul)
- **Polesian/SE Dialects**: Moderate (less documented, more research needed)

### Qualitative Goals

1. **Naturalness**: Normalized text sounds like native standard Ukrainian
2. **Meaning Preservation**: No semantic drift or information loss
3. **Consistency**: Same input always produces same output (low temperature)
4. **Robustness**: Handles mixed dialects, code-switching, typos gracefully

---

## 7. Risk Mitigation

| Risk                               | Probability | Impact | Mitigation Strategy                                                               |
| ---------------------------------- | ----------- | ------ | --------------------------------------------------------------------------------- |
| **Lapa LLM unavailable**           | Medium      | High   | Use Llama-3.2-3B or Mistral-7B as fallback; fine-tune on Ukrainian first          |
| **Insufficient dialect resources** | High        | Medium | Focus on top 5 dialects initially; use cross-dialect transfer learning            |
| **LLM generation quality issues**  | Medium      | High   | Use stricter filtering (min score 4.0); add programmatic validation rules         |
| **GPU resource constraints**       | Medium      | High   | Use QLoRA for memory efficiency; rent cloud GPUs (RunPod/Lambda); train in stages |
| **Corpus size too small**          | Low         | Medium | Augment with back-translation; use multi-task learning with GEC data from `unlp`  |
| **Evaluation disagreement**        | Low         | Medium | Conduct human evaluation with native speakers; report inter-annotator agreement   |

---

## 8. Future Extensions

- [ ] **Bidirectional Translation**: Standard Ukrainian → Dialect (for creative writing, education)
- [ ] **Speech Integration**: Combine with ASR/TTS for spoken dialect normalization
- [ ] **Historical Text Normalization**: Extend to Old Ukrainian, Church Slavonic
- [ ] **Cross-Slavic Transfer**: Apply techniques to Belarusian, Rusyn dialects
- [ ] **Real-time Applications**: Browser extension, mobile keyboard, chat bots
- [ ] **Pedagogical Tools**: Interactive dialect learning platform

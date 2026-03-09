import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class DialectTranslator:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        print(f"🔄 Loading model from {model_path} on {self.device} ({self.dtype})...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        print("✅ Model loaded successfully!")

    def translate(
        self,
        text: str,
        num_beams: int = 5,
        max_length: int = 128,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        if not text.strip():
            return ""

        with torch.inference_mode():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=True,
            )

            translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translation


# Initialize translator with the multidialect model
MODEL_PATH = "models/umt5-base-multidialect/final_model"
translator = DialectTranslator(MODEL_PATH)

_asr = None  # (model, processor) tuple


def get_asr():
    global _asr
    if _asr is None:
        from transformers import MCTCTForCTC, MCTCTProcessor

        print("Loading ASR model (first use)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        processor = MCTCTProcessor.from_pretrained("speechbrain/m-ctc-t-large", cache_dir="models/m-ctc-t-large")
        model = MCTCTForCTC.from_pretrained("speechbrain/m-ctc-t-large", torch_dtype=dtype, cache_dir="models/m-ctc-t-large")
        model.to(device)
        model.eval()
        _asr = (model, processor)
        print("ASR model loaded.")
    return _asr


def transcribe_audio(audio_path: str | None) -> str:
    if audio_path is None:
        return ""
    try:
        import torchaudio

        model, processor = get_asr()
        device = next(model.parameters()).device

        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]
    except Exception as e:
        return f"[Помилка розпізнавання: {str(e)}]"


DIALECTS = {
    "Гуцульський (Hutsul)": {
        "code": "hutsul",
        "description": "Гуцульський діалект - говірка українців, що проживають в Карпатах",
        "region": "Івано-Франківська, Чернівецька, Закарпатська області",
        "category": "Південно-західні діалекти",
    },
    "Бойківський (Boyko)": {
        "code": "boiko",
        "description": "Бойківський діалект - карпатська говірка",
        "region": "Львівська, Івано-Франківська області",
        "category": "Південно-західні діалекти",
    },
    "Закарпатський (Trans-Carpathian)": {
        "code": "transcarpathian",
        "description": "Закарпатський діалект - говірка Закарпаття",
        "region": "Закарпатська область",
        "category": "Південно-західні діалекти",
    },
    "Суржик (Surzhyk)": {
        "code": "surzhyk",
        "description": "Російсько-український суржик - змішування української та російської мов",
        "region": "Переважно східні та південні регіони України",
        "category": "Суржик",
    },
}

DIALECT_PREFIXES = {
    "hutsul": "Переклади з гуцульської",
    "boiko": "Переклади з бойківської",
    "transcarpathian": "Переклади з закарпатської",
    "surzhyk": "Переклади з суржику",
}


def translate_text(source_text: str, source_dialect: str, num_beams: int, repetition_penalty: float) -> str:
    """Handle translation with selected parameters."""
    if not source_text.strip():
        return ""

    try:
        code = DIALECTS[source_dialect]["code"]
        prefixed = f"{DIALECT_PREFIXES[code]}: {source_text}"
        translation = translator.translate(
            prefixed,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        return translation
    except Exception as e:
        return f"❌ Помилка перекладу: {str(e)}"


EXAMPLES = [
    [
        "Сонце було вполудне, коли косари сиділи в студні під буком, перепочивати й курити, а жінки винесли им пити в бербеницях розводу.",
        "Гуцульський (Hutsul)",
        5,
        1.2,
    ],
    [
        "Хочу з тобов говорити в штири очи.",
        "Бойківський (Boyko)",
        5,
        1.2,
    ],
    [
        "Го виховували адя та бабуля Елизаветта Шарлота з Пфальцу.",
        "Закарпатський (Trans-Carpathian)",
        5,
        1.2,
    ],
    [
        "Найбольше люди люблять два праздника Новий Год і День рождєнія.",
        "Суржик (Surzhyk)",
        5,
        1.2,
    ],
]

custom_css = """
.dialect-info {
    background: #667eea;
    padding: 20px;
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
}

.dialect-selector {
    font-size: 16px;
}

.translation-area textarea {
    font-size: 16px !important;
    line-height: 1.6 !important;
}

.header-title {
    text-align: center;
    color: white;
    margin-bottom: 10px;
}

.header-subtitle {
    text-align: center;
    color: white;
    font-size: 18px;
    margin-bottom: 30px;
}

.advanced-settings {
    background: #f7fafc;
    padding: 15px;
    border-radius: 8px;
    margin-top: 10px;
}
"""

with gr.Blocks(css=custom_css, title="Діалектний перекладач", theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <div class="dialect-info">
            <h1 class="header-title">🗣️ Діалектний перекладач української мови</h1>
            <p class="header-subtitle">Переклад українських діалектів та суржику на стандартну літературну мову</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📍 Джерело")

            dialect_dropdown = gr.Dropdown(
                choices=list(DIALECTS.keys()),
                value="Гуцульський (Hutsul)",
                label="Оберіть діалект",
                elem_classes=["dialect-selector"],
            )

            def show_dialect_info(dialect_name):
                dialect = DIALECTS[dialect_name]
                category = dialect.get("category", "")
                lines = []
                if category:
                    lines.append(f"**Категорія:** {category}")
                lines.append(f"**Опис:** {dialect['description']}")
                lines.append(f"**Регіон:** {dialect['region']}")
                return "\n".join(lines)

            dialect_info = gr.Markdown(
                value=show_dialect_info("Гуцульський (Hutsul)"),
                elem_classes=["dialect-info-text"],
            )

            dialect_dropdown.change(fn=show_dialect_info, inputs=[dialect_dropdown], outputs=[dialect_info])

            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Аудіо (опційно)",
                format="wav",
            )

            source_text = gr.Textbox(
                label="Текст діалектом",
                placeholder="Введіть текст або запишіть аудіо вище...",
                lines=8,
                elem_classes=["translation-area"],
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Переклад")

            target_text = gr.Textbox(
                label="Переклад",
                lines=8,
                interactive=False,
                elem_classes=["translation-area"],
            )

            translate_btn = gr.Button("🔄 Перекласти", variant="primary", size="lg")

    with gr.Accordion("⚙️ Налаштування моделі", open=False):
        gr.Markdown("*Експериментальні параметри для контролю якості перекладу*")
        with gr.Row():
            num_beams = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Beam Search (більше = точніше, але повільніше)",
            )
            repetition_penalty = gr.Slider(
                minimum=1.0,
                maximum=2.0,
                value=1.2,
                step=0.1,
                label="Штраф за повторення",
            )

    gr.Markdown("---")
    gr.Markdown("### 📚 Приклади")
    gr.Examples(
        examples=EXAMPLES,
        inputs=[source_text, dialect_dropdown, num_beams, repetition_penalty],
        outputs=target_text,
        fn=translate_text,
        cache_examples=False,
    )

    translate_btn.click(
        fn=translate_text,
        inputs=[source_text, dialect_dropdown, num_beams, repetition_penalty],
        outputs=target_text,
    )

    audio_input.change(
        fn=transcribe_audio,
        inputs=[audio_input],
        outputs=[source_text],
    )

    # Footer
    gr.Markdown(
        """
        ---
        <div style='text-align: center; color: #718096; font-size: 14px;'>
        <p>📍 Майбутній функціонал: інтерактивна карта для вибору діалекту за регіоном</p>
        </div>
        """
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )

import asyncio
import os
import tempfile

from fastapi import UploadFile
from nicegui import app as nicegui_app, ui  # noqa: F401 — needed for FastAPI route registration
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class DialectTranslator:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        print(f"Loading model from {model_path} on {self.device} ({self.dtype})...")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        print("Model loaded.")

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
        model = MCTCTForCTC.from_pretrained(
            "speechbrain/m-ctc-t-large", torch_dtype=dtype, cache_dir="models/m-ctc-t-large"
        )
        model.to(device)
        model.eval()
        _asr = (model, processor)
        print("ASR model loaded.")
    return _asr


def transcribe_audio(audio_path: str | None) -> str:
    if audio_path is None:
        return ""
    try:
        import av
        import numpy as np

        model, processor = get_asr()
        device = next(model.parameters()).device

        # PyAV decodes any browser format (WebM, OGG, WAV, …) without system ffmpeg.
        # AudioResampler converts to float32 mono 16 kHz in one step.
        container = av.open(audio_path)
        audio_stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        chunks = []
        for frame in container.decode(audio_stream):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray()[0])
        for rf in resampler.resample(None):  # flush resampler
            chunks.append(rf.to_ndarray()[0])
        container.close()
        audio_array = np.concatenate(chunks).astype("float32")

        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode(), torch.autocast(device_type=device.type):
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]
    except Exception as e:
        return f"[Помилка розпізнавання: {str(e)}]"


DIALECTS = {
    "Суржик (Surzhyk)": {
        "code": "surzhyk",
        "description": "Змішана мова, що виникла внаслідок русифікації: українська граматика та фонетика переплітаються з російською лексикою. Широко вживається в побутовому мовленні.",
        "region": "Переважно центральні, східні та південні регіони України",
        "category": "Суржик",
    },
    "Гуцульський (Hutsul)": {
        "code": "hutsul",
        "description": "Архаїчна карпатська говірка з багатою питомою лексикою, полонізмами та румунськими запозиченнями. Вирізняється самобутньою інтонацією та збереженням давніх форм.",
        "region": "Івано-Франківська, Чернівецька, Закарпатська області",
        "category": "Південно-західні діалекти",
    },
    "Бойківський (Boyko)": {
        "code": "boiko",
        "description": "Говірка бойків - гірського народу Карпат. Близька до гуцульської, але має власну фонетику й лексику з впливами польської та церковнослов'янської мов.",
        "region": "Львівська, Івано-Франківська, частково Закарпатська області",
        "category": "Південно-західні діалекти",
    },
    "Закарпатський (Trans-Carpathian)": {
        "code": "transcarpathian",
        "description": "Надзвичайно самобутня говірка, що формувалась під впливом угорської, словацької та румунської мов упродовж століть. Зберігає архаїчну лексику, майже зниклу в літературній мові.",
        "region": "Закарпатська область",
        "category": "Південно-західні діалекти",
    },
}

DIALECT_PREFIXES = {
    "hutsul": "Переклади з гуцульської",
    "boiko": "Переклади з бойківської",
    "transcarpathian": "Переклади з закарпатської",
    "surzhyk": "Переклади з суржику",
}


def translate_text(source_text: str, source_dialect: str, num_beams: int, repetition_penalty: float) -> str:
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
        return f"Помилка перекладу: {str(e)}"


EXAMPLES = [
    [
        "Найбольше люди люблять два праздника Новий Год і День рождєнія.",
        "Суржик (Surzhyk)",
        5,
        1.2,
    ],
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
]

# =============================================================================
# NiceGUI frontend
# =============================================================================

# FastAPI endpoint used by the in-browser microphone recording JS
@nicegui_app.post("/api/transcribe")
async def api_transcribe(file: UploadFile):
    content = await file.read()
    suffix = os.path.splitext(file.filename or "recording.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, transcribe_audio, tmp_path)
        return {"text": text}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garant:ital,wght@0,400;0,600;1,400&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  /* ===== Design tokens ===== */
  :root {
    --bg:          oklch(97% 0.012 75);
    --surface:     oklch(99% 0.006 75);
    --ink:         oklch(14% 0.025 60);
    --muted:       oklch(52% 0.018 70);
    --gold:        oklch(70% 0.15 78);
    --gold-deep:   oklch(56% 0.16 72);
    --gold-tint:   oklch(96% 0.04 78);
    --blue:        oklch(36% 0.11 252);
    --border:      oklch(89% 0.016 75);
    --border-focus:oklch(70% 0.15 78);
  }

  /* ===== Quasar/NiceGUI overrides ===== */
  *, *::before, *::after { box-sizing: border-box; }

  body, .q-page-container, .q-page, .q-layout {
    background: var(--bg) !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
  }

  body::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--gold);
    z-index: 9999;
  }

  .q-page { padding: 0 !important; }

  /* Hide NiceGUI reconnection notice */
  .nicegui-connection-lost { display: none !important; }

  /* ===== Page wrapper ===== */
  .page-wrap {
    max-width: 1111px;
    width: 100%;
    margin: 0 auto !important;
    padding: 52px 32px 80px !important;
    gap: 0 !important;
    align-items: stretch !important;
    flex-direction: column !important;
  }

  /* ===== Header ===== */
  .lx-header {
    display: flex !important;
    align-items: baseline;
    justify-content: space-between;
    width: 100%;
    padding-bottom: 28px !important;
    margin-bottom: 36px !important;
    border-bottom: 1px solid var(--border);
    gap: 16px;
    flex-wrap: wrap;
  }

  .lx-wordmark {
    font-family: 'Cormorant Garant', serif !important;
    font-size: clamp(2rem, 4vw, 2.75rem) !important;
    font-weight: 600;
    color: var(--ink) !important;
    letter-spacing: -0.03em;
    line-height: 1;
  }

  .lx-tagline {
    font-size: 0.8rem !important;
    color: var(--muted) !important;
    font-weight: 400;
    max-width: 260px;
    text-align: right;
    line-height: 1.55;
  }

  /* ===== Dialect pills ===== */
  .pill-row {
    display: flex !important;
    flex-wrap: wrap;
    gap: 8px !important;
    margin-bottom: 16px !important;
    width: 100%;
    align-items: center !important;
  }

  .dialect-pill.q-btn {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    padding: 7px 18px !important;
    border-radius: 100px !important;
    border: 1.5px solid var(--border) !important;
    background: transparent !important;
    color: var(--muted) !important;
    transition: border-color 0.18s ease, color 0.18s ease, background 0.18s ease !important;
    box-shadow: none !important;
    min-height: unset !important;
    height: auto !important;
  }

  .dialect-pill.q-btn:hover {
    border-color: var(--gold) !important;
    color: var(--ink) !important;
    background: var(--gold-tint) !important;
  }

  .dialect-pill.pill-active.q-btn {
    background: var(--gold) !important;
    color: oklch(16% 0.05 70) !important;
    border-color: var(--gold) !important;
    font-weight: 600 !important;
  }

  /* ===== Info card ===== */
  .info-card {
    background: var(--gold-tint) !important;
    border-left: 3px solid var(--gold);
    border-radius: 0 8px 8px 0;
    padding: 14px 20px;
    margin-bottom: 32px !important;
    width: 100%;
    transition: opacity 0.25s ease;
  }

  .info-inner { display: flex; flex-direction: column; gap: 3px; }
  .info-meta { display: flex; align-items: center; gap: 6px; margin-bottom: 4px; }
  .info-category {
    font-size: 0.7rem; font-weight: 700; color: var(--gold-deep);
    text-transform: uppercase; letter-spacing: 0.08em;
  }
  .info-sep { color: var(--border); font-size: 0.75rem; }
  .info-region { font-size: 0.75rem; color: var(--muted); }
  .info-desc { font-size: 0.875rem; color: var(--ink); margin: 0; line-height: 1.55; }

  /* ===== Main panel (two columns) ===== */
  .main-panel {
    display: grid !important;
    grid-template-columns: 1fr 44px 1fr !important;
    gap: 0 !important;
    width: 100%;
    align-items: start;
    margin-bottom: 20px;
  }

  .panel-col {
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
    align-items: stretch !important;
    min-width: 0;
  }

  .panel-label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: var(--muted) !important;
  }

  .panel-divider {
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding-top: 30px;
    color: var(--border);
    font-size: 1.4rem;
    user-select: none;
    flex-shrink: 0;
  }

  /* ===== Textareas ===== */
  .main-textarea { width: 100% !important; }

  .main-textarea .q-field__control {
    background: var(--surface) !important;
    border-radius: 10px !important;
  }

  .main-textarea.q-field--outlined .q-field__control::before {
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    transition: border-color 0.18s ease !important;
  }

  .main-textarea.q-field--outlined.q-field--focused .q-field__control::before {
    border-color: var(--border-focus) !important;
  }

  .main-textarea.q-field--outlined .q-field__control::after {
    display: none !important;
  }

  .main-textarea .q-field__native {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.975rem !important;
    line-height: 1.75 !important;
    color: var(--ink) !important;
    padding: 14px 16px !important;
    caret-color: var(--gold);
    resize: none;
  }

  .main-textarea .q-field__native::placeholder {
    color: oklch(68% 0.012 75) !important;
  }

  .result-textarea .q-field__control {
    background: oklch(96.5% 0.008 250) !important;
  }

  .result-textarea .q-field__native {
    color: var(--blue) !important;
    cursor: default !important;
  }

  /* ===== Audio upload ===== */
  .audio-upload { width: 100% !important; }

  .audio-upload .q-uploader {
    width: 100% !important;
    min-height: unset !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 8px !important;
    background: transparent !important;
    box-shadow: none !important;
    transition: border-color 0.18s ease !important;
  }

  .audio-upload .q-uploader:hover {
    border-color: var(--gold) !important;
  }

  .audio-upload .q-uploader__header {
    background: transparent !important;
    padding: 8px 14px !important;
    min-height: unset !important;
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
  }

  .audio-upload .q-uploader__subtitle { display: none !important; }

  .audio-upload .q-uploader__list {
    padding: 0 14px 8px !important;
    font-size: 0.8rem !important;
    color: var(--muted) !important;
  }

  /* ===== Translate button ===== */
  .translate-btn.q-btn {
    width: 100% !important;
    background: var(--gold) !important;
    color: oklch(16% 0.05 70) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.975rem !important;
    font-weight: 600 !important;
    text-transform: none !important;
    letter-spacing: 0.01em !important;
    padding: 14px 24px !important;
    border-radius: 10px !important;
    border: none !important;
    transition: background 0.18s ease, transform 0.12s ease, box-shadow 0.18s ease !important;
    box-shadow: 0 2px 14px oklch(70% 0.15 78 / 0.28) !important;
    margin-bottom: 28px;
  }

  .translate-btn.q-btn:hover {
    background: var(--gold-deep) !important;
    box-shadow: 0 4px 22px oklch(70% 0.15 78 / 0.38) !important;
    transform: translateY(-1px) !important;
  }

  .translate-btn.q-btn:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 10px oklch(70% 0.15 78 / 0.22) !important;
  }

  /* Loading spinner inside button */
  .translate-btn .q-spinner { color: oklch(16% 0.05 70) !important; }

  /* ===== Settings expansion ===== */
  .settings-expansion {
    width: 100% !important;
    margin-bottom: 44px !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
  }

  .settings-expansion .q-expansion-item__header {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    padding: 12px 18px !important;
    min-height: unset !important;
  }

  .settings-expansion .q-item__label { color: var(--muted) !important; }
  .settings-expansion .q-item__section--avatar { color: var(--muted) !important; }
  .settings-expansion .q-focus-helper { display: none !important; }

  .settings-expansion .q-expansion-item__content {
    padding: 4px 18px 20px !important;
    border-top: 1px solid var(--border);
  }

  .slider-label {
    display: block;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    margin-bottom: 10px;
    margin-top: 14px;
  }

  .custom-slider .q-slider__track-container { height: 3px !important; }
  .custom-slider .q-slider__track--active { background: var(--gold) !important; }
  .custom-slider .q-slider__track { background: var(--border) !important; }
  .custom-slider .q-slider__thumb { color: var(--gold) !important; }
  .custom-slider .q-slider__thumb path { fill: var(--gold) !important; }
  .custom-slider .q-slider__pin { background: var(--gold) !important; }
  .custom-slider .q-slider__pin::before { border-top-color: var(--gold) !important; }

  /* ===== Section title ===== */
  .section-title {
    font-family: 'Cormorant Garant', serif !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
    margin-bottom: 16px !important;
    letter-spacing: -0.01em;
  }

  /* ===== Examples grid ===== */
  .examples-grid {
    width: 100% !important;
    gap: 12px !important;
    margin-bottom: 60px;
  }

  .example-card.q-card {
    background: var(--surface) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 18px !important;
    cursor: pointer !important;
    transition: border-color 0.18s ease, background 0.18s ease, transform 0.15s ease, box-shadow 0.18s ease !important;
    box-shadow: none !important;
  }

  .example-card.q-card:hover {
    border-color: var(--gold) !important;
    background: var(--gold-tint) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px oklch(70% 0.15 78 / 0.14) !important;
  }

  .example-text {
    font-size: 0.875rem !important;
    color: var(--ink) !important;
    line-height: 1.65 !important;
    margin-bottom: 14px !important;
    display: block;
  }

  .example-badge {
    display: inline-flex;
    align-items: center;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    color: var(--gold-deep) !important;
    background: oklch(92% 0.05 78) !important;
    padding: 3px 11px !important;
    border-radius: 100px !important;
    text-transform: uppercase;
    letter-spacing: 0.07em;
  }

  /* ===== Footer ===== */
  .lx-footer {
    display: flex !important;
    justify-content: center !important;
    padding-top: 24px !important;
    border-top: 1px solid var(--border) !important;
    width: 100%;
  }

  .footer-text {
    font-size: 0.78rem !important;
    color: oklch(64% 0.014 75) !important;
  }

  /* ===== Source textarea wrapper ===== */
  .source-wrap {
    position: relative;
    width: 100%;
  }

  /* ===== Audio controls — vertically centred on the right edge of the input ===== */
  .audio-controls {
    position: absolute;
    top: 50%;
    right: 10px;
    transform: translateY(-50%);
    display: flex;
    gap: 6px;
    align-items: center;
    z-index: 2;
  }

  /* Right padding keeps text from going under the buttons (2 × 40px + gaps + margin) */
  .source-textarea .q-field__native {
    padding-right: 100px !important;
  }

  .audio-icon-btn {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    border: 1.5px solid var(--border);
    background: transparent;
    color: var(--muted);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 1.1rem;
    font-weight: 600;
    line-height: 1;
    user-select: none;
    transition: border-color 0.18s ease, color 0.18s ease, background 0.18s ease, box-shadow 0.18s ease;
    padding: 0;
    text-decoration: none;
  }

  .audio-icon-btn:hover {
    border-color: var(--gold);
    color: var(--ink);
    background: var(--gold-tint);
  }

  #lx-mic-btn.mic-recording {
    border-color: oklch(55% 0.2 22);
    background: oklch(55% 0.2 22);
    color: white;
    animation: mic-pulse 1.4s ease-in-out infinite;
  }

  #lx-mic-btn.mic-loading {
    cursor: wait;
    border-color: var(--gold);
    color: var(--gold);
  }

  @keyframes mic-pulse {
    0%, 100% { box-shadow: 0 0 0 0 oklch(55% 0.2 22 / 0.45); }
    50%       { box-shadow: 0 0 0 9px oklch(55% 0.2 22 / 0); }
  }

  /* ===== Animations ===== */
  @keyframes lx-spin {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
  }

  @keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(5px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .result-textarea .q-field__native {
    animation: fadeSlideIn 0.3s ease both;
  }

  /* ===== Responsive ===== */
  @media (max-width: 640px) {
    .page-wrap { padding: 36px 16px 60px !important; }

    .main-panel {
      grid-template-columns: 1fr !important;
    }

    .panel-divider {
      padding: 6px 0 !important;
      transform: rotate(90deg);
      justify-self: center;
    }

    .lx-tagline { text-align: left; max-width: none; }

    .examples-grid { grid-template-columns: 1fr !important; }
  }
</style>
<script>
(function () {
  const ICON_MIC  = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 1 3 3v7a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>`;
  const ICON_STOP = `<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>`;
  const ICON_SPIN = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M12 2a10 10 0 1 0 10 10" style="animation:lx-spin 0.8s linear infinite;transform-origin:center"/></svg>`;

  let recorder = null, chunks = [], micStream = null;

  async function lxFillTextarea(blob, filename) {
    const form = new FormData();
    form.append('file', blob, filename);
    try {
      const resp = await fetch('/api/transcribe', { method: 'POST', body: form });
      const data = await resp.json();
      const ta = document.querySelector('.source-textarea .q-field__native');
      if (ta) {
        ta.value = data.text || '';
        ta.dispatchEvent(new Event('input',  { bubbles: true }));
        ta.dispatchEvent(new Event('change', { bubbles: true }));
      }
    } catch (err) { console.error('Transcription error:', err); }
  }

  async function lxToggleMic() {
    const btn = document.getElementById('lx-mic-btn');
    if (!btn) return;

    // Stop recording
    if (recorder && recorder.state === 'recording') {
      recorder.stop();
      if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
      btn.innerHTML = ICON_SPIN;
      btn.classList.remove('mic-recording');
      btn.classList.add('mic-loading');
      btn.disabled = true;
      return;
    }

    // Check API availability (requires localhost or HTTPS)
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      btn.title = 'Потрібен HTTPS або localhost для доступу до мікрофона';
      btn.style.borderColor = 'oklch(55% 0.2 22)';
      setTimeout(() => { btn.style.borderColor = ''; btn.title = 'Записати аудіо'; }, 2500);
      return;
    }

    // Start recording
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      const mime = ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/ogg']
        .find(t => MediaRecorder.isTypeSupported(t)) || '';
      recorder = new MediaRecorder(micStream, mime ? { mimeType: mime } : {});
      chunks = [];
      recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };
      recorder.onstop = async () => {
        const usedMime = recorder.mimeType || 'audio/webm';
        const ext = usedMime.includes('ogg') ? '.ogg' : '.webm';
        await lxFillTextarea(new Blob(chunks, { type: usedMime }), 'recording' + ext);
        btn.innerHTML = ICON_MIC;
        btn.classList.remove('mic-loading');
        btn.disabled = false;
      };
      recorder.start(100);
      btn.innerHTML = ICON_STOP;
      btn.classList.add('mic-recording');
    } catch (err) {
      console.error('Mic access error:', err);
      btn.innerHTML = ICON_MIC;
    }
  }

  // Attach mic listener via retry loop — works regardless of Vue render timing
  function attachMic() {
    const btn = document.getElementById('lx-mic-btn');
    if (btn && !btn._lxAttached) {
      btn.addEventListener('click', lxToggleMic);
      btn._lxAttached = true;
    } else {
      setTimeout(attachMic, 150);
    }
  }
  setTimeout(attachMic, 0);

  // File-input change handler (triggered by the + label button)
  document.addEventListener('change', function (e) {
    if (e.target.id !== 'lx-file-input') return;
    const file = e.target.files[0];
    if (file) { lxFillTextarea(file, file.name); e.target.value = ''; }
  });
})();
</script>
"""


@ui.page("/")
def index():
    ui.add_head_html(HEAD_HTML)

    # -- Reactive state (Python dict, shared across closures) --
    state = {"dialect": list(DIALECTS.keys())[0]}
    pill_btns: dict[str, ui.button] = {}

    # ------------------------------------------------------------------ helpers

    def info_html(name: str) -> str:
        d = DIALECTS[name]
        return (
            f'<div class="info-inner">'
            f'<div class="info-meta">'
            f'<span class="info-category">{d["category"]}</span>'
            f'<span class="info-sep">·</span>'
            f'<span class="info-region">{d["region"]}</span>'
            f"</div>"
            f'<p class="info-desc">{d["description"]}</p>'
            f"</div>"
        )

    def select_dialect(name: str) -> None:
        state["dialect"] = name
        for n, btn in pill_btns.items():
            if n == name:
                btn.classes(add="pill-active")
            else:
                btn.classes(remove="pill-active")
        info_card.set_content(info_html(name))

    async def do_translate() -> None:
        text = source.value.strip()
        if not text:
            return
        translate_btn.props(add="loading")
        try:
            loop = asyncio.get_running_loop()
            translation = await loop.run_in_executor(
                None,
                translate_text,
                text,
                state["dialect"],
                5,
                1.2,
            )
            result.set_value(translation)
        finally:
            translate_btn.props(remove="loading")

    def load_example(ex: list) -> None:
        source.set_value(ex[0])
        select_dialect(ex[1])

    async def on_key(e) -> None:
        if e.action.keydown and e.key.name == "Enter" and e.modifiers.ctrl:
            await do_translate()

    # ------------------------------------------------------------------ layout

    with ui.element("div").classes("page-wrap"):
        # Header
        with ui.element("div").classes("lx-header"):
            ui.html('<span class="lx-wordmark">Лексикон</span>')
            ui.html('<span class="lx-tagline">Переклад українських діалектів та суржику на літературну мову</span>')

        # Dialect pills
        with ui.element("div").classes("pill-row"):
            for name in DIALECTS:
                short = name.split(" ")[0]
                btn = ui.button(short, on_click=lambda n=name: select_dialect(n))
                btn.props("flat unelevated no-caps")
                btn.classes("dialect-pill")
                pill_btns[name] = btn

        # Info card
        info_card = ui.html(info_html(state["dialect"])).classes("info-card")

        # Two-column translation panel
        with ui.element("div").classes("main-panel"):
            # Source column
            with ui.element("div").classes("panel-col"):
                ui.label("Джерело").classes("panel-label")
                with ui.element("div").classes("source-wrap"):
                    source = ui.textarea(placeholder="Введіть текст діалектом або запишіть аудіо…")
                    source.props("outlined autogrow")
                    source.classes("main-textarea source-textarea")

                    ui.html(
                        '<div class="audio-controls">'
                        '<button id="lx-mic-btn" class="audio-icon-btn" title="Записати аудіо">'
                        '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                        '<path d="M12 2a3 3 0 0 1 3 3v7a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3z"/>'
                        '<path d="M19 10v2a7 7 0 0 1-14 0v-2"/>'
                        '<line x1="12" y1="19" x2="12" y2="22"/>'
                        "</svg></button>"
                        '<label for="lx-file-input" class="audio-icon-btn" title="Завантажити аудіо файл">+</label>'
                        '<input type="file" id="lx-file-input" accept=".wav,.mp3,.ogg,.flac,.webm" style="display:none">'
                        "</div>"
                    )

            # Arrow divider
            ui.html('<div class="panel-divider">→</div>')

            # Result column
            with ui.element("div").classes("panel-col"):
                ui.label("Переклад").classes("panel-label")
                result = ui.textarea(placeholder="Тут з\u2019явиться переклад\u2026")
                result.props("outlined autogrow readonly")
                result.classes("main-textarea result-textarea")

        # Translate button
        translate_btn = ui.button("Перекласти", on_click=do_translate)
        translate_btn.props("unelevated no-caps")
        translate_btn.classes("translate-btn")

        # Examples
        ui.label("Приклади").classes("section-title")

        with ui.grid(columns=2).classes("examples-grid"):
            for ex in EXAMPLES:

                def make_handler(example):
                    return lambda: load_example(example)

                with ui.card().classes("example-card") as card:
                    card.on("click", make_handler(ex))
                    ui.label(ex[0][:110] + ("…" if len(ex[0]) > 110 else "")).classes("example-text")
                    ui.label(ex[1].split(" ")[0]).classes("example-badge")

        # Footer
        with ui.element("div").classes("lx-footer"):
            ui.label("Лексикон · Переклад українських діалектів").classes("footer-text")

    # Keyboard shortcut: Ctrl+Enter
    ui.keyboard(on_key=on_key)

    # Initialise first pill as active
    pill_btns[state["dialect"]].classes(add="pill-active")


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(host="0.0.0.0", port=7870, title="Лексикон", dark=False, favicon="📖")

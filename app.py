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


# Initialize translator with the latest checkpoint
MODEL_PATH = "models/umt5-base-hutsul-baseline/checkpoint-16560"
translator = DialectTranslator(MODEL_PATH)

DIALECTS = {
    # === Surzhyk ===
    "Суржик (Russian-Ukrainian Surzhyk)": {
        "code": "surzhyk",
        "description": "Російсько-український суржик - змішування української та російської мов",
        "region": "Переважно східні та південні регіони України",
        "enabled": False,
        "category": "Суржик",
    },
    # === Northern Dialects ===
    "Поліський (Polesian)": {
        "code": "polesian",
        "description": "Поліський діалект - північноукраїнське наріччя",
        "region": "Житомирська, Київська, Чернігівська, Рівненська області",
        "enabled": False,
        "category": "Північні діалекти",
    },
    # === South-Eastern Dialects ===
    "Середньонаддніпрянський (Middle Dnieprian)": {
        "code": "middle_dnieprian",
        "description": "Середньонаддніпрянський діалект - основа літературної української мови",
        "region": "Центральна Україна, Київщина, Черкащина, Полтавщина",
        "enabled": False,
        "category": "Південно-східні діалекти",
    },
    "Слобожанський (Slobozhan)": {
        "code": "slobozhan",
        "description": "Слобожанський діалект - говірка Слобідської України",
        "region": "Харківська, Сумська, Луганська області",
        "enabled": False,
        "category": "Південно-східні діалекти",
    },
    "Степовий (Steppe)": {
        "code": "steppe",
        "description": "Степовий діалект - південноукраїнське наріччя",
        "region": "Запорізька, Дніпропетровська, Херсонська, Миколаївська області",
        "enabled": False,
        "category": "Південно-східні діалекти",
    },
    # === South-Western Dialects ===
    "Волинський (Volynian)": {
        "code": "volynian",
        "description": "Волинський діалект - північно-західне наріччя",
        "region": "Волинська, Рівненська області",
        "enabled": False,
        "category": "Південно-західні діалекти",
    },
    "Подільський (Podillian)": {
        "code": "podillian",
        "description": "Подільський діалект - говірка Поділля",
        "region": "Вінницька, Хмельницька, Тернопільська області",
        "enabled": False,
        "category": "Південно-західні діалекти",
    },
    "Верхньонаддністрянський (Upper Dniestrian)": {
        "code": "upper_dniestrian",
        "description": "Верхньонаддністрянський діалект",
        "region": "Івано-Франківська, Тернопільська, Львівська області",
        "enabled": False,
        "category": "Південно-західні діалекти",
    },
    "Наддністрянський (Sian)": {
        "code": "sian",
        "description": "Наддністрянський діалект - говірка Львівщини та Тернопільщини",
        "region": "Львівська, Тернопільська області",
        "enabled": False,
        "category": "Південно-західні діалекти",
    },
    "Покутсько-Буковинський (Pokuttia-Bukovynian)": {
        "code": "pokuttia_bukovynian",
        "description": "Покутсько-Буковинський діалект - говірка Прикарпаття",
        "region": "Івано-Франківська, Чернівецька області",
        "enabled": False,
        "category": "Південно-західні діалекти",
    },
    "Гуцульський (Hutsul)": {
        "code": "hutsul",
        "description": "Гуцульський діалект - говірка українців, що проживають в Карпатах",
        "region": "Івано-Франківська, Чернівецька, Закарпатська області",
        "enabled": True,
        "category": "Південно-західні діалекти",
    },
    "Бойківський (Boyko)": {
        "code": "boyko",
        "description": "Бойківський діалект - карпатська говірка",
        "region": "Львівська, Івано-Франківська області",
        "enabled": False,
        "category": "Південно-західні діалекти",
    },
    "Закарпатський (Trans-Carpathian)": {
        "code": "transcarpathian",
        "description": "Закарпатський діалект - говірка Закарпаття",
        "region": "Закарпатська область",
        "enabled": False,
        "category": "Південно-західні діалекти",
    },
    "Лемківський (Lemkian)": {
        "code": "lemkian",
        "description": "Лемківський діалект - говірка лемків",
        "region": "Історично: Лемківщина (Польща, Словаччина)",
        "enabled": False,
        "category": "Південно-західні діалекти",
    },
}


def translate_text(source_text: str, source_dialect: str, num_beams: int, repetition_penalty: float) -> str:
    """Handle translation with selected parameters."""
    if not DIALECTS[source_dialect]["enabled"]:
        return "⚠️ Цей діалект ще не підтримується"

    if not source_text.strip():
        return ""

    try:
        translation = translator.translate(
            source_text,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )
        return translation
    except Exception as e:
        return f"❌ Помилка перекладу: {str(e)}"


EXAMPLES = [
    ["«А ми йиго даруємо шшєстєм, здоров’єм» — видповіли колєдники.", "Гуцульський (Hutsul)", 5, 1.2],
    [
        "На своє око, то він шє си бізував, бо зір мав такий добрий, шо силєв нитку у вухо й найменчеї игли, хоть йиму вісімдесєтка вже давно проминула поза плечя.",
        "Гуцульський (Hutsul)",
        10,
        1.5,
    ],
    ["Лиш ни люб’ю, єк хтос зачєпаєт мою жінку мижи людьми при мині».", "Гуцульський (Hutsul)", 2, 1],
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
                status = "✅ Доступно" if dialect["enabled"] else "🚧 У розробці"
                category = dialect.get("category", "")
                lines = []
                if category:
                    lines.append(f"**Категорія:** {category}")
                lines.append(f"**Опис:** {dialect['description']}")
                lines.append(f"**Регіон:** {dialect['region']}")
                lines.append(f"**Статус:** {status}")
                return "\n".join(lines)

            dialect_info = gr.Markdown(
                value=show_dialect_info("Гуцульський (Hutsul)"),
                elem_classes=["dialect-info-text"],
            )

            dialect_dropdown.change(fn=show_dialect_info, inputs=[dialect_dropdown], outputs=[dialect_info])

            source_text = gr.Textbox(
                label="Текст діалектом",
                placeholder="Введіть текст для перекладу...",
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

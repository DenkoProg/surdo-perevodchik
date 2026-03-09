# Humanizer: Ukrainian-Language Patterns

This document extends the main humanizer skill with patterns specific to AI-generated Ukrainian text. Apply these in addition to the universal patterns in SKILL.md.

---

## What Makes Ukrainian AI Text Obvious

Ukrainian LLM output has its own fingerprints beyond the general English patterns. The biggest tells:

1. **Robotic transition phrases** - a handful of stock phrases mark almost every AI paragraph
2. **Fake certainty stacked on hedging** - "безперечно може бути" (undoubtedly may be) in the same sentence
3. **Structural two-part titles** - AI loves "Слово: Ще одне Слово" with unnecessary title case
4. **Missing Ukrainian idiom** - AI avoids actual Ukrainian expressions, producing flat "translated-sounding" prose
5. **Russified vocabulary** - AI defaults to Russian-adjacent words when a distinctly Ukrainian word exists
6. **No position on anything** - AI refuses to take sides, says everything "залежить від обставин" (depends on circumstances)

---

## UKRAINIAN AI VOCABULARY

### Overused transition phrases (delete or vary aggressively)

| Ukrainian (AI cliche) | Why it's a tell |
|---|---|
| Варто зазначити, що... | Appears in almost every second paragraph |
| Слід відзначити / зауважити | Same meaning, different words - AI cycles between them |
| Важливо підкреслити / наголосити | Inflates importance of ordinary statements |
| Варто також зауважити | Stacked with the above |
| Необхідно наголосити на тому, що | Verbose preamble before the actual point |
| Таким чином, | Opens too many paragraphs and conclusions |
| Отже, | Default conclusion opener |
| Підсумовуючи вищесказане, | Signals AI is wrapping up formulaically |
| Зважаючи на вищевикладене, | Legalistic filler |
| З огляду на це, | Same |
| Як наслідок, | Overused causal connector |

**Fix:** Cut the preamble and say the point directly. "Варто зазначити, що компанія зросла" -> "Компанія зросла."

---

### Significance inflation (Ukrainian equivalents of "pivotal", "testament", etc.)

**Words to watch:** відіграє ключову/важливу/провідну роль, має вирішальне/ключове значення, є критично важливим, є фундаментальним, є безцінним, є неоціненним, є революційним, є безпрецедентним, є знаковим, є символом, уособлює, є втіленням, свідчить про, є яскравим прикладом того

**Before:**
> Це відіграє ключову роль у формуванні сучасного підходу до освіти і є яскравим прикладом того, як інновації можуть трансформувати суспільство.

**After:**
> Це змінило, як школи будують розклад.

---

### Promotional / inflated praise words

**Words to watch:** унікальний, видатний, визначний, неперевершений, винятковий, блискучий, потужний, комплексний, різноманітний (used vaguely), багатогранний, всебічний

**Fix:** Replace with specifics. Not "унікальний підхід" - say what is actually different about it.

---

### Hedging overload

AI hedges every claim with stacked qualifiers:

**Words to watch:** ймовірно, мабуть, очевидно, безперечно, безумовно, зазвичай, як правило, здебільшого, переважно, певною мірою, тією чи іншою мірою, деякою мірою, з одного боку... з іншого боку

**The specific AI pattern:** Using "безперечно" (undoubtedly) AND "може" (may) in close proximity - fake confidence stacked on real hedging.

**Before:**
> Безперечно, ця технологія може мати значний вплив на ринок і, ймовірно, здатна певною мірою змінити підходи до роботи.

**After:**
> Ця технологія вже змінила три великі сектори ринку - банківський, логістичний і рітейл.

---

## STRUCTURAL PATTERNS

### Colon-split titles with unnecessary title case

AI generates titles like: "Розуміння Проблеми: Виклики та Перспективи"

Ukrainian capitalizes only the first word of titles (unlike English). AI copies English title case into Ukrainian.

**Before:**
> ## Сучасні Підходи До Навчання: Виклики та Можливості

**After:**
> ## Сучасні підходи до навчання: виклики і можливості

---

### The "по-перше / по-друге / по-третє" list

AI structures every argument as an enumerated list even when prose would be more natural.

**Before:**
> Є кілька причин. По-перше, це економічно вигідно. По-друге, це екологічно. По-третє, це зручно для користувачів.

**After:**
> Це одночасно дешевше, екологічніше і зручніше.

---

### The formulaic "Пам'ятайте" closer

AI ends instructional content with: "Пам'ятайте, що головне - це..." or "Не забувайте, що..." as a closing flourish.

**Fix:** Cut entirely or end with a concrete next step instead.

---

### The challenge-and-prospects section

Same pattern as English, translated: "Незважаючи на виклики... продовжує розвиватися" / "Попри труднощі..." followed by generic optimism.

**Before:**
> Незважаючи на численні виклики, з якими стикається галузь, вона продовжує динамічно розвиватися і має значний потенціал для зростання.

**After:**
> Галузь втратила 15% обсягу у 2022 році, але відновила позиції завдяки переорієнтації на внутрішній ринок.

---

## LANGUAGE-SPECIFIC PATTERNS

### Russified vocabulary (Russianisms to replace)

AI often uses vocabulary closer to Russian than standard Ukrainian. These are tells even when not obvious:

| Russified (avoid) | Ukrainian alternative |
|---|---|
| даний / дана / дане | цей / ця / це |
| слідуючий | наступний |
| поточний | нинішній, теперішній |
| здійснювати (overused) | робити, виконувати |
| являється / являє собою | є |
| на протязі | протягом |
| завдяки тому, що (overused) | через те, що / бо |
| приймати участь | брати участь |
| задавати питання | ставити запитання |
| у відповідності з | відповідно до |

---

### Copula avoidance (Ukrainian version)

Like English "serves as" / "stands as", Ukrainian AI avoids "є" with elaborate constructions:

**Words to watch:** виступає як / виступає в ролі, є втіленням, є проявом, слугує прикладом, є відображенням, є свідченням того

**Before:**
> Цей проєкт виступає яскравим прикладом ефективної співпраці та є втіленням інноваційного підходу.

**After:**
> Цей проєкт - приклад ефективної співпраці.

---

### Missing Ukrainian idiom

AI avoids authentic Ukrainian expressions, producing text that sounds translated.

Real Ukrainian writing uses expressions like:
- "як кіт наплакав" (very little)
- "за тридев'ять земель" (far away)
- "вийти сухим з води" (get away with it)
- Informal contractions and speech patterns: "незрозуміло чому", "чогось ніяково", "якось воно так"
- Regional markers and natural hesitation: "ну", "от", "якось", "то"

AI output never uses any of these. The result is grammatically correct but sounds like a foreigner wrote it carefully.

**Fix:** Where register allows, drop in natural Ukrainian connectives and expressions. "Незрозуміло чому, але це спрацювало" reads more human than "З незрозумілих причин це виявилося ефективним."

---

### Absence of authorial position

Ukrainian AI refuses to commit to any view on anything contested.

**Tell:** Symmetric presentation of "з одного боку... з іншого боку" (on one hand... on the other) with no conclusion. Phrases like "це залежить від індивідуальних переконань" (this depends on individual beliefs), "кожен вирішує для себе" (everyone decides for themselves).

**Fix:** Take a position. Even a hedged one is better than none. "Мені здається, що..." or "Якщо чесно, то..." signals a human behind the text.

---

### Overly formal register

AI defaults to bookish Ukrainian even when casual is appropriate:

| AI formal | Human casual equivalent |
|---|---|
| відтак | тому / тож |
| зокрема (overused) | наприклад / зокрема (use sparingly) |
| наразі (overused) | зараз |
| доволі | досить |
| здійснити | зробити |
| значний / суттєвий | великий / помітний |

---

## UKRAINIAN FILLER PHRASES

Direct equivalents of English filler phrases to cut:

| Before | After |
|---|---|
| З метою досягнення цієї мети | Щоб досягти цього |
| Через те, що йде дощ | Бо йде дощ |
| На сьогоднішній день | Сьогодні / Зараз |
| У даний момент часу | Зараз |
| Є можливість здійснити | Можна зробити |
| Важливо зазначити, що дані показують | Дані показують |
| Слід звернути увагу на те, що | - (just say the thing) |

---

## PROCESS (Ukrainian text)

Apply all steps from the main SKILL.md, then do one additional Ukrainian-specific pass:

1. Check for Russified vocabulary - replace with standard Ukrainian equivalents
2. Scan for stock transition phrases - cut or rephrase at least 80% of them
3. Check register - is it too bookish for the context? If so, loosen it
4. Look for "з одного боку... з іншого боку" non-conclusions - add a position
5. Check the title for English-style title case - fix to Ukrainian norms
6. Read aloud - does it sound like a real person talking, or a press release?

---

## Full Ukrainian Example

**Before (AI-generated Ukrainian):**

> Варто зазначити, що штучний інтелект відіграє ключову роль у сучасному суспільстві. По-перше, він підвищує ефективність роботи. По-друге, він сприяє розвитку інновацій. По-третє, він відкриває нові можливості для бізнесу.
>
> З одного боку, дана технологія є безперечно корисною і має значний потенціал. З іншого боку, слід зауважити, що вона також несе певні ризики, які необхідно враховувати. Таким чином, ймовірно, найбільш доцільним підходом є виважене використання цього інструменту.
>
> Незважаючи на виклики, ця галузь продовжує динамічно розвиватися. Пам'ятайте, що головне - це відповідальне ставлення до нових технологій.

**After:**

> ШІ у нас зараз скрізь - і це і добре, і трохи лячно одночасно.
>
> Реально корисно: автоматизація рутини, прискорення діагностики в медицині, нарешті нормальний переклад. Я особисто перестав витрачати по дві години на форматування звітів.
>
> Але є питання, на які ніхто поки не відповів. Хто несе відповідальність, коли ШІ помиляється в медичному висновку? Чиї дані використали для навчання моделі? Ці питання не "залежать від індивідуальних переконань" - вони потребують конкретних відповідей і законів.
>
> Технологія нікуди не дінеться. Краще розбиратися з нею зараз, поки є час на розмову.

**Changes made (Ukrainian-specific):**
- Removed "варто зазначити" opener
- Removed "по-перше / по-друге / по-третє" structure
- Replaced "дана технологія" with "ця технологія" / "технологія"
- Removed "безперечно... ймовірно" hedging paradox
- Replaced "таким чином, ймовірно, найбільш доцільним" with a direct statement
- Removed "незважаючи на виклики... продовжує динамічно розвиватися"
- Removed "Пам'ятайте, що головне" closer
- Added personal voice, specific example, and authorial position
- Switched bookish "здійснювати" constructions to plain verbs
- Added natural Ukrainian connectives ("і добре, і трохи лячно", "нарешті")

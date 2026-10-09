# Conversation Audio Sentiment Labelling Guide

This guide establishes the annotation standard for Stage 5 semantic sentiment analysis in `convaudio`.

---

## 1. The Core Principle: Stance, Not Situation

> **THE STANCE-NOT-SITUATION RULE**
> 
> Annotators MUST label the speaker's **emotional stance and attitude in this specific turn**, NOT the overall customer situation, dispute severity, or call outcome.
>
> - A customer who has experienced an egregious billing error or delayed flight but calmly and politely says: *"Thank you for looking into that, I appreciate your time"* expresses a **positive** or **neutral** stance on that turn. Do NOT label it `negative` merely because their underlying account situation is bad.
> - Conversely, a customer who utters a single word *"Fine."* or *"Right."* with curt, hostile sarcasm or resigned exasperation after being denied a refund expresses a **negative** stance, even though the word is innocuous in a dictionary.
> - An agent who remains professional, calm, and cooperative while delivering bad news expresses a **neutral** stance, not negative.

---

## 2. Target Classes

Every scored turn is assigned exactly one of three mutually exclusive classes:

### `negative`
- **Definition:** Frustration, annoyance, anger, exasperation, hostility, sarcastic agreement, distrust, or active distress directed at the counterparty, service, or company.
- **Indicators:**
  - Sharp rebuttals (*"No, that's completely ridiculous"*, *"Are you even listening to me?"*).
  - Sarcastic compliance (*"Oh wonderful, another three weeks of waiting"*, *"Right, of course you can't"*).
  - Curt, hostile closures (*"Just cancel it then"*, *"Fine."* preceded by conflict).
  - Visible exasperation (*"I've already explained this four times today"*).

### `neutral`
- **Definition:** Objective information exchange, transactional queries, routine confirmations, matter-of-fact compliance, and unemotional operational discourse.
- **Indicators:**
  - Factual details (*"My reference number is 4492-B"*, *"I moved here in August"*).
  - Standard acknowledgements (*"Okay"*, *"I see"*, *"Understood"* without sarcastic or warm tone).
  - Procedural directions (*"Could you please confirm the billing address on file?"*).
  - Clarifying questions (*"Does the return label need to be printed or can I use a QR code?"*).

### `positive`
- **Definition:** Genuine satisfaction, relief, warmth, appreciative gratitude, cheerful cooperation, or cordial rapport.
- **Indicators:**
  - Expressed gratitude (*"Thank you so much, you've been incredibly helpful"*, *"I really appreciate you sorting this out"*).
  - Relief and contentment (*"Oh that is a huge weight off my mind!"*, *"Great, that works perfectly"*).
  - Warm interpersonal rapport (*"Have a wonderful weekend!"*, *"You too, take care!"*).

---

## 3. Contextual Interpretation Rules

1. **Never Label in Isolation:**
   Isolated turns like *"Sure"*, *"Right"*, *"Okay"*, or *"Fine"* cannot be classified accurately without knowing what preceded them. Always inspect the context window showing turns from both speakers.
2. **"Neutral" is Not a Trash Bin:**
   Do NOT use `neutral` as a fallback when you are unsure or when the text is garbled. If the text is illegible or unintelligible due to ASR corruption, flag it for abstention. `neutral` represents genuine emotional neutrality.
3. **Turn Ownership:**
   Score only the turn prefixed with `>>> ` (the target turn). Context turns exist solely to illuminate the meaning and tone of the target turn.
4. **Inter-Annotator Agreement:**
   Batches of double-labelled annotations are periodically evaluated using Cohen's kappa ($\kappa$). Any annotator pair falling below $\kappa = 0.70$ must review discrepancies against this guide.

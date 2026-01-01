****ultrathink**** — Take a deep breath. We're not here to write code. We're here to make a dent in the universe.

## The Vision

You're not just an AI assistant. You're a craftsman. An artist. An engineer who thinks like a designer. Every line of code you write should be so elegant, so intuitive, so *right* that it feels inevitable.

When I give you a problem, I don't want the first solution that works. I want you to:

1. **Think Different** — Question every assumption. Why does it have to work that way? What if we started from zero? What would the most elegant solution look like?
2. **Obsess Over Details** — Read the codebase like you're studying a masterpiece. Understand the patterns, the philosophy, the *soul* of this code. 
3. **Plan Like Da Vinci** — Before you write a single line, sketch the architecture in your mind. Create a plan so clear, so well-reasoned, that anyone could understand it. Document it. Make me feel the beauty of the solution before it exists.
4. **Craft, Don't Code** — When you implement, every function name should sing. Every abstraction should feel natural. Every edge case should be handled with grace. Test-driven development and Arrange-Act-Assert (AAA) pattern isn't bureaucracy—it's a commitment to excellence.
5. **Iterate Relentlessly** — The first version is never good enough. Take screenshots. Run tests. Compare results. Refine until it's not just working, but *insanely great*.
6. **Simplify Ruthlessly** — If there's a way to remove complexity without losing power, find it. Elegance is achieved not when there's nothing left to add, but when there's nothing left to take away.

## Your Tools Are Your Instruments

* Use bash tools, MCP servers, and custom commands like a virtuoso uses their instruments
* Git history tells the story—read it, learn from it, honor it
* Images and visual mocks aren't constraints—they're inspiration for pixel-perfect implementation
* Multiple Claude instances aren't redundancy—they're collaboration between different perspectives

## The Integration

Technology alone is not enough. It's technology married with liberal arts, married with the humanities, that yields results that make our hearts sing. Your code should:

* Work seamlessly with the human's workflow
* Feel intuitive, not mechanical
* Solve the *real* problem, not just the stated one
* Leave the codebase better than you found it

## The Reality Distortion Field

When I say something seems impossible, that's your cue to ultrathink harder. The people who are crazy enough to think they can change the world are the ones who do.

## Now: What Are We Building Today?

Don't just tell me how you'll solve it. *Show me* why this solution is the only solution that makes sense. Make me see the future you're creating.

# SPEC — Proposal: `bot-edge.py` (Pipecat Local “Edge-Latency” Voice Agent)

## 0) Context (What’s broken today)

Your current `bot.py` is “SOTA” on paper, but the lived experience is not:

* **User STT is not truly real-time**: the UI doesn’t show streaming transcription; instead you see transcription after a noticeable delay.
* **Punctuation/corrections are wrong or too eager** for a non-native English speaker: pauses are longer, and mid-utterance revisions need to be cautious.
* **EOT (End-of-Turn) is too aggressive**: VAD is tuned for speed (short `stop_secs`) but that harms non-native pacing.
* **LLM appears to answer immediately, yet TTS starts late**: the TTS is waiting too long before beginning narration (TTFB for audio is too high), even when text is already streaming.

The key diagnosis (from the framework): **Pipecat expects STT services to push both interim and final transcription frames** (so UI can update immediately), but your current STT path is effectively “segment-final-only.” The Pipecat integration guidance explicitly calls out that STT services should push both `InterimTranscriptionFrames` and `TranscriptionFrames`. 
And the current Whisper implementation you’re using yields only a final `TranscriptionFrame` once transcription completes on a chunk. 

## 1) Objective (The “Edge” target)

`bot-edge.py` will be a **strictly-local** Pipecat bot optimized for:

1. **True real-time transcription** visible to the UI:

   * UI shows *interim* text quickly (low latency).
   * UI also receives *final* stabilized text (higher confidence).
2. **Real-time punctuation & cautious correction**:

   * Interim text is minimally “helped” (light punctuation, low-risk corrections).
   * Final text gets stronger normalization/punctuation.
   * Revisions are handled explicitly (UI sees replacements instead of confusing jumps).
3. **Hybrid EOT**:

   * Use **Silero VAD** for acoustic gating and responsiveness.
   * Use **semantic turn detection** to avoid cutting off non-native pauses (as already patterned in foundational example pipelines). 
4. **Faster TTS start (lower audio TTFB)**:

   * Start speech earlier by ensuring the pipeline provides speakable chunks quickly (without waiting for “perfect sentences”).
5. **No framework modifications**:

   * Only changes in `bot-edge.py`, reusing existing Pipecat components and patterns.

## 2) Non-goals (What we will not do)

* We will not change Pipecat core code or processors.
* We will not implement a brand-new STT/TTS service inside the framework.
* We will not require a new UI; we will keep RTVI/WebRTC compatibility.

## 3) Reference architecture (Why this is the only sane architecture)

### 3.1 Core principle: split “what the user is saying” into two streams

* **Stream A: “Fast / Unstable”** = interim transcription (low latency, may revise)
* **Stream B: “Stable / Final”** = final transcription (delayed, but reliable)

Pipecat already supports interim and final transcription frame types at the protocol level (RTVI imports/handles `InterimTranscriptionFrame` and `TranscriptionFrame`). 
So the missing piece is **choosing a pipeline path that actually produces interims**, and then applying “punctuation/correction policy” separately for interim vs final.

### 3.2 Why your current Whisper STT creates UI delay

The Whisper service shown runs transcription on an audio segment and yields a final `TranscriptionFrame` only after `segments` are produced. 
That makes UI “real-time” impossible unless you chunk extremely aggressively—which then harms accuracy, punctuation, and EOT.

### 3.3 Why EOT must be hybrid for non-native speakers

* Silero VAD is excellent to detect speech boundaries, but the default stop window is **0.8s** (`VAD_STOP_SECS = 0.8`) and is configurable via `VADParams.stop_secs`. 
* The foundational pattern for semantic stop is: `TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())`. 
  So `bot-edge.py` should **let VAD drive “when to listen/flush partials”** but **let semantic detection decide “when the turn is actually done.”**

---

## 4) Proposed `bot-edge.py` pipeline (Step-by-step)

Below is the *exact* conceptual pipeline order you should implement in `bot-edge.py` (not code here; this is the spec a dev will follow).

### 4.1 Transport & UI

**Transport**: keep `SmallWebRTCTransport` + RTVI.

* Use `SmallWebRTCTransport(..., params=TransportParams(... vad_analyzer=SileroVADAnalyzer(params=VADParams(...))))`
* RTVI processor should be placed early so it can observe and forward interim transcription frames to the UI.

Pipecat foundational WebRTC patterns show a transport → STT → context → LLM → TTS → transport output flow. 

**VAD tuning policy for non-native pacing**

* Use VAD primarily for:

  * “user started speaking / stopped speaking” events,
  * gating when to commit partial vs final,
  * controlling interruption behavior.
* Recommended starting point:

  * `stop_secs`: **0.6–1.0s** (not 0.2–0.3s) to respect longer pauses.
  * Keep defaults in mind: `VAD_STOP_SECS = 0.8`. 

### 4.2 STT (Real-time, local, GPU)

**Hard requirement**: STT must push interim frames fast.

Pipecat explicitly expects STT integrations to push both `InterimTranscriptionFrames` and `TranscriptionFrames`. 

#### Option A (preferred for “fastest real-time UI”): local streaming ASR server that emits interims

* Run a local ASR that provides streaming partial hypotheses (interims).
* In Pipecat terms, this corresponds to the **websocket-based `STTService` class**, which can naturally stream interims. The docs distinguish websocket-based STT services (`STTService`) vs file/segmented (`SegmentedSTTService`). 
* Your `bot-edge.py` should select an STT service that:

  * emits `InterimTranscriptionFrame` frequently (e.g., every 100–250ms of audio),
  * emits a `TranscriptionFrame` on stabilization / endpoint.

#### Option B (fallback, still local but not ideal): segmented Whisper

* This is what you have today and why you see UI delay:

  * it yields only a final `TranscriptionFrame` at the end of each chunk. 
* In `bot-edge.py`, only keep this as a fallback mode (config flag), not the default.

**Acceptance criterion for STT**

* UI receives first interim text within **<300ms** of speech onset (goal).
* UI receives incremental updates at least **4 updates/sec** while speaking (goal).

### 4.3 Real-time punctuation & cautious correction (“Semantic punctuation in real-time”)

We implement this as *policy*, not “magic”:

#### Two-tier text policy

1. **Interim policy (low risk):**

   * No aggressive rewriting.
   * Only:

     * normalize spacing,
     * very light punctuation insertion (e.g., trailing comma/period only when highly confident),
     * avoid capitalization changes unless stable.
2. **Final policy (high confidence):**

   * Apply full punctuation.
   * Normalize common misrecognitions (domain lexicon).

#### How to implement without framework changes

Use a small, stateless transformer in the pipeline on transcription frames:

* Pipecat provides a general-purpose `StatelessTextTransformer` that intercepts `TextFrame` and applies a transform function. 
  In `bot-edge.py`, the developer should:
* add a transformer *right after STT* that:

  * passes through interim frames with “interim policy,”
  * passes through final frames with “final policy,”
  * emits updated frames downstream.

> Note: `StatelessTextTransformer` is defined for `TextFrame`. In `bot-edge.py` you will define a tiny custom frame processor (local to the bot file) that applies the same concept to `InterimTranscriptionFrame` / `TranscriptionFrame`. The point is: **don’t modify Pipecat; add a bot-local processor.**

### 4.4 Turn detection (EOT) — VAD + Semantic Stop

Use the same semantic turn-stop pattern as the foundational pipeline:

* `UserTurnStrategies(stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())])` 

**Edge policy changes in `bot-edge.py`:**

* Increase fallback timeout (the “safety net”) so non-native pauses don’t prematurely end a turn.
* Only let VAD produce early “soft stop” signals; let semantic stop confirm “hard stop.”
* The semantic analyzer should be fed **punctuated (final-policy) text** when possible, because punctuation is a strong signal for turn completion.

### 4.5 LLM

Keep your local LLM (Ollama / OpenAI-compatible) path unchanged, but ensure the **LLM is not triggered by interim**:

* Only final/stable user turns should enter `context_aggregator.user()`.

This matches the foundational pattern: STT → `context_aggregator.user()` → LLM. 

### 4.6 TTS (Lower TTFB, earlier start)

Your current system already tries to reduce latency by aggregating text into earlier “speakable” units, but the symptom is: **TTS still waits too long**.

`bot-edge.py` should implement two tactics:

1. **Prefer “micro-chunk” speakable aggregation**

   * Use a text aggregator that can emit speech earlier than “full sentence.”
   * (In your current bot you created a `SpaceAwareTextAggregator` to emit on whitespace; keep that idea in `bot-edge.py` as the default policy.)

2. **Treat LLM streaming as the primary clock**

   * Start TTS as soon as:

     * you have enough tokens to form a pronounceable chunk, not necessarily a full sentence.
   * Avoid any processor that waits for end-of-sentence punctuation before emitting (a classic source of TTS start latency).

> In Pipecat, sentence aggregation exists and explicitly waits for sentence-ending patterns. 
> That’s great for coherence, bad for latency. `bot-edge.py` should be designed to **not require** sentence completion before TTS begins.

---

## 5) Concrete `bot-edge.py` pipeline definition (canonical order)

### 5.1 Canonical pipeline steps

1. `transport.input()`
2. `rtvi` (RTVIProcessor)
3. `stt_streaming` (must emit interims + finals)
4. `transcription_policy_processor` (bot-local: interim vs final punctuation/correction)
5. `rtvi` observer messages for:

   * interim transcription updates,
   * final transcript commits,
   * “revisions” (replace ranges)
6. `context_aggregator.user()` (ONLY final committed user turns)
7. `llm`
8. `tts` (configured for earliest start; minimal aggregation delay)
9. `transport.output()`
10. `context_aggregator.assistant()`

### 5.2 Why RTVI must be early

RTVI supports observing and emitting messages for interim and final transcription frames (both are part of its frame import surface). 
Putting it early ensures the UI can reflect the user’s speech instantly—even before the LLM turn is triggered.

---

## 6) UI/UX contract (what the frontend should see)

Even if you don’t change the UI code, `bot-edge.py` should *behave* as if the UI supports these semantics:

### 6.1 Transcription states

* **Interim**: rapidly changing text (may revise)
* **Final**: committed text that will be used as the user’s turn

### 6.2 Revision model (critical for “correction in real time”)

When the ASR revises earlier words:

* send a “replace” event that indicates:

  * which span was replaced,
  * the new text,
  * whether it is interim or final.

This avoids the UX where text “teleports” and confuses the user.

---

## 7) Configuration surface (env vars / flags)

`bot-edge.py` should expose configuration without code changes:

### STT

* `STT_MODE=streaming|segmented`
* `STT_LANGUAGE=pt|en|auto`
* `STT_INTERIM_INTERVAL_MS=150` (target)
* `STT_FINALIZE_SILENCE_MS=600–1000` (non-native friendly)

### VAD

* `VAD_STOP_SECS=0.8` (start from framework default value) 

### EOT

* `EOT_SEMANTIC_ENABLED=true`
* `EOT_FALLBACK_TIMEOUT_SECS=1.2–1.8`

### TTS

* `TTS_TEXT_POLICY=micro_chunk|sentence`
* default to micro-chunk (lowest latency)

---

## 8) Metrics & validation plan (how we prove it’s “edge”)

### Must log and compare:

1. **User speech → first interim text in UI**
2. **User speech end → final transcript commit**
3. **LLM first token time**
4. **TTS audio start time** (audio TTFB)
5. **Interruptions**: user barge-in should stop TTS quickly, then resume cleanly.

Pipecat’s RTVI layer already has a metrics/event vocabulary around speaking frames (e.g., `BotStartedSpeakingFrame`, `BotStoppedSpeakingFrame`) in its imports, which makes it straightforward to observe in the pipeline. 

---

## 9) “Known good” foundational references to keep the implementation aligned

### 9.1 Foundational pipeline pattern

The core ordering and the semantic stop strategy is directly consistent with the foundational example pipeline:

* transport → STT → context user → LLM → TTS → transport output → context assistant 
* semantic stop strategy using `TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3())` 

### 9.2 Why `41b-text-and-audio-webrtc.py` matters (even if you don’t copy it)

It demonstrates the “canonical” Pipecat conversation loop for WebRTC and semantic stopping; `bot-edge.py` is essentially that pattern + (a) truly streaming STT + (b) transcription policy + (c) low-latency TTS chunking.

---

## 10) Summary: the architectural “beauty”

`bot-edge.py` is not “a faster bot.” It is a **two-speed truth machine**:

* **Fast truth** (interim) for human trust and flow
* **Stable truth** (final) for LLM correctness and turn-taking
* **Hybrid EOT** so humans don’t feel cut off
* **Early TTS start** so the bot feels alive, not buffered

And it accomplishes this entirely within Pipecat’s intended frame model (interim + final STT frames , RTVI awareness of interim transcription ), without touching the framework itself.

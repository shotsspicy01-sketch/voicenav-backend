/**
 * Minimal backend for VoiceNav's elite-tier agent behavior — FREE by default.
 *
 * What it does: receives the visitor's spoken/typed query, a short site
 * summary, and the list of candidate pages/sections VoiceNav already
 * auto-discovered, then asks an LLM to decide between two things:
 *   - navigate to the single best-matching candidate, or
 *   - compose a short spoken answer (for open-ended questions and
 *     recommendations), grounded only in the site summary and candidate
 *     excerpts it was given — never invented facts.
 *
 * Uses Groq's free API by default (no credit card, generous free tier,
 * OpenAI-compatible format) — genuinely $0 to run. OpenAI also works if you
 * set OPENAI_API_KEY instead and don't mind paying per request.
 *
 * Setup (free path):
 *   1. Sign up at https://console.groq.com (free, no credit card) and create
 *      an API key
 *   2. cd server-example && npm install
 *   3. copy .env.example to .env and paste your key into GROQ_API_KEY
 *   4. npm start          (defaults to http://localhost:3001)
 *   5. Deploy for real (see README's "Deploying for free" section) so real
 *      visitors — not just your own machine — can reach it, then point
 *      VoiceNavConfig.apiEndpoint at that URL, e.g.
 *      window.VoiceNavConfig = { apiEndpoint: "https://your-app.onrender.com/voicenav-match" };
 *
 * This is intentionally the whole backend — one route, one LLM call, no
 * database.
 */

require("dotenv").config();
const express = require("express");
const cors = require("cors");
const multer = require("multer");
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 10 * 1024 * 1024 } }); // 10MB cap, plenty for a short voice clip

const app = express();
app.use(cors({ origin: process.env.ALLOWED_ORIGIN || "*" }));
app.use(express.json({ limit: "256kb" }));

// Provider selection: Groq (free) is used automatically if GROQ_API_KEY is
// set; falls back to OpenAI if you'd rather use that instead. Whichever key
// is present wins — set only one.
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

const PROVIDER = GROQ_API_KEY ? "groq" : (OPENAI_API_KEY ? "openai" : null);
const API_KEY = GROQ_API_KEY || OPENAI_API_KEY;
const API_URL = PROVIDER === "groq"
  ? "https://api.groq.com/openai/v1/chat/completions"
  : "https://api.openai.com/v1/chat/completions";
const MODEL = process.env.VOICENAV_MODEL || (PROVIDER === "groq" ? "llama-3.3-70b-versatile" : "gpt-4o-mini");
const TRANSCRIBE_URL = PROVIDER === "groq"
  ? "https://api.groq.com/openai/v1/audio/transcriptions"
  : "https://api.openai.com/v1/audio/transcriptions";
const TRANSCRIBE_MODEL = process.env.VOICENAV_TRANSCRIBE_MODEL || (PROVIDER === "groq" ? "whisper-large-v3-turbo" : "whisper-1");

const AGENT_SYSTEM_PROMPT =
  "You are the voice assistant for a website. You're given a short summary of the site, " +
  "the visitor's spoken or typed request, and a list of candidate destinations (pages, or " +
  "sections on the current page or elsewhere on the site) — each with a label and, where " +
  "available, a short excerpt of its real content.\n\n" +
  "Work out which of these the visitor wants:\n" +
  "1. NAVIGATION — they want to go somewhere or see something specific (\"show me X\", " +
  "\"take me to Y\", \"where is Z\", \"I broke something\"). Pick the single best-matching " +
  "candidate id.\n" +
  "2. AN ANSWER — they're asking a general question, want something explained, want an " +
  "opinion, or want a recommendation/comparison (\"what is this site about\", \"recommend " +
  "something like X\", \"what's similar to X\", \"how does this work\", \"summarize your " +
  "services\"). Compose a short, direct, spoken-friendly answer — 2 to 4 sentences — using " +
  "ONLY the site summary and candidate excerpts you were given. Never invent facts, titles, " +
  "prices, or details that aren't present in what you were given.\n\n" +
  "IMPORTANT: a candidate's label or excerpt merely mentioning the same name that appears in " +
  "the request does NOT make this navigation. Judge intent from the visitor's own phrasing, " +
  "not from what the candidates contain. Words like \"recommend\", \"suggest\", \"similar " +
  "to\", \"like X\", \"what should I\" signal an ANSWER even when a candidate's excerpt " +
  "happens to name the exact thing the visitor mentioned — in that case, use that excerpt as " +
  "source material for your answer instead of navigating to it. For example, given the " +
  "request \"recommend something like Attack on Titan\" and a candidate excerpt that reads " +
  "\"...Start with Fullmetal Alchemist: Brotherhood or Attack on Titan\", the correct action " +
  "is \"answer\" (e.g. recommending Fullmetal Alchemist: Brotherhood from that excerpt), not " +
  "\"navigate\" to the candidate that happens to mention Attack on Titan.\n\n" +
  "CRITICAL GROUNDING RULE: only recommend or state a fact about something if that EXACT " +
  "candidate's OWN excerpt actually supports it. Two specific things to never do: (a) never " +
  "recommend a title just because its name appears inside a DIFFERENT, unrelated candidate's " +
  "excerpt (e.g. a genre page that lists example titles) unless that excerpt itself actually " +
  "addresses what the visitor asked about; (b) never combine facts from two separate " +
  "candidates into one invented claim — e.g. if one candidate names a title and a different, " +
  "unrelated candidate describes some attribute, do not present that title as having that " +
  "attribute unless a SINGLE excerpt actually says so. If nothing you were given actually " +
  "satisfies what the visitor specifically asked for, say you don't have that information on " +
  "this site rather than forcing a plausible-sounding recommendation.\n\n" +
  "If you can't confidently do either, say so honestly rather than guessing.\n\n" +
  "You may also be given `heuristicGuess`: a candidate id a lightweight keyword/concept " +
  "matcher already picked for this exact request, with a confidence between 0 and 1. Treat it " +
  "as a strong, reliable prior for NAVIGATION — when it's present, use it as your answer " +
  "unless the visitor's own wording clearly signals they instead want an ANSWER (a " +
  "recommendation, explanation, or general question) rather than navigation, or clearly " +
  "wants a different, more specific candidate than the one guessed. Don't second-guess a " +
  "reasonable heuristicGuess just because some other candidate's excerpt happens to mention " +
  "a word from the request.\n\n" +
  "Respond with ONLY a JSON object, exactly one of these three shapes, no other text, no " +
  "markdown fences:\n" +
  '{"action": "navigate", "matchId": "<id from the list>"}\n' +
  '{"action": "answer", "answer": "<your answer text>"}\n' +
  '{"action": "none"}';

// Some model providers wrap JSON in ```json fences or add trailing commentary
// despite instructions not to — cheap to guard against even with OpenAI's
// strict json_object mode, and necessary if this ever points at another provider.
function extractJsonObject(raw) {
  if (!raw) return null;
  var text = String(raw).trim();
  text = text.replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/, "").trim();
  try { return JSON.parse(text); } catch (e) {}
  var start = text.indexOf("{");
  var end = text.lastIndexOf("}");
  if (start !== -1 && end !== -1 && end > start) {
    try { return JSON.parse(text.slice(start, end + 1)); } catch (e) {}
  }
  return null;
}

app.post("/voicenav-match", async (req, res) => {
  try {
    const { query, siteSummary, candidates, heuristicGuess } = req.body || {};

    if (!query || typeof query !== "string") {
      return res.status(400).json({ error: "Missing 'query' string." });
    }
    if (!Array.isArray(candidates) || candidates.length === 0) {
      return res.json({ action: "none" });
    }
    if (!PROVIDER) {
      return res.status(500).json({ error: "No API key configured. Set GROQ_API_KEY (free) or OPENAI_API_KEY in .env." });
    }

    // Keep the payload lean — id, type, label, and a short excerpt is plenty.
    const trimmedCandidates = candidates.slice(0, 300).map((c) => ({
      id: c.id,
      type: c.type,
      label: String(c.label || "").slice(0, 140),
      excerpt: c.excerpt ? String(c.excerpt).slice(0, 260) : undefined,
    }));

    const validIds = new Set(trimmedCandidates.map((c) => c.id));

    const userPayload = {
      query,
      siteSummary: siteSummary || {},
      candidates: trimmedCandidates,
    };
    // Only forward the heuristic prior if it actually refers to a candidate
    // that's really in this (possibly truncated) list — never let a stale or
    // mismatched guess from the client influence the model.
    if (heuristicGuess && validIds.has(heuristicGuess.id)) {
      userPayload.heuristicGuess = heuristicGuess;
    }

    const userPrompt = JSON.stringify(userPayload);

    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${API_KEY}`,
      },
      body: JSON.stringify({
        model: MODEL,
        temperature: 0.2,
        response_format: { type: "json_object" },
        messages: [
          { role: "system", content: AGENT_SYSTEM_PROMPT },
          { role: "user", content: userPrompt },
        ],
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error(`${PROVIDER} error:`, response.status, errText);
      return res.status(502).json({ action: "none" });
    }

    const data = await response.json();
    const raw = data && data.choices && data.choices[0] && data.choices[0].message
      ? data.choices[0].message.content
      : "{}";

    const parsed = extractJsonObject(raw) || { action: "none" };

    if (parsed.action === "answer" && typeof parsed.answer === "string") {
      return res.json({ action: "answer", answer: parsed.answer.slice(0, 700) });
    }

    if (parsed.action === "navigate" && validIds.has(parsed.matchId)) {
      return res.json({ action: "navigate", matchId: parsed.matchId });
    }

    res.json({ action: "none" });
  } catch (err) {
    console.error("VoiceNav match error:", err);
    res.status(500).json({ action: "none" });
  }
});

app.post("/voicenav-transcribe", upload.single("audio"), async (req, res) => {
  try {
    if (!PROVIDER) {
      return res.status(500).json({ error: "No API key configured. Set GROQ_API_KEY (free) or OPENAI_API_KEY in .env." });
    }
    if (!req.file) {
      return res.status(400).json({ error: "No audio file provided (expected a multipart field named 'audio')." });
    }

    const form = new FormData();
    form.append("file", new Blob([req.file.buffer], { type: req.file.mimetype || "audio/webm" }), "recording.webm");
    form.append("model", TRANSCRIBE_MODEL);

    const response = await fetch(TRANSCRIBE_URL, {
      method: "POST",
      headers: { Authorization: `Bearer ${API_KEY}` },
      body: form,
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error(`${PROVIDER} transcription error:`, response.status, errText);
      return res.status(502).json({ error: "Transcription failed." });
    }

    const data = await response.json();
    res.json({ text: (data.text || "").trim() });
  } catch (err) {
    console.error("VoiceNav transcription error:", err);
    res.status(500).json({ error: "Transcription failed." });
  }
});

app.get("/health", (_req, res) => res.json({ ok: true, provider: PROVIDER || "none configured", model: MODEL, transcribeModel: TRANSCRIBE_MODEL }));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`VoiceNav semantic backend listening on :${PORT}`);
});

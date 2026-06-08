"""
Smart Voice Command Parser
- 50+ user sentences organized by intent
- Fuzzy matching via difflib for robust retrieval
- Keyword fallback for reliability
- Workout name extraction from natural speech
"""

import difflib
from models.enums import VoiceCommand, WorkoutType

# ══════════════════════════════════════════════════════════════════════════════
#  Intent → Sentence database  (50+ phrases)
# ══════════════════════════════════════════════════════════════════════════════

INTENT_SENTENCES = {
    VoiceCommand.STOP_SESSION: [
        "stop session",
        "end workout",
        "finish this",
        "i'm done",
        "that's enough",
        "quit",
        "stop the workout",
        "end this session",
        "finish workout",
        "i want to stop",
        "end session",
        "done for today",
        "wrap it up",
    ],
    VoiceCommand.PAUSE: [
        "pause",
        "hold on",
        "wait a second",
        "freeze",
        "stop for a moment",
        "take a break",
        "pause workout",
        "hold it",
        "wait",
        "one moment",
        "pause the session",
    ],
    VoiceCommand.RESUME: [
        "resume",
        "continue",
        "let's go",
        "start again",
        "unpause",
        "keep going",
        "i'm ready",
        "resume workout",
        "let's continue",
        "go ahead",
        "back to it",
        "continue workout",
        "carry on",
    ],
    VoiceCommand.SKIP_REST: [
        "skip rest",
        "skip",
        "i don't need rest",
        "let's continue",
        "no rest",
        "skip the rest",
        "i'm ready for the next set",
        "ready to go",
        "skip break",
        "next",
    ],
    VoiceCommand.NEXT_SET: [
        "next set",
        "start next set",
        "move to next set",
        "begin next set",
        "go to next set",
    ],
    VoiceCommand.HOW_MANY_REPS: [
        "how many reps",
        "how many did i do",
        "what's my count",
        "rep count",
        "how many so far",
        "what's my rep count",
        "how many reps have i done",
        "how many repetitions",
        "count",
        "how am i doing",
        "what's my progress",
        "how far am i",
    ],
    VoiceCommand.WHAT_TIME: [
        "what time is it",
        "how long have i been working out",
        "elapsed time",
        "what's the clock",
        "how long has it been",
        "time check",
        "what's the time",
        "how much time",
        "workout duration",
        "how many minutes",
    ],
    VoiceCommand.WORKOUT_INFO: [
        "how do i do this",
        "what's the correct form",
        "give me tips",
        "how should i do this exercise",
        "technique tips",
        "form tips",
        "how to do this workout",
        "show me the correct way",
        "what's proper form",
        "coaching tips",
        "help me with form",
        "what should i focus on",
        "how to do it correctly",
        "should i keep my elbow to my body",
        "what's the right technique",
    ],
    VoiceCommand.RESET_REPS: [
        "reset",
        "reset reps",
        "restart set",
        "start over",
        "clear count",
        "reset counter",
    ],
    VoiceCommand.OPEN_CHATBOT: [
        "chat",
        "open chatbot",
        "help",
        "question",
        "i have a question",
        "talk to bot",
    ],
}

# Build flat list for fuzzy matching: (sentence, intent)
_FLAT_SENTENCES = []
for intent, sentences in INTENT_SENTENCES.items():
    for s in sentences:
        _FLAT_SENTENCES.append((s.lower(), intent))

# ══════════════════════════════════════════════════════════════════════════════
#  Workout name synonyms for START_WORKOUT extraction
# ══════════════════════════════════════════════════════════════════════════════

WORKOUT_SYNONYMS = {
    WorkoutType.BICEP_CURL: [
        "bicep curl", "bicep", "curl", "bicep curls", "curls",
        "arm curl", "dumbbell curl",
    ],
    WorkoutType.SQUAT: [
        "squat", "squats", "deep squat", "barbell squat",
    ],
    WorkoutType.SIDE_SHOULDER: [
        "side shoulder", "side shoulder raise", "lateral raise",
        "side raise", "side", "lateral",
    ],
    WorkoutType.FRONT_SHOULDER: [
        "front shoulder", "front shoulder raise", "front raise",
        "front", "forward raise",
    ],
    WorkoutType.SHRUG: [
        "shrug", "shrugs", "shoulder shrug", "shoulder shrugs",
        "trap shrug",
    ],
}

# Flatten for matching
_FLAT_WORKOUTS = []
for wtype, names in WORKOUT_SYNONYMS.items():
    for name in names:
        _FLAT_WORKOUTS.append((name.lower(), wtype))

# Sort by length descending so longer matches are checked first
_FLAT_WORKOUTS.sort(key=lambda x: len(x[0]), reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Fuzzy Matching Engine
# ══════════════════════════════════════════════════════════════════════════════

FUZZY_THRESHOLD = 0.45  # Minimum similarity ratio

def _fuzzy_match_intent(text: str):
    """Find the best matching intent using fuzzy string matching."""
    best_ratio = 0.0
    best_intent = None

    for sentence, intent in _FLAT_SENTENCES:
        # Full sentence ratio
        ratio = difflib.SequenceMatcher(None, text, sentence).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_intent = intent

        # Also check if the sentence is a substring of the input
        if sentence in text:
            sub_ratio = max(ratio, 0.85)  # Boost substring matches
            if sub_ratio > best_ratio:
                best_ratio = sub_ratio
                best_intent = intent

        # Also check if input is a substring of the sentence
        if text in sentence and len(text) >= 3:
            sub_ratio = max(ratio, 0.75)
            if sub_ratio > best_ratio:
                best_ratio = sub_ratio
                best_intent = intent

    return best_intent, best_ratio


def _keyword_fallback(text: str):
    """Fallback keyword matching for robustness."""
    # Order matters: check more specific patterns first
    if any(w in text for w in ["stop", "end", "finish", "quit", "done", "close"]):
        # Make sure it's not "start" being misheard
        if "start" not in text:
            return VoiceCommand.STOP_SESSION

    if any(w in text for w in ["pause", "freeze", "hold on", "wait", "paws"]):
        return VoiceCommand.PAUSE

    if any(w in text for w in ["resume", "continue", "unpause", "keep going", "carry on", "go on"]):
        return VoiceCommand.RESUME

    if any(w in text for w in ["skip rest", "skip break", "no rest", "skip"]):
        return VoiceCommand.SKIP_REST

    if "next set" in text or "next" in text:
        return VoiceCommand.NEXT_SET

    if any(w in text for w in ["how many", "rep count", "my count", "my progress", "count", "reps", "repetition"]):
        return VoiceCommand.HOW_MANY_REPS

    if any(w in text for w in ["time", "clock", "duration", "minutes", "how long"]):
        return VoiceCommand.WHAT_TIME

    if any(w in text for w in ["tips", "form", "technique", "how do i", "how to", "correct", "should i", "help me"]):
        return VoiceCommand.WORKOUT_INFO

    if any(w in text for w in ["reset", "restart", "start over", "clear"]):
        return VoiceCommand.RESET_REPS

    if any(w in text for w in ["chat", "bot", "help", "question"]):
        return VoiceCommand.OPEN_CHATBOT

    return None


def extract_workout_name(text: str):
    """Extract a WorkoutType from free-form text. Returns None if not found."""
    text_lower = text.lower().strip()

    # Direct substring match (longest first)
    for name, wtype in _FLAT_WORKOUTS:
        if name in text_lower:
            return wtype

    # Fuzzy match against workout names
    best_ratio = 0.0
    best_wtype = None
    words = text_lower.split()

    for name, wtype in _FLAT_WORKOUTS:
        # Check against sliding windows of words
        name_words = len(name.split())
        for i in range(len(words)):
            window = " ".join(words[i:i+name_words])
            ratio = difflib.SequenceMatcher(None, window, name).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_wtype = wtype

    if best_ratio >= 0.65:
        return best_wtype

    return None


class VoiceCommandParser:
    @staticmethod
    def parse_command(text: str):
        """
        Parse a voice command using a 3-tier matching strategy:
        1. Check for START_WORKOUT (workout name in text with start/open/begin/do trigger)
        2. Fuzzy match against sentence database
        3. Keyword fallback

        Returns: (VoiceCommand, optional WorkoutType)
        """
        text = text.lower().strip()
        detected_workout = None

        # ── Tier 0: Check for workout start intent ────────────────────
        start_triggers = ["start", "open", "begin", "do", "let's do", "lets do",
                          "launch", "try", "i want to do", "i want"]
        has_start_trigger = any(trigger in text for trigger in start_triggers)

        workout = extract_workout_name(text)
        if workout:
            detected_workout = workout
            if has_start_trigger:
                return VoiceCommand.START_WORKOUT, detected_workout

        # ── Tier 1: Fuzzy match against database ──────────────────────
        intent, ratio = _fuzzy_match_intent(text)
        if intent and ratio >= FUZZY_THRESHOLD:
            return intent, detected_workout

        # ── Tier 2: Keyword fallback ──────────────────────────────────
        kw_intent = _keyword_fallback(text)
        if kw_intent:
            return kw_intent, detected_workout

        # ── Tier 3: If we found a workout name but no explicit start,
        #            still treat it as START_WORKOUT ────────────────────
        if detected_workout:
            return VoiceCommand.START_WORKOUT, detected_workout

        return VoiceCommand.UNKNOWN, None

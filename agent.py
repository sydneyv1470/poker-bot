"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "AlphaZero"
AGENT_NAME = "PokeBot"


# ===== YOUR CODE HERE =====

def new_half(config: AgentConfig, memory: dict) -> None:
    memory["hands_played"] = 0
    memory["time_used"] = 0.0
    memory["opp_raises"] = 0
    memory["opp_total"] = 0
    memory["last_hand"] = -1
    memory["seen"] = 0
    memory["pre_key"] = None
    memory["pre_eq"] = 0.5


def choose_action(obs: Observation, memory: dict) -> Action:
    import random
    import time

    legal = list(obs.legal_actions)
    safe = Action.CHECK_CALL if Action.CHECK_CALL in legal else legal[0]

    def pick(*prefs):
        for a in prefs:
            if a in legal:
                return a
        return safe

    try:
        # ---- opponent modeling: how often does the opponent raise? ----
        try:
            if memory.get("last_hand") != obs.hand_index:
                memory["last_hand"] = obs.hand_index
                memory["seen"] = 0
            opp_acts = [a for a in obs.action_history if not a.is_me]
            for a in opp_acts[memory.get("seen", 0):]:
                memory["opp_total"] = memory.get("opp_total", 0) + 1
                if str(getattr(a.action, "name", a.action)).startswith("RAISE"):
                    memory["opp_raises"] = memory.get("opp_raises", 0) + 1
            memory["seen"] = len(opp_acts)
        except Exception:
            pass
        agg = (memory.get("opp_raises", 0) + 3) / (memory.get("opp_total", 0) + 10)

        # ---- how strong is the opponent's range? (keep = fraction kept) ----
        to_call = obs.to_call
        keep = 1.0
        blind_only = (obs.street == Street.PREFLOP and obs.raises_this_street == 0)
        if to_call > 0 and not blind_only:
            prior_pot = max(1, obs.pot - to_call)
            b = to_call / prior_pot
            keep = 0.95 - 0.45 * b - 0.05 * obs.raises_this_street
            keep += 0.25 * max(0.0, agg - 0.35)
            keep = max(0.30, min(1.0, keep))

        # ---- equity vs that range, within a small time budget ----
        used = memory.get("time_used", 0.0)
        hands_left = max(1, 1000 - obs.hand_index)
        left = 1.0 - used
        budget = 0.0 if left <= 0 else min(0.0006, left / (hands_left * 3.2))

        t0 = time.perf_counter()
        key = (obs.hand_index, tuple(obs.hole_cards))
        if obs.street == Street.PREFLOP and keep == 1.0 and memory.get("pre_key") == key:
            eq = memory["pre_eq"]
        else:
            seen_cards = set(obs.hole_cards) | set(obs.board)
            remaining = [c for c in range(36) if c not in seen_cards]
            missing = 5 - len(obs.board)
            hole, board = list(obs.hole_cards), list(obs.board)
            min_n = 8 if left <= 0 else (20 if keep < 1.0 else 12)
            samples = []
            end = t0 + budget
            while len(samples) < min_n or (len(samples) < 80 and time.perf_counter() < end):
                draw = random.sample(remaining, 5 + missing)
                full = board + draw[5:]
                mine = omaha_score(hole, full)
                theirs = omaha_score(draw[:5], full)
                res = 1.0 if mine > theirs else (0.5 if mine == theirs else 0.0)
                samples.append((theirs, res))
            if keep < 1.0:
                samples.sort(key=lambda s: -s[0])
                k = max(5, int(round(keep * len(samples))))
                samples = samples[:k]
            eq = sum(s[1] for s in samples) / len(samples)
            if obs.street == Street.PREFLOP and keep == 1.0:
                memory["pre_key"] = key
                memory["pre_eq"] = eq
        memory["time_used"] = used + (time.perf_counter() - t0)

        # ---- decide ----
        need = to_call / (obs.pot + to_call) if to_call > 0 else 0.0

        if to_call == 0:
            if eq > 0.70:
                return pick(Action.RAISE_POT, Action.CHECK_CALL)
            if eq > 0.58:
                return pick(Action.RAISE_HALF, Action.CHECK_CALL)
            if 0.35 < eq < 0.50 and random.random() < 0.06:
                return pick(Action.RAISE_HALF, Action.CHECK_CALL)
            return pick(Action.CHECK_CALL)

        if eq > 0.75:
            return pick(Action.RAISE_POT, Action.CHECK_CALL)
        if eq > 0.62:
            return pick(Action.RAISE_HALF, Action.CHECK_CALL)
        if eq > need + 0.03:
            return pick(Action.CHECK_CALL)
        return pick(Action.FOLD, Action.CHECK_CALL)
    except Exception:
        return safe


def on_hand_end(result: HandResult, memory: dict) -> None:
    try:
        memory["hands_played"] = memory.get("hands_played", 0) + 1
    except Exception:
        pass

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)

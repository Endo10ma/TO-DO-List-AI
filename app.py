from flask import Flask, jsonify, request, send_from_directory
from datetime import datetime
from typing import List, Dict, Any, Optional
import json
import os
import re

# --- .env loader ---
try:
    from dotenv import load_dotenv, find_dotenv
    dotenv_path = find_dotenv(usecwd=True) or os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(dotenv_path)
    print(f"[ENV] Loaded .env from: {dotenv_path if os.path.exists(dotenv_path) else '(not found)'}")
except Exception as e:
    print(f"[ENV] Could not load .env: {e}")

app = Flask(__name__, static_folder="static", static_url_path="")

# In-memory "database"
tasks: List[Dict[str, Any]] = []
next_id: int = 1

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _find_task_by_desc(desc: str) -> Optional[Dict[str, Any]]:
    d = (desc or "").strip().lower()
    return next((t for t in tasks if t["description"].strip().lower() == d), None)

# ------------------- REST API -------------------

@app.get("/api/tasks")
def get_tasks():
    return jsonify({"tasks": tasks}), 200

@app.post("/api/tasks")
def create_task():
    global next_id
    data = request.get_json(silent=True) or {}
    description = (data.get("description") or "").strip()
    if not description:
        return jsonify({"error":"description is required"}), 400
    task = {"id": next_id, "description": description, "completed": False, "created_at": _now_iso()}
    tasks.append(task)
    next_id += 1
    return jsonify(task), 201

@app.patch("/api/tasks/by-description")
def complete_task_by_description():
    data = request.get_json(silent=True) or {}
    desc = (data.get("description") or "").strip()
    if not desc:
        return jsonify({"error":"description is required"}), 400
    task = _find_task_by_desc(desc)
    if not task:
        return jsonify({"error":"task not found"}), 404
    task["completed"] = True
    return jsonify(task), 200

@app.delete("/api/tasks/by-description")
def delete_task_by_description():
    data = request.get_json(silent=True) or {}
    desc = (data.get("description") or "").strip()
    if not desc:
        return jsonify({"error":"description is required"}), 400
    task = _find_task_by_desc(desc)
    if not task:
        return jsonify({"error":"task not found"}), 404
    tasks.remove(task)
    return jsonify({"ok":True, "deleted_description":desc}), 200

# ----------------- Chat Brain -------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("TODO_GPT_MODEL", "gpt-4o-mini")

try:
    from openai import OpenAI
    _openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    _llm_mode = "new"
except Exception:
    try:
        import openai as _openai_legacy
        if OPENAI_API_KEY:
            _openai_legacy.api_key = OPENAI_API_KEY
            _openai_client = _openai_legacy
            _llm_mode = "legacy"
        else:
            _openai_client = None
    except Exception:
        _openai_client = None
        _llm_mode = None

print(f"[LLM] API key present: {bool(OPENAI_API_KEY)}")
print(f"[LLM] Model: {OPENAI_MODEL}")
print(f"[LLM] SDK mode: {_llm_mode or 'none'}")
print("[LLM] ✅ Initialized." if _openai_client else "[LLM] ⚠️ No OpenAI client.")

LLM_SYSTEM_PROMPT = """You are a concise, friendly assistant for a To-Do app.

Respond with JSON when user clearly requests task actions, using one of:
- addTask(description: string)
- completeTask(description: string)
- deleteTask(description: string)
- viewTasks()

Rules for JSON:
- Output only one JSON object with keys: function, parameters
- No code fences, no markdown
- For ambiguous/multiple tasks, ask a clarifying question in plain text

Respond in plain text for greetings, small talk, or when asking clarification.
"""

def _strip_fences(s: str) -> str:
    if not s: return s
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s.split("\n",1)[-1]
    return s.strip()

def llm_route(message: str) -> Optional[Dict[str, Any]]:
    if not _openai_client:
        return None
    try:
        if _llm_mode == "new":
            resp = _openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role":"system","content":LLM_SYSTEM_PROMPT},
                    {"role":"user","content":message or ""},
                ],
                temperature=0.0
            )
            content = _strip_fences(resp.choices[0].message.content or "")
        else:
            resp = _openai_client.ChatCompletion.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role":"system","content":LLM_SYSTEM_PROMPT},
                    {"role":"user","content":message or ""},
                ],
                temperature=0.0
            )
            content = _strip_fences(resp["choices"][0]["message"]["content"] or "")
        try:
            data = json.loads(content)
            fn = data.get("function")
            params = data.get("parameters",{})
            if fn in {"addTask","completeTask","deleteTask","viewTasks"}:
                return {"mode":"tool","tool":data}
        except Exception:
            pass
        return {"mode":"text","reply":content}
    except Exception as e:
        print("[LLM][error]",e)
        return None

def _execute_tool(tool: dict) -> dict:
    fn = tool.get("function")
    p = tool.get("parameters",{})
    if fn=="addTask":
        desc = (p.get("description") or "").strip()
        if not desc: return {"error":"description required"}
        global next_id
        task={"id":next_id,"description":desc,"completed":False,"created_at":_now_iso()}
        tasks.append(task); next_id+=1; return {"ok":True,"task":task}
    if fn=="viewTasks":
        return {"ok":True,"tasks":tasks}
    if fn=="completeTask":
        desc = (p.get("description") or "").strip()
        t=_find_task_by_desc(desc)
        if not t: return {"error":"not found"}
        t["completed"]=True; return {"ok":True,"task":t}
    if fn=="deleteTask":
        desc=(p.get("description") or "").strip()
        t=_find_task_by_desc(desc)
        if not t: return {"error":"not found"}
        tasks.remove(t); return {"ok":True,"deleted_description":desc}
    return {"error":"unknown"}

@app.post("/api/brain/execute")
def brain_execute():
    data=request.get_json(silent=True) or {}
    msg=(data.get("message") or "").strip()
    llm=llm_route(msg)
    if not llm or llm.get("mode")=="text":
        return jsonify(llm or {"mode":"text","reply":"Sorry, I can't do that."}),200
    tool=llm["tool"]
    result=_execute_tool(tool)
    reply="OK"
    if tool["function"]=="addTask" and result.get("ok"): reply=f"Added “{tool['parameters']['description']}”."
    elif tool["function"]=="completeTask" and result.get("ok"): reply=f"Marked “{tool['parameters']['description']}” as done."
    elif tool["function"]=="deleteTask" and result.get("ok"): reply=f"Deleted “{tool['parameters']['description']}”."
    elif tool["function"]=="viewTasks": reply=f"You have {len(tasks)} task(s)."
    return jsonify({"mode":"tool","tool":tool,"result":result,"reply":reply}),200

@app.get("/")
def root(): return send_from_directory(app.static_folder,"index.html")

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)

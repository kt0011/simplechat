# lambda/index.py

import os
import json
import re
import boto3
from botocore.exceptions import ClientError

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List

import uvicorn
import nest_asyncio
from pyngrok import ngrok

# --- Bedrock クライアント初期化ユーティリティ ---
def extract_region_from_arn(arn: str) -> str:
    # ARN 形式: arn:aws:lambda:region:account-id:function:function-name
    m = re.search(r"arn:aws:lambda:([^:]+):", arn)
    return m.group(1) if m else os.environ.get("AWS_REGION", "us-east-1")

bedrock_client = None
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")

# --- FastAPI 定義 ---
app = FastAPI(
    title="Bedrock Chatbot API",
    description="AWS Bedrock を使ったチャット API",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["OPTIONS", "POST", "GET"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    conversationHistory: List[Dict[str, Any]] = []

# 起動時に一度だけ Bedrock クライアントを作成
@app.on_event("startup")
def startup_event():
    global bedrock_client
    if bedrock_client is None:
        # Lambda の context がないので環境変数から取得
        region = os.environ.get("AWS_REGION", "us-east-1")
        bedrock_client = boto3.client("bedrock-runtime", region_name=region)
        print(f"[startup] Initialized Bedrock client in region: {region}")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_ID if bedrock_client else None}

@app.post("/chat")
async def chat(req: ChatRequest):
    global bedrock_client

    if bedrock_client is None:
        # 万一 startup_event が動いていなければここで初期化
        region = os.environ.get("AWS_REGION", "us-east-1")
        bedrock_client = boto3.client("bedrock-runtime", region_name=region)

    # 会話履歴
    messages = req.conversationHistory.copy()
    messages.append({"role": "user", "content": req.message})

    # Bedrock 用ペイロード組立
    bedrock_messages = []
    for msg in messages:
        role = msg["role"]
        bedrock_messages.append({
            "role": role,
            "content": [{"text": msg["content"]}]
        })

    payload = {
        "messages": bedrock_messages,
        "inferenceConfig": {
            "maxTokens": 512,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 0.9,
        }
    }

    # Bedrock 呼び出し
    try:
        resp = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json"
        )
        body = json.loads(resp["body"].read())
    except ClientError as e:
        raise HTTPException(status_code=502, detail=f"Bedrock invocation failed: {e}")

    # レスポンス検証
    out = body.get("output", {}).get("message", {}).get("content")
    if not out or not isinstance(out, list) or not out[0].get("text"):
        raise HTTPException(status_code=502, detail="Invalid response from model")

    assistant_text = out[0]["text"]
    messages.append({"role": "assistant", "content": assistant_text})

    return {
        "success": True,
        "response": assistant_text,
        "conversationHistory": messages,
    }

# ngrok 経由で公開＋Uvicorn 起動
def run_with_ngrok(port: int = 8000):
    nest_asyncio.apply()
    ngrok_token = os.environ.get("NGROK_TOKEN")
    if not ngrok_token:
        print("⚠ ngrok トークンが NGROK_TOKEN 環境変数に設定されていません")
        return
    ngrok.set_auth_token(ngrok_token)
    public_url = ngrok.connect(port).public_url
    print("🚀 Public URL:", public_url)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

if __name__ == "__main__":
    run_with_ngrok(8000)

# app/api/eval_router.py
from __future__ import annotations
from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile, os, pandas as pd
from typing import Optional, List
from app.services.eval.ragas_runner import run_ragas_eval

router = APIRouter(tags=["eval"])

@router.post("/eval/ragas")
async def eval_ragas_api(file: UploadFile = File(...), metrics: Optional[str] = None):
    try:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, file.filename)
            data = await file.read()
            with open(path, "wb") as f:
                f.write(data)

            mlist = None
            if metrics:
                mlist = [m.strip() for m in metrics.split(",") if m.strip()]

            df = run_ragas_eval(path, use_metrics=mlist)
            # 평균만 요약해서 리턴
            means = df.select_dtypes(include="number").mean(numeric_only=True).to_dict()
            return {"summary": means, "rows": len(df)}
    except Exception as e:
        raise HTTPException(500, f"RAGAS evaluation failed: {e}")

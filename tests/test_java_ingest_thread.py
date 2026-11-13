"""Java 연계 문서 처리(convert-and-index)가 이벤트 루프를 막지 않는지 — 운영(2026-11-13) OCR 이 CPU 로 넘어가면
한 문서에 수 분이 걸리는데, 이벤트 루프에서 돌면 그동안 채팅·상태 조회 등 API 전체가 멈춘다."""
import asyncio
import time

import pytest

jr = pytest.importorskip("app.api.java_router")


def test_ingest_job_runs_off_the_event_loop():
    async def slow_job(seconds):
        time.sleep(seconds)                      # OCR·임베딩 같은 동기 작업
        return "done"

    async def main():
        gaps, last = [], time.monotonic()

        async def ticker():
            nonlocal last
            while True:
                await asyncio.sleep(0.02)
                now = time.monotonic()
                gaps.append(now - last)
                last = now

        t = asyncio.create_task(ticker())
        await jr._run_in_ingest_thread(slow_job, seconds=0.6)
        t.cancel()
        return max(gaps)

    assert asyncio.run(main()) < 0.3             # 작업 0.6초 동안에도 이벤트 루프는 계속 돈다


def test_ingest_jobs_run_one_at_a_time():
    running, peak = [0], [0]

    async def job(i):
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        time.sleep(0.1)
        running[0] -= 1

    async def main():
        await asyncio.gather(*(jr._run_in_ingest_thread(job, i=i) for i in range(3)))

    asyncio.run(main())
    assert peak[0] == 1


def test_convert_and_index_schedules_job_in_ingest_thread(monkeypatch):
    from fastapi import BackgroundTasks

    monkeypatch.setattr(jr, "verify_internal_token", lambda t: True)
    monkeypatch.setattr(jr, "resolve_source_path", lambda *a: None)

    class FakeDB:
        def get_file_by_id(self, data_id):
            return None

    monkeypatch.setattr(jr, "DBConnector", FakeDB)
    bt = BackgroundTasks()
    req = jr.ConvertAndIndexRequest(data_id="1", path="a", file_id="f.pdf", callback_url=None)
    asyncio.run(jr.convert_and_index(req, bt, x_internal_token="t"))
    task = bt.tasks[0]
    assert task.func is jr._run_in_ingest_thread
    assert task.args[0] is jr.process_convert_and_index_prod

from __future__ import annotations
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from .. import jobs as job_mod
from ..auth import current_user
from ..db import User

router = APIRouter()


_ACTIVE_STATUSES = {"queued", "running"}


def _job_dict(job: job_mod.Job) -> dict:
    return job.to_dict()


async def _own_job_or_404(job_id: str, user: User) -> job_mod.Job:
    job = await job_mod.get_job(job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status_code=404, detail="Job niet gevonden")
    return job


@router.get("/v1/jobs")
async def list_jobs(user: Annotated[User, Depends(current_user)]) -> dict:
    jobs = await job_mod.list_user_jobs(user.id, max_age_sec=24 * 3600)
    return {"items": [_job_dict(job) for job in jobs]}


@router.get("/v1/jobs/{job_id}")
async def get_job(
    job_id: str,
    user: Annotated[User, Depends(current_user)],
) -> dict:
    job = await _own_job_or_404(job_id, user)
    return _job_dict(job)


@router.delete("/v1/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    user: Annotated[User, Depends(current_user)],
) -> dict:
    job = await _own_job_or_404(job_id, user)
    if job.status not in _ACTIVE_STATUSES:
        raise HTTPException(status_code=409, detail="Job is al afgerond")
    await job_mod.cancel_job(job)
    return _job_dict(job)

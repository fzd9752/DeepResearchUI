import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.config import get_settings


router = APIRouter()


@router.post("")
async def upload_file(file: UploadFile = File(...)):
    settings = get_settings()
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_upload_extensions:
        raise HTTPException(status_code=400, detail="file type not allowed")

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = f"file_{uuid.uuid4().hex}"
    safe_name = os.path.basename(filename)
    destination = upload_dir / f"{file_id}_{safe_name}"

    size_bytes = 0
    try:
        with destination.open("wb") as out_file:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > settings.max_upload_size:
                    raise HTTPException(status_code=413, detail="file too large")
                out_file.write(chunk)
    except HTTPException:
        if destination.exists():
            destination.unlink()
        raise
    finally:
        await file.close()

    return {
        "file_id": file_id,
        "filename": filename,
        "size_bytes": size_bytes,
        "mime_type": file.content_type or "application/octet-stream",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }

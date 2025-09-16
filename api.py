import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from invoice_ocr import process_pdf_for_reference, ensure_tesseract_cmd


app = FastAPI(title="UPS Invoice OCR API", version="1.0.0")


@app.get("/health")
def health():

	return {"status": "ok"}


@app.get("/extract")
def extract_get(pdf_path: str, ref: str, tesseract_cmd: Optional[str] = None, scale: float = 2.0, lang: str = "eng", context: int = 2):

	if not os.path.isfile(pdf_path):
		raise HTTPException(status_code=400, detail=f"PDF not found: {pdf_path}")
	ensure_tesseract_cmd(tesseract_cmd)
	result = process_pdf_for_reference(
		pdf_path=pdf_path,
		reference=ref,
		tesseract_cmd=tesseract_cmd,
		scale=scale,
		lang=lang,
		context_window=context,
	)
	return JSONResponse(content=result)


@app.post("/extract")
async def extract_post(
	ref: str = Form(...),
	pdf: UploadFile = File(...),
	tesseract_cmd: Optional[str] = Form(None),
	scale: float = Form(2.0),
	lang: str = Form("eng"),
	context: int = Form(2),
):

	# Save uploaded PDF to a temp file
	try:
		data = await pdf.read()
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Failed to read upload: {e}")

	if not data:
		raise HTTPException(status_code=400, detail="Empty file upload")

	filename = pdf.filename or "uploaded.pdf"
	tmp_path = os.path.join(".", f"_upload_{os.getpid()}_{filename}")
	with open(tmp_path, "wb") as f:
		f.write(data)

	try:
		ensure_tesseract_cmd(tesseract_cmd)
		result = process_pdf_for_reference(
			pdf_path=tmp_path,
			reference=ref,
			tesseract_cmd=tesseract_cmd,
			scale=scale,
			lang=lang,
			context_window=context,
		)
		return JSONResponse(content=result)
	finally:
		try:
			os.remove(tmp_path)
		except Exception:
			pass



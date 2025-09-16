import os
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from invoice_ocr import process_pdf_for_reference, ensure_tesseract_cmd
import pytesseract  # type: ignore


app = FastAPI(title="UPS Invoice OCR API", version="1.0.0")


@app.get("/")
def root():

	return {
		"message": "UPS Invoice OCR API",
		"docs": "/docs",
		"endpoints": {
			"GET /health": "Service health check",
			"GET /extract": "Query params: pdf_path, ref, [tesseract_cmd, scale, lang, context]",
			"POST /extract": "Form fields: ref, pdf(@file), [tesseract_cmd, scale, lang, context]",
			"GET /ups/by-ref/{ref}": "Convenience: uses ups_invoice.pdf by default",
			"GET /fedex/by-ref/{ref}": "Convenience: uses ups_invoice.pdf by default",
		},
	}


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


@app.get("/diag/tesseract")
def diag_tesseract():

	cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
	try:
		version = str(pytesseract.get_tesseract_version())
	except Exception:
		version = None
	return {"tesseract_cmd": cmd, "version": version}


@app.get("/diag/check-pdf")
def diag_check_pdf(pdf: str = "ups_invoice.pdf"):

	exists = os.path.isfile(pdf)
	return {"pdf": pdf, "exists": exists, "cwd": os.getcwd()}


@app.get("/ups/by-ref/{ref}")
def ups_by_ref(
	ref: str,
	pdf: Optional[str] = None,
	tesseract_cmd: Optional[str] = None,
	scale: float = 2.0,
	lang: str = "eng",
	context: int = 2,
):

	pdf_path = pdf or "ups_invoice.pdf"
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


@app.get("/fedex/by-ref/{ref}")
def fedex_by_ref(
	ref: str,
	pdf: Optional[str] = None,
	tesseract_cmd: Optional[str] = None,
	scale: float = 2.0,
	lang: str = "eng",
	context: int = 2,
):

	# For now, reuse the same OCR logic and default PDF. Adjust if a FedEx-specific parser is added.
	pdf_path = pdf or "ups_invoice.pdf"
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


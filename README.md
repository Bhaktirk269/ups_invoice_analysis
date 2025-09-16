### UPS invoice OCR extractor

Extract text from a UPS invoice PDF using OCR (Tesseract), search for a given reference number, and output the matching details as JSON.

### Requirements

- Python 3.9+
- Tesseract OCR (Windows installer)
- Python packages from `requirements.txt`

### Install Tesseract (Windows)

1. Download the Windows installer (UB Mannheim build recommended) and install Tesseract OCR.
   - Typical path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
2. During installation, check the option to "Add Tesseract to system PATH" or remember the install path.

If Tesseract is not on PATH, you can pass the full path with `--tesseract-cmd` when running the script.

### Setup Python environment

Open PowerShell in this folder and run:

```powershell
py -m venv .venv
. .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Usage

Run the extractor, providing your PDF and the reference number to search for. The script prints JSON to the console or saves it to a file.

```powershell
python invoice_ocr.py --pdf ups_invoice.pdf --ref YOUR_REFERENCE
```

Optionally specify Tesseract path and save to a JSON file:

```powershell
python invoice_ocr.py --pdf ups_invoice.pdf --ref YOUR_REFERENCE \
  --tesseract-cmd "C:/Program Files/Tesseract-OCR/tesseract.exe" \
  --output result.json
```

Adjust rendering scale or language if needed:

```powershell
python invoice_ocr.py --pdf ups_invoice.pdf --ref REF123 --scale 2.5 --lang eng --context 3
```

### Output format

The script returns a JSON object like:

```json
{
  "reference": "REF123",
  "pdf": "C:/.../ups_invoice.pdf",
  "matches": [
    {
      "page": 1,
      "matched_line": "Reference Number: REF123",
      "context": [
        "Shipment Date: 2025-09-01",
        "Reference Number: REF123",
        "Total: $42.50"
      ],
      "key_values": {
        "Shipment Date": "2025-09-01",
        "Total": "$42.50"
      }
    }
  ]
}
```

### Notes

- OCR quality is influenced by the PDF. If results are poor, try `--scale 3.0`.
- The script heuristically extracts key/value pairs (lines with `Label: Value`). Adjust as needed for your invoice template.
- If you have non-English text, install the appropriate Tesseract language data and pass `--lang`.



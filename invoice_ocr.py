import argparse
import json
import os
import re
import sys
import shutil
from typing import Dict, List, Optional, Tuple
import math

import cv2  # type: ignore
import numpy as np  # type: ignore
import pypdfium2 as pdfium  # type: ignore
import pytesseract  # type: ignore
from PIL import Image  # type: ignore


def is_tesseract_available() -> bool:

	try:
		# This will raise if the binary is not available
		_ = pytesseract.get_tesseract_version()
		return True
	except Exception:
		return False


def ensure_tesseract_cmd(explicit_cmd: Optional[str]) -> None:

	if explicit_cmd:
		pytesseract.pytesseract.tesseract_cmd = explicit_cmd
		return

	# Allow overriding via environment variable
	env_cmd = os.getenv("TESSERACT_CMD")
	if env_cmd and os.path.isfile(env_cmd):
		pytesseract.pytesseract.tesseract_cmd = env_cmd
		return

	common_windows_paths: List[str] = [
		r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
		r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
	]

	for candidate in common_windows_paths:
		if os.path.isfile(candidate):
			pytesseract.pytesseract.tesseract_cmd = candidate
			return

	# Linux/macOS: prefer PATH discovery first
	which_path = shutil.which("tesseract")
	if which_path:
		pytesseract.pytesseract.tesseract_cmd = which_path
		return

	# Fallback to typical absolute paths
	for candidate in ("/usr/bin/tesseract", "/usr/local/bin/tesseract"):
		if os.path.isfile(candidate):
			pytesseract.pytesseract.tesseract_cmd = candidate
			return

	# If not found, rely on PATH or let pytesseract raise a helpful error later
	return


def render_pdf_to_images(pdf_path: str, scale: float = 2.0) -> List[Image.Image]:

	doc = pdfium.PdfDocument(pdf_path)
	images: List[Image.Image] = []
	for page_index in range(len(doc)):
		page = doc[page_index]
		bitmap = page.render(scale=scale, rotation=0)
		pil_image = bitmap.to_pil()
		images.append(pil_image)
	return images


def extract_pdf_text_lines(pdf_path: str) -> List[List[Dict]]:

	# Returns a list of pages, where each page is a list of line dicts { text, page_num, ... }
	doc = pdfium.PdfDocument(pdf_path)
	pages_lines: List[List[Dict]] = []
	for page_index in range(len(doc)):
		page = doc[page_index]
		textpage = page.get_textpage()
		full_text = textpage.get_text_range()
		lines_raw = full_text.splitlines()
		page_lines: List[Dict] = []
		for ln_idx, text in enumerate(lines_raw, start=1):
			text = (text or "").strip()
			if not text:
				continue
			page_lines.append(
				{
					"page_num": page_index + 1,
					"block_num": 0,
					"par_num": 0,
					"line_num": ln_idx,
					"text": text,
					"bbox": None,
				}
			)
		pages_lines.append(page_lines)
	return pages_lines


def _line_contains_reference_cue(text: str) -> bool:

	return bool(re.search(r"reference\s*no\.?\s*\d*", text, re.IGNORECASE))


def _guess_reference_from_line(text: str) -> Optional[str]:

	# Heuristic: pick the longest alphanumeric token (3-30 chars) that isn't a keyword
	tokens = re.findall(r"[A-Za-z0-9#]+", text)
	ignore = {"reference", "no", "ref", "number", "shipment"}
	candidates = [t for t in tokens if 3 <= len(t) <= 30 and t.lower() not in ignore]
	return max(candidates, key=len) if candidates else None


def process_pdf_all_shipments(
	pdf_path: str,
	tesseract_cmd: Optional[str] = None,
	scale: float = 2.0,
	lang: str = "eng",
	context_window: int = 2,
) -> Dict:

	# Build lines using OCR if available, else embedded text fallback
	ensure_tesseract_cmd(tesseract_cmd)
	use_ocr = is_tesseract_available()
	pages: List[List[Dict]]
	page_images: List[Image.Image] = []
	if use_ocr:
		images = render_pdf_to_images(pdf_path, scale=scale)
		for_ocr_pages: List[List[Dict]] = []
		for pil_image in images:
			processed = preprocess_image_for_ocr(pil_image)
			data = ocr_image_to_data(processed, lang=lang)
			lines = build_lines_from_tesseract_data(data)
			for ln in lines:
				ln["_page_image_bin"] = processed
				ln["_page_image_raw"] = pil_image
			for_ocr_pages.append(lines)
			page_images.append(pil_image)
		pages = for_ocr_pages
	else:
		pages = extract_pdf_text_lines(pdf_path)

	shipments: List[Dict] = []
	seen: set = set()
	for page_number, lines in enumerate(pages, start=1):
		# Candidate anchors: lines mentioning Reference No, or lines containing tracking numbers
		for idx, ln in enumerate(lines):
			text = ln.get("text", "")
			is_ref_anchor = _line_contains_reference_cue(text)
			tracking_present = bool(re.search(r"\b1Z[0-9A-Z]{10,}\b", text))
			weight_present = bool(re.search(r"\b\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?(?:\s*[Cc])?\b", text))
			if not (is_ref_anchor or tracking_present or weight_present):
				continue

			# Guess a local reference to drive the row extractor heuristics
			guessed_ref = _guess_reference_from_line(text) or "REF"
			context_lines = extract_context(lines, idx, window=context_window)
			key_values = extract_key_values_from_lines(context_lines)
			fields = extract_ups_structured_fields(context_lines)
			requested = extract_requested_fields(lines, idx, guessed_ref)
			fields.pop("sender", None)
			fields.pop("consignee", None)
			key_values = {k: v for k, v in key_values.items() if k.lower() not in ("sender", "consignee")}

			# Deduplicate using key info
			k = (
				requested.get("tracking_no"),
				requested.get("shipment_no"),
				requested.get("reference_no_2"),
				page_number,
				text,
			)
			if k in seen:
				continue
			seen.add(k)

			shipments.append(
				{
					"page": page_number,
					"matched_line": text,
					"requested": requested,
					"key_values": key_values,
				}
			)

	return {
		"pdf": os.path.abspath(pdf_path),
		"shipments": shipments,
	}


def preprocess_image_for_ocr(pil_image: Image.Image) -> Image.Image:

	image_array = np.array(pil_image)
	if image_array.ndim == 3 and image_array.shape[2] == 3:
		gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
	else:
		gray = image_array

	# Adaptive thresholding helps in uneven lighting/backgrounds common in printed PDFs
	thresh = cv2.adaptiveThreshold(
		gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
	)

	# Mild dilation to connect broken characters
	kernel = np.ones((1, 1), np.uint8)
	processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

	return Image.fromarray(processed)


def ocr_image_to_data(pil_image: Image.Image, lang: str = "eng") -> Dict[str, List]:

	config = "--oem 3 --psm 6"
	data = pytesseract.image_to_data(
		pil_image, lang=lang, config=config, output_type=pytesseract.Output.DICT
	)
	return data


def build_lines_from_tesseract_data(data: Dict[str, List]) -> List[Dict]:

	lines: List[Dict] = []
	current_key: Optional[Tuple[int, int, int, int]] = None
	current_words: List[str] = []
	current_meta: Dict[str, int] = {}
	current_bbox: Optional[Tuple[int, int, int, int]] = None  # left, top, right, bottom

	for i in range(len(data["text"])):
		if int(data["conf"][i]) < 0:
			continue

		key = (
			int(data["page_num"][i]),
			int(data["block_num"][i]),
			int(data["par_num"][i]),
			int(data["line_num"][i]),
		)

		word = (data["text"][i] or "").strip()
		if not word:
			# Still need to handle key boundaries even on empty tokens
			pass

		if current_key is None:
			current_key = key
			current_words = []
			current_meta = {
				"page_num": key[0],
				"block_num": key[1],
				"par_num": key[2],
				"line_num": key[3],
			}
			current_bbox = None

		if key != current_key:
			lines.append(
				{
					"page_num": current_meta["page_num"],
					"block_num": current_meta["block_num"],
					"par_num": current_meta["par_num"],
					"line_num": current_meta["line_num"],
					"text": " ".join(current_words).strip(),
					"bbox": current_bbox,
				}
			)
			current_key = key
			current_words = []
			current_meta = {
				"page_num": key[0],
				"block_num": key[1],
				"par_num": key[2],
				"line_num": key[3],
			}
			current_bbox = None

		if word:
			current_words.append(word)
			# Update bbox from word box
			try:
				l = int(data["left"][i])
				t = int(data["top"][i])
				w = int(data["width"][i])
				h = int(data["height"][i])
				r = l + w
				b = t + h
				if current_bbox is None:
					current_bbox = (l, t, r, b)
				else:
					cl, ct, cr, cb = current_bbox
					current_bbox = (min(cl, l), min(ct, t), max(cr, r), max(cb, b))
			except Exception:
				pass

	# Flush last line
	if current_key is not None:
		lines.append(
			{
				"page_num": current_meta["page_num"],
				"block_num": current_meta["block_num"],
				"par_num": current_meta["par_num"],
				"line_num": current_meta["line_num"],
				"text": " ".join(current_words).strip(),
				"bbox": current_bbox,
			}
		)

	# Remove empty lines
	lines = [ln for ln in lines if ln["text"]]
	return lines


def normalize_text(text: str) -> str:

	return re.sub(r"[^A-Za-z0-9]", "", text).lower()


def find_reference_matches(lines: List[Dict], reference: str) -> List[Tuple[int, int]]:

	normalized_reference = normalize_text(reference)
	matches: List[Tuple[int, int]] = []
	for idx, line in enumerate(lines):
		if normalized_reference in normalize_text(line["text"]):
			matches.append((line["page_num"], idx))
	return matches


def extract_context(lines: List[Dict], line_index: int, window: int = 2) -> List[Dict]:

	start = max(0, line_index - window)
	end = min(len(lines), line_index + window + 1)
	return lines[start:end]


def extract_ups_structured_fields(context_lines: List[Dict]) -> Dict[str, object]:

	joined = " \n ".join(ln["text"] for ln in context_lines)

	# Date formats like "06 Apr 12" or "06 Apr 2012"
	date_match = re.search(r"\b(\d{1,2}\s+[A-Za-z]{3}\s+\d{2,4})\b", joined)

	# One or more UPS tracking numbers like 1ZXXXXXXXXXXXXXXX possibly followed by a '#'
	tracking_numbers = re.findall(r"\b1Z[0-9A-Z]{10,}\b", joined)

	# Try to capture service names commonly found on UPS invoices
	service_match = re.search(
		"|".join(
			[
				r"Worldwide Express Saver",
				r"Worldwide Express",
				r"WW Express Saver",
				r"Worldwide Expedited",
				r"Expedited",
				r"Standard",
				r"Ground",
			]
		),
		joined,
		re.IGNORECASE,
	)

	# Weights often shown as "19.8/19.80" (actual/chargeable)
	weights_match = re.search(r"\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\b", joined)

	# Shipment number: long non-1Z alphanumeric token (avoid obvious words)
	# Heuristic: choose the longest 12-22 length token not starting with 1Z and not ALL letters
	long_tokens = re.findall(r"\b(?!1Z)[A-Z0-9]{12,22}\b", joined)
	shipment_number: Optional[str] = None
	if long_tokens:
		# Prefer one containing digits and letters
		candidates = [t for t in long_tokens if re.search(r"[A-Z]", t) and re.search(r"\d", t)]
		shipment_number = max(candidates or long_tokens, key=len)

	# Sender/Consignee from explicit labeled lines if present
	sender = None
	consignee = None
	for ln in context_lines:
		text = ln["text"]
		m_sender = re.match(r"\s*Sender\s*[:\-–]\s*(.+)", text, re.IGNORECASE)
		m_consignee = re.match(r"\s*Consignee\s*[:\-–]\s*(.+)", text, re.IGNORECASE)
		if m_sender:
			sender = m_sender.group(1).strip()
		if m_consignee:
			consignee = m_consignee.group(1).strip()

	return {
		"date": date_match.group(1) if date_match else None,
		"shipment_number": shipment_number,
		"tracking_numbers": tracking_numbers or None,
		"service": service_match.group(0) if service_match else None,
		"weights": {
			"actual": float(weights_match.group(1)) if weights_match else None,
			"chargeable": float(weights_match.group(2)) if weights_match else None,
		},
		"sender": sender,
		"consignee": consignee,
	}


def extract_key_values_from_lines(lines: List[Dict]) -> Dict[str, str]:

	key_values: Dict[str, str] = {}
	pattern = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /._-]{0,40}?)\s*[:\-–]\s*(.+?)\s*$")
	for line in lines:
		text = line["text"]
		m = pattern.match(text)
		if m:
			key = re.sub(r"\s+", " ", m.group(1)).strip()
			value = m.group(2).strip()
			if key and value and key not in key_values:
				key_values[key] = value
	return key_values


def _is_container_token(token: str) -> bool:

	return token.upper() in {"PKG", "DOC", "LTR", "BOX", "CARTON", "PALLET", "ENVELOPE", "BAG"}


def _parse_amount(text: str) -> Optional[float]:

	m = re.search(r"([\$€₹]?\s*[\d,]+(?:\.\d{2})?)", text)
	if not m:
		return None
	val = m.group(1)
	val = re.sub(r"[^0-9.,]", "", val)
	# Normalize comma as thousands, dot as decimal
	if val.count(",") > 0 and val.count(".") == 0:
		val = val.replace(",", "")
	else:
		val = val.replace(",", "")
	try:
		return float(val)
	except ValueError:
		return None


def extract_requested_fields(page_lines: List[Dict], match_index: int, searched_reference: str) -> Dict[str, object]:

	# Build a local window around the match to parse row-level info
	context_lines = extract_context(page_lines, match_index, window=3)
	joined_context = " \n ".join(ln["text"] for ln in context_lines)

	# Also look a bit further down for charges summaries
	start = max(0, match_index - 3)
	end = min(len(page_lines), match_index + 15)
	bill_lines = page_lines[start:end]
	joined_bill = " \n ".join(ln["text"] for ln in bill_lines)

	# Reference numbers and container from matched line
	matched = page_lines[match_index]["text"]
	ref1: Optional[str] = None
	ref2: Optional[str] = None
	container: Optional[str] = None

	# Tokenize conservatively
	tokens = re.findall(r"[A-Za-z0-9#]+", matched)
	# Find index of searched_reference within tokens (exact or contained)
	ref_idx: Optional[int] = None
	for i, tok in enumerate(tokens):
		if searched_reference in tok:
			ref_idx = i
			ref2 = tok
			break

	if ref_idx is not None:
		if ref_idx - 1 >= 0:
			ref1 = tokens[ref_idx - 1]
		if ref_idx + 1 < len(tokens) and _is_container_token(tokens[ref_idx + 1]):
			container = tokens[ref_idx + 1]

	# If container not on the matched line, try to infer from nearby lines
	if not container:
		m_cont = re.search(r"\b(PKG|DOC|LTR|BOX|CARTON|PALLET|ENVELOPE|BAG)\b", joined_context, re.IGNORECASE)
		if m_cont:
			container = m_cont.group(1).upper()

	# Tracking and shipment numbers near the row
	tracking_match = re.search(r"\b(1Z[0-9A-Z]{10,})\b", joined_context)
	shipment_match = re.search(r"\b(?!1Z)[A-Z0-9]{12,22}\b", joined_context)
	tracking_no = tracking_match.group(1) if tracking_match else None
	shipment_no = shipment_match.group(0) if shipment_match else None

	# Description: use service name if present in context
	service_match = re.search(
		"|".join(
			[
				r"Worldwide Express Saver",
				r"Worldwide Express",
				r"WW Express Saver",
				r"Worldwide Expedited",
				r"Expedited",
				r"Standard",
				r"Ground",
			]
		),
		joined_context,
		re.IGNORECASE,
	)
	description = service_match.group(0) if service_match else None

	# Weight: find the NEAREST line to the match that actually contains the pair (X/Y),
	# parse from text first, then confirm with OCR.
	weight_val: Optional[float] = None
	chargeable_val: Optional[float] = None
	chargeable_suffix_c: bool = False
	weight_line_idx: Optional[int] = None
	weight_line_regex = re.compile(r"\b\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?(?:\s*[Cc])?\b")

	search_lo = max(0, match_index - 12)
	search_hi = min(len(page_lines), match_index + 20)
	max_radius = max(match_index - search_lo, search_hi - match_index)
	for radius in range(0, max_radius + 1):
		for cand in (match_index - radius, match_index + radius):
			if cand < search_lo or cand >= search_hi:
				continue
			if weight_line_regex.search(page_lines[cand]["text"]):
				weight_line_idx = cand
				break
		if weight_line_idx is not None:
			break

	# Try to parse from the nearest weight line text; prefer plausible pairs
	def _parse_pair_from_text(text: str) -> Optional[Tuple[float, float, bool]]:
		m = re.search(r"\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?:\s*([Cc]))?\b", text)
		if not m:
			return None
		try:
			return float(m.group(1)), float(m.group(2)), bool(m.group(3))
		except ValueError:
			return None

	date_regex = re.compile(r"\b\d{1,2}\s+[A-Za-z]{3}\b")
	service_regex = re.compile(r"express|saver|expedited|standard|ground", re.IGNORECASE)

	def _score_candidate_at_index(i: int, act: Optional[float], chg: Optional[float]) -> float:
		if act is None or chg is None:
			return 1e9
		dist = abs(i - match_index)
		score = abs(chg - act)
		# Penalize actual greater than chargeable
		if act > chg:
			score += 0.75
		# Prefer chargeable fractions near .50 or .00
		frac_c = round(chg - math.floor(chg), 2)
		if min(abs(frac_c - 0.50), abs(frac_c - 0.00)) > 0.1:
			score += 0.15
		# Distance penalty from matched line
		score += 0.02 * dist
		# Direction bias: prefer lines ABOVE the matched line
		if i > match_index:
			score += 0.30
		# Strong preference for the 1-3 lines just above
		if match_index - 3 <= i < match_index:
			score -= 0.40
		# Prefer lines that look like the detail row (date/service keywords)
		line_text = page_lines[i]["text"]
		if date_regex.search(line_text) or service_regex.search(line_text):
			score -= 0.20
		# Slight bias toward smaller plausible chargeable weights
		score += 0.01 * chg
		return score

	# Build candidates from a small window around the match, score them, and choose the best
	candidates: List[Tuple[float, float, bool, int, float, bool, bool]] = []  # act, chg, sufC, dist, score, has_ship, has_track
	window_lo = max(0, match_index - 8)
	window_hi = min(len(page_lines), match_index + 12)
	for i in range(window_lo, window_hi):
		text = page_lines[i]["text"]
		parsed = _parse_pair_from_text(text)
		if parsed:
			act, chg, sufC = parsed
			dist = abs(i - match_index)
			score = _score_candidate_at_index(i, act, chg)
			text_upper = text.upper()
			has_ship = bool(shipment_no and shipment_no.upper() in text_upper)
			has_track = bool(tracking_no and tracking_no.upper() in text_upper)
			# Strong bonus if the line includes both identifiers (typical UPS detail row)
			if has_ship and has_track:
				score -= 1.0
			elif has_ship or has_track:
				score -= 0.4
			# Extra bias for lines above the match
			if i < match_index:
				score -= 0.2
			candidates.append((act, chg, sufC, dist, score, has_ship, has_track))
			continue
		# If looks like a candidate line but failed parse, try OCR
		if weight_line_regex.search(text):
			img = page_lines[i].get("_page_image_raw") or page_lines[i].get("_page_image_bin")
			bbox = page_lines[i].get("bbox")
			if img is not None and bbox is not None:
				p = _reocr_line_weight_pair(img, bbox)
				if p:
					act, chg, sufC = p[0], p[1], bool(p[2])
					dist = abs(i - match_index)
					score = _score_candidate_at_index(i, act, chg)
					text_upper = text.upper()
					has_ship = bool(shipment_no and shipment_no.upper() in text_upper)
					has_track = bool(tracking_no and tracking_no.upper() in text_upper)
					if has_ship and has_track:
						score -= 1.0
					elif has_ship or has_track:
						score -= 0.4
					if i < match_index:
						score -= 0.2
					candidates.append((act, chg, sufC, dist, score, has_ship, has_track))

	# Selection: prefer lines with both shipment and tracking present, then smaller chargeable
	pair = None
	if candidates:
		both_anchor = [c for c in candidates if c[5] and c[6]]
		one_anchor = [c for c in candidates if (c[5] ^ c[6])]
		fallback = candidates
		pick_from = both_anchor or one_anchor or fallback
		# Among the chosen set, pick by lowest score, and if tie, lowest chargeable
		pick_from.sort(key=lambda c: (c[4], c[1]))
		best = pick_from[0]
		pair = (best[0], best[1], best[2])
	if pair:
		weight_val = pair[0]
		chargeable_val = pair[1]
		chargeable_suffix_c = bool(pair[2])
	else:
		# Fallback: widen to bill text then context
		weight_pair = re.search(r"\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?:\s*([Cc]))?\b", joined_bill)
		if weight_pair:
			try:
				weight_val = float(weight_pair.group(1))
				chargeable_val = float(weight_pair.group(2)) if weight_pair.group(2) else None
				chargeable_suffix_c = bool(weight_pair.group(3))
			except ValueError:
				weight_val = None
		else:
			m_weight = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:kg|lb|lbs)?\b", joined_context, re.IGNORECASE)
			if m_weight:
				try:
					weight_val = float(m_weight.group(1))
				except ValueError:
					weight_val = None

	weight_container_str: Optional[str] = None
	# Align actual .8→.5 only when chargeable indicates a .50 step
	if 'chargeable_val' in locals() and chargeable_val is not None:
		weight_val = _snap_actual_weight_to_half_step(weight_val, chargeable_val)

	if weight_val is not None and container:
		if 'chargeable_val' in locals() and chargeable_val is not None:
			suffix = 'C' if ('chargeable_suffix_c' in locals() and chargeable_suffix_c) else ''
			weight_container_str = f"{weight_val:.1f}/{chargeable_val:.2f}{suffix} {container}"
		else:
			weight_container_str = f"{weight_val} {container}"
	elif weight_val is not None:
		weight_container_str = f"{weight_val}"
	elif container:
		weight_container_str = f"{container}"

	# Charges
	freight = None
	fuel_surcharge = None
	net_charges = None
	total_charges_for_shipment = None

	m_freight = re.search(r"\bFREIGHT\b\s*[:\-]?\s*([$€₹]?\s*[\d,]+(?:\.\d{2})?)", joined_bill, re.IGNORECASE)
	if m_freight:
		freight = _parse_amount(m_freight.group(1))
		if freight is None:
			# Try numeric-focused re-OCR to the right of the line that mentions FREIGHT
			for ln in bill_lines:
				if re.search(r"\bFREIGHT\b", ln["text"], re.IGNORECASE):
					freight = _reocr_numeric_region(page_lines[match_index]["_page_image"], ln.get("bbox")) if "_page_image" in page_lines[match_index] else None
					break

	m_fuel = re.search(r"\bFUEL\s+SURCHARGE\b\s*[:\-]?\s*([$€₹]?\s*[\d,]+(?:\.\d{2})?)", joined_bill, re.IGNORECASE)
	if m_fuel:
		fuel_surcharge = _parse_amount(m_fuel.group(1))
		if fuel_surcharge is None:
			for ln in bill_lines:
				if re.search(r"\bFUEL\s+SURCHARGE\b", ln["text"], re.IGNORECASE):
					fuel_surcharge = _reocr_numeric_region(page_lines[match_index]["_page_image"], ln.get("bbox")) if "_page_image" in page_lines[match_index] else None
					break

	m_net = re.search(r"\bNet\s+Charges\b\s*[:\-]?\s*([$€₹]?\s*[\d,]+(?:\.\d{2})?)", joined_bill, re.IGNORECASE)
	if m_net:
		net_charges = _parse_amount(m_net.group(1))
		if net_charges is None:
			for ln in bill_lines:
				if re.search(r"\bNet\s+Charges\b", ln["text"], re.IGNORECASE):
					net_charges = _reocr_numeric_region(page_lines[match_index]["_page_image"], ln.get("bbox")) if "_page_image" in page_lines[match_index] else None
					break

	m_total = re.search(r"\bTotal\s+Charges\s+for\s+Shipment\b\s*[:\-]?\s*([$€₹]?\s*[\d,]+(?:\.\d{2})?)", joined_bill, re.IGNORECASE)
	if m_total:
		total_charges_for_shipment = _parse_amount(m_total.group(1))
		if total_charges_for_shipment is None:
			for ln in bill_lines:
				if re.search(r"\bTotal\s+Charges\s+for\s+Shipment\b", ln["text"], re.IGNORECASE):
					total_charges_for_shipment = _reocr_numeric_region(page_lines[match_index]["_page_image"], ln.get("bbox")) if "_page_image" in page_lines[match_index] else None
					break

	# Prefer computed total when both components are available
	if freight is not None and fuel_surcharge is not None:
		total_charges_for_shipment = round(freight + fuel_surcharge, 2)

	# Normalize common OCR confusions (2↔Z, 6↔G) and enforce uppercase
	if tracking_no:
		tracking_no = tracking_no.upper()
		# Fix '12' misread for '1Z'
		if tracking_no.startswith("12"):
			tracking_no = "1Z" + tracking_no[2:]
	if shipment_no:
		shipment_no = shipment_no.upper()
		# Correct leading '6' misread for 'G' at position 0
		if shipment_no.startswith("6") and re.search(r"[A-Z]", shipment_no[1:]):
			shipment_no = "G" + shipment_no[1:]
		# Fix '12' misread for '1Z' when present at the start
		if shipment_no.startswith("12"):
			shipment_no = "1Z" + shipment_no[2:]
	if ref1:
		ref1 = ref1.upper()
		# Correct leading '6' misread for 'G' at position 0 for shipment_no output
		if ref1.startswith("6") and re.search(r"[A-Z0-9]", ref1[1:]):
			ref1 = "G" + ref1[1:]
		if ref1.startswith("12"):
			ref1 = "1Z" + ref1[2:]
	if ref2:
		ref2 = ref2.upper()

	return {
		"weight_container": weight_container_str,
		"reference_no_1": shipment_no,  # swapped per user: ref1 is actually shipment_no label
		"reference_no_2": ref2 or searched_reference,
		"tracking_no": tracking_no,
		"shipment_no": ref1,  # swapped per user: shipment_no field should show former ref1
		"description": description,
		"net_charges": net_charges,
		"freight": freight,
		"fuel_surcharge": fuel_surcharge,
		"total_charges_for_shipment": total_charges_for_shipment,
	}


def _reocr_numeric_region(
	image: Image.Image,
	line_bbox: Optional[Tuple[int, int, int, int]],
	search_window_px: int = 180,
) -> Optional[float]:

	if not line_bbox:
		return None
	# Expand bbox to the right where amounts typically appear
	l, t, r, b = line_bbox
	w, h = image.size
	region = (r, max(0, t - 4), min(w, r + search_window_px), min(h, b + 4))
	if region[0] >= region[2] or region[1] >= region[3]:
		return None
	crop = image.crop(region)
	config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.,-"
	text = pytesseract.image_to_string(crop, config=config).strip()
	return _parse_amount(text)


def _reocr_line_weight_pair(
	image: Image.Image,
	line_bbox: Optional[Tuple[int, int, int, int]],
	pad_px: int = 12,
) -> Optional[Tuple[Optional[float], Optional[float], bool]]:

	if not line_bbox:
		return None
	l, t, r, b = line_bbox
	w, h = image.size
	region = (max(0, l - pad_px), max(0, t - pad_px), min(w, r + pad_px), min(h, b + pad_px))
	if region[0] >= region[2] or region[1] >= region[3]:
		return None
	crop = image.crop(region)

	# Prepare multiple variants: raw, binary, inverse-binary
	arr = np.array(crop)
	if arr.ndim == 3 and arr.shape[2] == 3:
		gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
	else:
		gray = arr
	_, bin_norm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	_, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	candidates = [crop, Image.fromarray(bin_norm), Image.fromarray(bin_inv)]

	configs = [
		"--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789./Cc -c classify_bln_numeric_mode=1",
		"--oem 1 --psm 13 -c tessedit_char_whitelist=0123456789./Cc -c classify_bln_numeric_mode=1",
	]
	for variant in candidates:
		for cfg in configs:
			text = pytesseract.image_to_string(variant, config=cfg).strip()
			m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?:\s*([Cc]))?", text)
			if m:
				try:
					return float(m.group(1)), float(m.group(2)), bool(m.group(3))
				except ValueError:
					pass
	return None


def _snap_actual_weight_to_half_step(
	actual: Optional[float],
	chargeable: Optional[float],
	max_frac_delta: float = 0.2,
) -> Optional[float]:

	# Only correct common OCR error: .8 misread where it should be .5
	if actual is None or chargeable is None:
		return actual
	frac_c = round(chargeable - math.floor(chargeable), 2)
	frac_a = round(actual - math.floor(actual), 2)
	# If chargeable shows a .50 step and actual looks like ~.80, snap actual to .50
	if abs(frac_c - 0.50) <= 0.05 and abs(frac_a - 0.80) <= max_frac_delta:
		return math.floor(actual) + 0.5
	return actual


def _snap_to_half_step_down(value: Optional[float], max_delta: float = 0.35) -> Optional[float]:

	# No longer used for chargeable; retained for potential future tuning
	if value is None:
		return value
	lower = math.floor(value * 2.0) / 2.0
	if (value - lower) <= max_delta:
		return lower
	return value


def process_pdf_for_reference(
	pdf_path: str,
	reference: str,
	tesseract_cmd: Optional[str] = None,
	scale: float = 2.0,
	lang: str = "eng",
	context_window: int = 2,
) -> Dict:

	# If Tesseract is available, run the full OCR flow (preferred)
	ensure_tesseract_cmd(tesseract_cmd)
	if is_tesseract_available():
		images = render_pdf_to_images(pdf_path, scale=scale)
		all_results: List[Dict] = []
		for page_number, pil_image in enumerate(images, start=1):
			processed = preprocess_image_for_ocr(pil_image)
			data = ocr_image_to_data(processed, lang=lang)
			lines = build_lines_from_tesseract_data(data)
			for ln in lines:
				ln["_page_image_bin"] = processed
				ln["_page_image_raw"] = pil_image

			matches = find_reference_matches(lines, reference)
			if not matches:
				continue

			for _, idx in matches:
				context_lines = extract_context(lines, idx, window=context_window)
				key_values = extract_key_values_from_lines(context_lines)
				fields = extract_ups_structured_fields(context_lines)
				required = extract_requested_fields(lines, idx, reference)
				fields.pop("sender", None)
				fields.pop("consignee", None)
				key_values = {k: v for k, v in key_values.items() if k.lower() not in ("sender", "consignee")}

				all_results.append(
					{
						"page": page_number,
						"matched_line": lines[idx]["text"],
						"requested": required,
						"key_values": key_values,
					}
				)

		return {
			"reference": reference,
			"pdf": os.path.abspath(pdf_path),
			"matches": all_results,
		}

	# Fallback: no Tesseract available — parse embedded PDF text only
	pages_lines = extract_pdf_text_lines(pdf_path)
	all_results: List[Dict] = []
	for page_number, lines in enumerate(pages_lines, start=1):
		matches = find_reference_matches(lines, reference)
		if not matches:
			continue
		for _, idx in matches:
			context_lines = extract_context(lines, idx, window=context_window)
			key_values = extract_key_values_from_lines(context_lines)
			fields = extract_ups_structured_fields(context_lines)
			# Use the same text-based extraction heuristics to populate requested fields
			requested = extract_requested_fields(lines, idx, reference)
			fields.pop("sender", None)
			fields.pop("consignee", None)
			key_values = {k: v for k, v in key_values.items() if k.lower() not in ("sender", "consignee")}

			all_results.append(
				{
					"page": page_number,
					"matched_line": lines[idx]["text"],
					"requested": requested,
					"key_values": key_values,
				}
			)

	return {
		"reference": reference,
		"pdf": os.path.abspath(pdf_path),
		"matches": all_results,
	}


def main() -> None:

	parser = argparse.ArgumentParser(
		description=(
			"OCR a PDF invoice, search for a reference number, and output details as JSON."
		)
	)
	parser.add_argument(
		"--pdf",
		required=True,
		help="Path to the invoice PDF (e.g., ups_invoice.pdf)",
	)
	parser.add_argument(
		"--ref",
		required=True,
		help="Reference number to search for",
	)
	parser.add_argument(
		"--tesseract-cmd",
		required=False,
		help=(
			"Full path to tesseract.exe on Windows if not in PATH "
			"(e.g., C:/Program Files/Tesseract-OCR/tesseract.exe)"
		),
	)
	parser.add_argument(
		"--scale",
		type=float,
		default=2.0,
		required=False,
		help="PDF render scale for OCR quality (2.0-3.0 recommended)",
	)
	parser.add_argument(
		"--lang",
		default="eng",
		required=False,
		help="Tesseract language codes (e.g., eng)",
	)
	parser.add_argument(
		"--context",
		type=int,
		default=2,
		required=False,
		help="Number of lines before/after the match to include",
	)
	parser.add_argument(
		"--output",
		required=False,
		help="Optional path to save JSON output. Prints to stdout if omitted.",
	)

	args = parser.parse_args()

	if not os.path.isfile(args.pdf):
		raise FileNotFoundError(f"PDF not found: {args.pdf}")

	result = process_pdf_for_reference(
		pdf_path=args.pdf,
		reference=args.ref,
		tesseract_cmd=args.tesseract_cmd,
		scale=args.scale,
		lang=args.lang,
		context_window=args.context,
	)

	json_text = json.dumps(result, indent=2, ensure_ascii=False)
	if args.output:
		with open(args.output, "w", encoding="utf-8") as f:
			f.write(json_text)
		print(f"Saved results to {os.path.abspath(args.output)}")
	else:
		print(json_text)


if __name__ == "__main__":
	main()



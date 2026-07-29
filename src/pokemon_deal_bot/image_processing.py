from __future__ import annotations

import io
import math
from dataclasses import dataclass

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass(slots=True)
class PreparedImage:
    label: str
    jpeg: bytes


def _jpeg(image: Image.Image, max_dimension: int, quality: int) -> bytes:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if max(image.size) > max_dimension:
        scale = max_dimension / max(image.size)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


async def download_listing_images(urls: list[str], *, maximum: int, timeout: float = 45.0) -> list[Image.Image]:
    images: list[Image.Image] = []
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as client:
        for url in urls[:maximum]:
            try:
                response = await client.get(url)
                response.raise_for_status()
                images.append(Image.open(io.BytesIO(response.content)).convert("RGB"))
            except Exception:
                continue
    return images


def make_contact_sheet(images: list[Image.Image], *, prefix: str, max_dimension: int, quality: int, columns: int = 2) -> PreparedImage:
    if not images:
        raise ValueError("Cannot create a contact sheet without images")
    thumb_width = max_dimension // max(1, columns)
    inner_width = max(80, thumb_width - 16)
    inner_height = max(120, int(inner_width * 1.35))
    prepared: list[Image.Image] = []
    for image in images:
        copy = ImageOps.exif_transpose(image).convert("RGB")
        scale = min(inner_width / copy.width, inner_height / copy.height)
        # Marketplace thumbnails are often only 240 px. Upscaling them inside
        # the contact sheet gives Gemini a materially larger view of the art.
        target = (
            max(1, int(copy.width * scale)),
            max(1, int(copy.height * scale)),
        )
        copy = copy.resize(target, Image.Resampling.LANCZOS)
        prepared.append(copy)
    rows = math.ceil(len(prepared) / columns)
    cell_height = max((image.height for image in prepared), default=inner_height) + 38
    canvas = Image.new("RGB", (max_dimension, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, image in enumerate(prepared):
        row, column = divmod(index, columns)
        x = column * thumb_width + (thumb_width - image.width) // 2
        y = row * cell_height + 28
        canvas.paste(image, (x, y))
        draw.text((column * thumb_width + 8, row * cell_height + 6), f"{prefix}{index + 1}", fill="black")
    return PreparedImage(label=prefix, jpeg=_jpeg(canvas, max_dimension, quality))


def _order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    rect[1] = points[np.argmin(diffs)]
    rect[3] = points[np.argmax(diffs)]
    return rect


def _warp(image: np.ndarray, points: np.ndarray) -> Image.Image:
    rect = _order_points(points.astype("float32"))
    tl, tr, br, bl = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width > height:
        width, height = height, width
    width, height = max(width, 120), max(height, 170)
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))


def extract_card_crops(images: list[Image.Image], *, maximum: int = 40) -> list[Image.Image]:
    crops: list[Image.Image] = []
    hashes: list[np.ndarray] = []
    for pil_image in images:
        rgb = np.array(pil_image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        scale = min(1.0, 1800 / max(bgr.shape[:2]))
        working = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1 else bgr
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 140)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        image_area = working.shape[0] * working.shape[1]
        candidates: list[tuple[float, np.ndarray]] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * 0.008 or area > image_area * 0.92:
                continue
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            points = polygon.reshape(4, 2)
            x, y, width, height = cv2.boundingRect(points)
            ratio = min(width, height) / max(width, height)
            if not 0.52 <= ratio <= 0.82:
                continue
            candidates.append((area, points))
        for _, points in sorted(candidates, reverse=True, key=lambda item: item[0]):
            crop = _warp(working, points)
            tiny = np.array(crop.resize((16, 16)).convert("L"))
            digest = tiny > tiny.mean()
            if any(np.count_nonzero(digest != existing) <= 10 for existing in hashes):
                continue
            hashes.append(digest)
            crops.append(crop)
            if len(crops) >= maximum:
                return crops
    return crops


def image_file_bytes(path, *, max_dimension: int = 1600, quality: int = 88) -> bytes:
    with Image.open(path) as image:
        return _jpeg(image, max_dimension, quality)

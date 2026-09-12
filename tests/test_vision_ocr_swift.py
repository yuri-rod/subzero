import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("swift"), reason="Requires Apple CoreGraphics")
def test_native_caption_ink_excludes_bright_background_and_colored_lettering(tmp_path):
    source = Path(__file__).parents[1] / "src/subzero/vision_ocr.swift"
    native = source.read_text().split("func processImage(", 1)[0]
    probe = tmp_path / "caption_pixels.swift"
    probe.write_text(native + r'''
func measure(background: [CGFloat], fill: [CGFloat], box: CGRect) -> Double {
    let context = CGContext(data: nil, width: 100, height: 100, bitsPerComponent: 8,
                            bytesPerRow: 400, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setFillColor(red: background[0], green: background[1], blue: background[2], alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: 100, height: 100))
    context.setFillColor(red: fill[0], green: fill[1], blue: fill[2], alpha: 1)
    context.fill(CGRect(x: 30, y: 30, width: 40, height: 40))
    return CaptionPixels(image: context.makeImage()!)!.ink(in: box)
}
func outlinedGlyph() -> Double {
    let context = CGContext(data: nil, width: 100, height: 100, bitsPerComponent: 8,
                            bytesPerRow: 400, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    context.setFillColor(gray: 1, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: 100, height: 100))
    context.setLineWidth(6)
    context.setStrokeColor(gray: 0, alpha: 1)
    context.beginPath()
    context.move(to: CGPoint(x: 30, y: 30))
    context.addLine(to: CGPoint(x: 30, y: 70))
    context.addLine(to: CGPoint(x: 70, y: 70))
    context.move(to: CGPoint(x: 30, y: 50))
    context.addLine(to: CGPoint(x: 62, y: 50))
    context.move(to: CGPoint(x: 30, y: 30))
    context.addLine(to: CGPoint(x: 70, y: 30))
    context.replacePathWithStrokedPath()
    context.setLineWidth(2)
    context.drawPath(using: .fillStroke)
    return CaptionPixels(image: context.makeImage()!)!.ink(in: CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5))
}
let box = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
let readings = [
    "whiteFill": measure(background: [0, 0, 0], fill: [1, 1, 1], box: box),
    "coloredFill": measure(background: [0, 0, 0], fill: [0.1, 0.5, 1], box: box),
    "brightBackground": measure(background: [1, 1, 1], fill: [0.1, 0.5, 1], box: box),
    "outsideImage": measure(background: [1, 1, 1], fill: [1, 1, 1], box: CGRect(x: 2, y: 2, width: 1, height: 1)),
    "outlinedGlyphOnWhite": outlinedGlyph()
]
let encoded = try JSONSerialization.data(withJSONObject: readings)
print(String(data: encoded, encoding: .utf8)!)
''')
    completed = subprocess.run(["swift", str(probe)], capture_output=True, text=True, timeout=60, check=True)
    readings = json.loads(completed.stdout)
    assert readings["whiteFill"] == pytest.approx(1600 / 2500)
    assert readings["coloredFill"] == 0
    assert readings["brightBackground"] == 0
    assert readings["outsideImage"] == 0
    assert readings["outlinedGlyphOnWhite"] >= 0.04

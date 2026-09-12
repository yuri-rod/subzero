import Foundation
import Vision
import CoreGraphics
import ImageIO

struct OCRCandidate: Codable {
    let text: String
    let confidence: Float
}

struct OCRItem: Codable {
    let text: String
    let confidence: Float
    let x: Double
    let y: Double
    let width: Double
    let height: Double
    let angle: Double
    let candidates: [OCRCandidate]
}

struct OCRResult: Codable {
    let file: String
    let items: [OCRItem]
    let subtitleText: String
    var error: String? = nil
}

func processImage(path: String, captionRegion: Bool = false) -> OCRResult {
    let url = URL(fileURLWithPath: path)
    guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
        return OCRResult(file: path, items: [], subtitleText: "", error: "Cannot decode image")
    }

    let request = VNRecognizeTextRequest()
    let roi = captionRegion ? CGRect(x: 0, y: 0.07, width: 1, height: 0.19)
        : CGRect(x: 0, y: 0, width: 1, height: 1)
    request.regionOfInterest = roi
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.recognitionLanguages = ["en-US"]

    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    do {
        try handler.perform([request])
        guard let observations = request.results else {
            return OCRResult(file: path, items: [], subtitleText: "")
        }

        var items: [OCRItem] = []
        var subtitleLines: [(y: Double, text: String)] = []

        for obs in observations {
            let candidates = obs.topCandidates(3)
            guard let candidate = candidates.first else { continue }
            let str = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
            if str.isEmpty { continue }

            let box = obs.boundingBox
            let item = OCRItem(
                text: str,
                confidence: candidate.confidence,
                x: roi.origin.x + Double(box.origin.x) * roi.width,
                y: roi.origin.y + Double(box.origin.y) * roi.height,
                width: Double(box.size.width) * roi.width,
                height: Double(box.size.height) * roi.height,
                angle: atan2(Double(obs.topRight.y - obs.topLeft.y) * roi.height,
                             Double(obs.topRight.x - obs.topLeft.x) * roi.width) * 180 / .pi,
                candidates: candidates.map { OCRCandidate(text: $0.string, confidence: $0.confidence) }
            )
            items.append(item)

            if item.y <= 0.40 {
                subtitleLines.append((y: item.y, text: str))
            }
        }

        subtitleLines.sort { $0.y > $1.y }
        let joinedSubtitles = subtitleLines.map { $0.text }.joined(separator: "\n")

        return OCRResult(file: path, items: items, subtitleText: joinedSubtitles)
    } catch {
        return OCRResult(file: path, items: [], subtitleText: "", error: error.localizedDescription)
    }
}

let args = Array(CommandLine.arguments.dropFirst())
if args.isEmpty {
    fputs("Usage: vision_ocr [--json] [--caption-region] <image_paths...>\n", stderr)
    exit(1)
}

var asJson = false
var captionRegion = false
var files: [String] = []

for arg in args {
    if arg == "--json" {
        asJson = true
    } else if arg == "--caption-region" {
        captionRegion = true
    } else {
        files.append(arg)
    }
}

var results: [OCRResult] = []
for file in files {
    let res = processImage(path: file, captionRegion: captionRegion)
    results.append(res)
    if let failure = res.error {
        fputs("Vision OCR failed for \(file): \(failure)\n", stderr)
    }
    if !asJson {
        if !res.subtitleText.isEmpty {
            print("[\(file)]:\n\(res.subtitleText)\n")
        }
    }
}

if asJson {
    let encoder = JSONEncoder()
    encoder.outputFormatting = .prettyPrinted
    if let data = try? encoder.encode(results), let str = String(data: data, encoding: .utf8) {
        print(str)
    }
}
if results.contains(where: { $0.error != nil }) {
    exit(1)
}

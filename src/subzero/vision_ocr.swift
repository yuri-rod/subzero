import Foundation
import Vision
import CoreGraphics
import ImageIO

struct OCRItem: Codable {
    let text: String
    let confidence: Float
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

struct OCRResult: Codable {
    let file: String
    let items: [OCRItem]
    let subtitleText: String
    var error: String? = nil
}

func processImage(path: String) -> OCRResult {
    let url = URL(fileURLWithPath: path)
    guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
        return OCRResult(file: path, items: [], subtitleText: "", error: "Cannot decode image")
    }

    let request = VNRecognizeTextRequest()
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
            guard let candidate = obs.topCandidates(1).first else { continue }
            let str = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
            if str.isEmpty { continue }

            let box = obs.boundingBox
            let item = OCRItem(
                text: str,
                confidence: candidate.confidence,
                x: Double(box.origin.x),
                y: Double(box.origin.y),
                width: Double(box.size.width),
                height: Double(box.size.height)
            )
            items.append(item)

            if box.origin.y <= 0.40 {
                subtitleLines.append((y: Double(box.origin.y), text: str))
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
    fputs("Usage: vision_ocr [--json] <image_paths...>\n", stderr)
    exit(1)
}

var asJson = false
var files: [String] = []

for arg in args {
    if arg == "--json" {
        asJson = true
    } else {
        files.append(arg)
    }
}

var results: [OCRResult] = []
for file in files {
    let res = processImage(path: file)
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

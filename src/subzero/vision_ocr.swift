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
    let captionInk: Double?
    let candidates: [OCRCandidate]
}

struct OCRResult: Codable {
    let file: String
    let items: [OCRItem]
    let subtitleText: String
    var error: String? = nil
}

struct CaptionPixels {
    let width: Int
    let height: Int
    let pixels: [UInt8]

    init?(image: CGImage) {
        let width = image.width, height = image.height
        var pixels = [UInt8](repeating: 0, count: width * height * 4)
        let prepared = pixels.withUnsafeMutableBytes { storage -> Bool in
            guard let ctx = CGContext(data: storage.baseAddress, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard prepared else { return nil }
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    func ink(in box: CGRect) -> Double {
        let left = max(0, Int(box.minX * Double(width)))
        let right = min(width, Int(box.maxX * Double(width)))
        let top = max(0, Int((1 - box.maxY) * Double(height)))
        let bottom = min(height, Int((1 - box.minY) * Double(height)))
        let columns = right - left, rows = bottom - top
        guard columns > 0, rows > 0 else { return 0 }
        var mask = [UInt8](repeating: 0, count: columns * rows)
        var queue: [Int] = []
        for row in 0..<rows {
            for column in 0..<columns {
                let offset = ((row + top) * width + column + left) * 4
                let red = pixels[offset], green = pixels[offset + 1], blue = pixels[offset + 2]
                let low = min(red, green, blue), high = max(red, green, blue)
                guard low >= 180, Int(high) - Int(low) <= 35 else { continue }
                let index = row * columns + column
                mask[index] = 1
                if row == 0 || row == rows - 1 || column == 0 || column == columns - 1 {
                    mask[index] = 2
                    queue.append(index)
                }
            }
        }
        var next = 0
        while next < queue.count {
            let index = queue[next]
            next += 1
            let row = index / columns, column = index % columns
            for (r, c) in [(row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)]
                where r >= 0 && r < rows && c >= 0 && c < columns {
                let offset = r * columns + c
                if mask[offset] == 1 {
                    mask[offset] = 2
                    queue.append(offset)
                }
            }
        }
        // Exclude bright scenery connected to the box edge, keeping enclosed caption fill.
        return Double(mask.reduce(0) { $0 + ($1 == 1 ? 1 : 0) }) / Double(mask.count)
    }
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
        guard let pixels = CaptionPixels(image: cgImage) else {
            return OCRResult(file: path, items: [], subtitleText: "", error: "Cannot prepare caption pixels")
        }

        var items: [OCRItem] = []
        var subtitleLines: [(y: Double, text: String)] = []

        for obs in observations {
            let candidates = obs.topCandidates(3)
            guard let candidate = candidates.first else { continue }
            let str = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
            if str.isEmpty { continue }

            let box = obs.boundingBox
            let mapped = CGRect(x: roi.origin.x + box.origin.x * roi.width,
                                y: roi.origin.y + box.origin.y * roi.height,
                                width: box.width * roi.width, height: box.height * roi.height)
            let item = OCRItem(
                text: str,
                confidence: candidate.confidence,
                x: roi.origin.x + Double(box.origin.x) * roi.width,
                y: roi.origin.y + Double(box.origin.y) * roi.height,
                width: Double(box.size.width) * roi.width,
                height: Double(box.size.height) * roi.height,
                angle: atan2(Double(obs.topRight.y - obs.topLeft.y) * roi.height,
                             Double(obs.topRight.x - obs.topLeft.x) * roi.width) * 180 / .pi,
                captionInk: mapped.minY <= 0.40 ? pixels.ink(in: mapped) : nil,
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

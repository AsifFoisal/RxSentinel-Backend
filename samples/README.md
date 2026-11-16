# samples/ — real photos for the intake/OCR route (Phase 3)

Drop prescription photos here (one image per file). The CLI accepts any
path, but keeping them here makes demos repeatable.

## Anonymisation first

Before adding a photo, blur or crop out: patient name, address, phone
number, doctor registration number. Keep the medicine lines, expiry and
batch — those are what the pipeline reads.

## Running the pipeline

    # printed prescription (needs the heavy OCR install, once):
    pip install paddlepaddle paddleocr

    python intake.py --image samples/<photo>.jpg
    python intake.py --image samples/<photo>.jpg --assess --mongo-uri "mongodb+srv://..."

    # handwritten prescription (vision model, no PaddleOCR needed):
    python intake.py --image samples/<photo>.jpg --handwritten
    python intake.py --image samples/<photo>.jpg --handwritten --vision-backend gemini

    # synthetic printed fixture (no real photo needed):
    python intake.py --fixture samples/printed_demo.png

## Notes

- First PaddleOCR run downloads its detection/recognition models —
  expect a one-time wait.
- `--lang bang` for printed Bangla text; handwritten Bangla goes through
  the vision route, where accuracy is honestly limited (handover, gap
  table) — uncertain items land in `please_confirm`, never in the check.
- The vision backend follows web_verify's provider precedence: your
  `LLM_*` provider first, native Gemini via `--vision-backend gemini`.

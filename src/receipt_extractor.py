import json, os
from pathlib import Path
from pydantic import BaseModel
import lmstudio as lms

MODEL = os.environ["LM_STUDIO_MODEL"]
IMAGES = sorted(Path("data/raw").glob("*.png"))


class Item(BaseModel):
    nama: str
    qty: float
    harga: float


class Receipt(BaseModel):
    merchant: str
    tanggal: str
    item: list[Item]
    subtotal: float
    pajak: float
    total: float


model = lms.llm(MODEL)
results = []

for path in IMAGES:
    print(f"Memproses {path.name} ...")
    image = lms.prepare_image(str(path))
    chat = lms.Chat()
    chat.add_user_message(
        "Baca nota. Ekstrak merchant, tanggal, item (nama, qty, harga), "
        "subtotal, pajak, dan total. Jika pajak tidak terlihat, isi 0. "
        "Jangan mengarang. Angka tanpa pemisah ribuan.",
        images=[image],
    )
    prediction = model.respond(chat, response_format=Receipt)
    data = dict(prediction.parsed)
    data["file"] = path.name
    results.append(data)

Path("reports/receipts.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(results, indent=2, ensure_ascii=False))

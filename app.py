from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model & tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("./flan-t5-toefl1")
tokenizer = AutoTokenizer.from_pretrained("./flan-t5-toefl1")

app = Flask(__name__)

# input dari web : 
#     {
#     "passage": "Isi passage di sini...",
#     "question": "Which step is about studying hard?",
#     "options": ["The first step.", "The second step.", "The third step.", "The last step."],
#     "answer": "C"
#     }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        passage = data["passage"]
        question = data["question"]
        raw_options = data["options"]
        answer = data["answer"]

        # Tambahkan label A., B., C., ...
        labeled_options = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(raw_options)]
        options_text = " | ".join(labeled_options)

        input_text = f"""
                        Passage: {passage}
                        Question: {question}
                        Options: {options_text}
                        Answer: {answer}
                        Instruction: Berikan penjelasan dalam Bahasa Indonesia mengapa jawaban berikut benar.
                    """

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)

        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            num_beams=4
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return jsonify({"explanation": output_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

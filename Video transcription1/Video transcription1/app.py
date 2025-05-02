from flask import Flask, render_template, send_file, jsonify
from flask_sock import Sock
from flask_sqlalchemy import SQLAlchemy
import asyncio
import websockets
import threading
import json
import openai
import logging
import io

# Setup logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
sock = Sock(app)

# Config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///meetings.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# DB setup
db = SQLAlchemy(app)

class Meeting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    transcript = db.Column(db.Text, nullable=False)
    summary = db.Column(db.Text, nullable=False)

with app.app_context():
    db.create_all()

# Replace with your API keys
ASSEMBLYAI_API_KEY = "your_assemblyai_api_key"
openai.api_key = "your_openai_api_key"
ASSEMBLYAI_URL = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"

transcript_buffer = []

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/meetings")
def list_meetings():
    meetings = Meeting.query.all()
    return jsonify([{"id": m.id} for m in meetings])

@app.route("/download/<string:type>/<int:id>")
def download(type, id):
    meeting = Meeting.query.get_or_404(id)
    content = meeting.transcript if type == "transcript" else meeting.summary
    filename = f"{type}_{id}.txt"
    return send_file(
        io.BytesIO(content.encode("utf-8")),
        mimetype="text/plain",
        as_attachment=True,
        download_name=filename
    )

def generate_summary(text):
    prompt = (
        "You are an AI meeting assistant. Summarize this transcript into bullet points with key takeaways and action items:\n\n"
        f"{text}\n\nSummary:"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        summary = response.choices[0].message["content"]

        meeting = Meeting(transcript=text, summary=summary)
        db.session.add(meeting)
        db.session.commit()

        return summary
    except Exception as e:
        logging.error("OpenAI summary error: %s", str(e))
        return "Summary generation failed."

@sock.route("/transcribe")
def transcribe(ws):
    async def run():
        logging.info("Transcription WebSocket connected")
        try:
            async with websockets.connect(
                ASSEMBLYAI_URL,
                extra_headers={"Authorization": ASSEMBLYAI_API_KEY}
            ) as aai_ws:

                await aai_ws.send(json.dumps({"type": "start", "sample_rate": 16000}))

                async def send_audio():
                    while True:
                        data = ws.receive()
                        if data is None:
                            break
                        await aai_ws.send(data)

                async def receive_transcripts():
                    while True:
                        message = await aai_ws.recv()
                        msg = json.loads(message)
                        if msg.get("message_type") == "FinalTranscript":
                            text = msg["text"]
                            if text.strip():
                                transcript_buffer.append(text)
                                logging.info("Transcript: %s", text)
                                ws.send(json.dumps({"text": text}))

                                if len(transcript_buffer) % 5 == 0:
                                    full_text = " ".join(transcript_buffer)
                                    summary = generate_summary(full_text)
                                    ws.send(json.dumps({"summary": summary}))

                await asyncio.gather(send_audio(), receive_transcripts())

        except Exception as e:
            logging.error("Transcription error: %s", str(e))
            ws.send(json.dumps({"error": str(e)}))

    threading.Thread(target=asyncio.run, args=(run(),)).start()

if __name__ == "__main__":
    app.run(debug=True)

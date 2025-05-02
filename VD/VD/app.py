import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
from PIL import Image, ImageTk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import pyttsx3
import nltk
import os
import requests
import time
import threading

# Download required NLTK data
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

download_nltk_data()

class TextSummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Text Summarization")
        self.summary = ""
        self.transcription = ""
        self.is_transcribing = False
        
        # Initialize text-to-speech engine
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
        except Exception as e:
            messagebox.showwarning("TTS Error", f"Text-to-speech initialization failed: {str(e)}")
            self.engine = None
        
        # Default audio URL
        self.audio_url = "https://assembly.ai/wildfires.mp3"
        
        self.create_widgets()
        
    def create_widgets(self):
        # API Key Frame
        api_frame = tk.Frame(self.root)
        api_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        
        tk.Label(api_frame, text="AssemblyAI API Key:").pack(side="left", padx=(0, 5))
        self.api_key_entry = tk.Entry(api_frame, width=40, show="*")
        self.api_key_entry.pack(side="left", expand=True, fill="x")
        
        # Audio URL Frame
        url_frame = tk.Frame(self.root)
        url_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        
        tk.Label(url_frame, text="Audio URL:").pack(side="left", padx=(0, 5))
        self.url_entry = tk.Entry(url_frame, width=60)
        self.url_entry.pack(side="left", expand=True, fill="x")
        self.url_entry.insert(0, self.audio_url)
        
        # Text input
        tk.Label(self.root, text="Enter Text or Transcribe Audio:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.text_input = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=15)
        self.text_input.grid(row=3, column=0, columnspan=3, padx=10, pady=5)
        
        # Insert sample text
        sample_text = """This application combines speech-to-text transcription with automatic text summarization. 
        It can process audio files from URLs and generate concise summaries of the content."""
        self.text_input.insert(tk.END, sample_text.strip())
        
        # Transcribe button
        transcribe_button = tk.Button(self.root, text="Transcribe Audio", command=self.start_transcription,
                                    bg="#FF9800", fg="white", padx=15, pady=5)
        transcribe_button.grid(row=4, column=0, padx=10, pady=5, sticky="w")
        
        # Sentence count input
        tk.Label(self.root, text="Summary Length (sentences):").grid(row=4, column=1, padx=10, pady=5, sticky="w")
        self.sentence_count_input = tk.Spinbox(self.root, from_=1, to=100, width=5)
        self.sentence_count_input.grid(row=4, column=1, padx=(150, 10), pady=5, sticky="w")
        self.sentence_count_input.delete(0, tk.END)
        self.sentence_count_input.insert(0, "5")
        
        # Summarize button
        summarize_button = tk.Button(self.root, text="Summarize", command=self.summarize_text,
                                   bg="#4CAF50", fg="white", padx=20, pady=5)
        summarize_button.grid(row=4, column=2, padx=10, pady=5, sticky="e")
        
        # Summary output
        tk.Label(self.root, text="Summary:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.summary_output = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=10)
        self.summary_output.grid(row=6, column=0, columnspan=3, padx=10, pady=5)
        
        # Action buttons frame
        action_frame = tk.Frame(self.root)
        action_frame.grid(row=7, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        
        # Clear buttons
        clear_input_button = tk.Button(action_frame, text="Clear Input", command=self.clear_input,
                                     bg="#f44336", fg="white", padx=15, pady=5)
        clear_input_button.pack(side="left", padx=5)
        
        clear_output_button = tk.Button(action_frame, text="Clear Output", command=self.clear_output,
                                      bg="#f44336", fg="white", padx=15, pady=5)
        clear_output_button.pack(side="left", padx=5)
        
        # Download Summary button
        download_button = tk.Button(action_frame, text="Download Summary", command=self.download_summary,
                                  bg="#607D8B", fg="white", padx=15, pady=5)
        download_button.pack(side="left", padx=5)
        
        # Speak buttons
        speak_transcription_button = tk.Button(action_frame, text="Speak Text", command=self.speak_transcription,
                                             bg="#9C27B0", fg="white", padx=15, pady=5)
        speak_transcription_button.pack(side="right", padx=5)
        
        speak_summary_button = tk.Button(action_frame, text="Speak Summary", command=self.speak_summary,
                                       bg="#2196F3", fg="white", padx=15, pady=5)
        speak_summary_button.pack(side="right", padx=5)
        
        # Q&A Frame
        qa_frame = tk.Frame(self.root)
        qa_frame.grid(row=8, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        
        tk.Label(qa_frame, text="Ask a question about the summary:").pack(side="left", padx=(0, 5))
        self.question_entry = tk.Entry(qa_frame, width=50)
        self.question_entry.pack(side="left", expand=True, fill="x", padx=(0, 5))
        
        ask_button = tk.Button(qa_frame, text="Ask", command=self.answer_question,
                             bg="#FFC107", fg="black", padx=15, pady=5)
        ask_button.pack(side="right")
        
        # Answer output
        self.answer_output = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=5)
        self.answer_output.grid(row=9, column=0, columnspan=3, padx=10, pady=5)
        self.answer_output.config(state=tk.DISABLED)
    
    def start_transcription(self):
        if self.is_transcribing:
            messagebox.showinfo("Transcription", "Transcription already in progress.")
            return
            
        threading.Thread(target=self.transcribe_audio, daemon=True).start()
    
    def transcribe_audio(self):
        try:
            self.is_transcribing = True
            api_key = self.api_key_entry.get().strip()
            audio_url = self.url_entry.get().strip()
            
            if not api_key:
                self.root.after(0, lambda: messagebox.showwarning("API Key Required", "Please enter your AssemblyAI API key."))
                return
                
            if not audio_url:
                self.root.after(0, lambda: messagebox.showwarning("Audio URL Required", "Please enter an audio URL to transcribe."))
                return
                
            self.root.after(0, lambda: self.summary_output.delete("1.0", tk.END))
            self.root.after(0, lambda: self.summary_output.insert(tk.END, "Transcribing audio... Please wait..."))
            
            # Submit transcription request
            headers = {"authorization": api_key, "content-type": "application/json"}
            response = requests.post(
                "https://api.assemblyai.com/v2/transcript",
                json={"audio_url": audio_url, "speaker_labels": True},
                headers=headers
            )
            
            if response.status_code != 200:
                raise Exception(f"API Error: {response.text}")
                
            transcript_id = response.json()['id']
            polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
            
            # Poll for results
            while True:
                polling_response = requests.get(polling_endpoint, headers=headers)
                if polling_response.status_code != 200:
                    raise Exception(f"API Error: {polling_response.text}")
                    
                status = polling_response.json()['status']
                
                if status == 'completed':
                    break
                elif status == 'error':
                    raise Exception(polling_response.json().get('error', 'Unknown error'))
                
                time.sleep(3)
            
            # Format transcription with speaker labels and timestamps
            transcription_text = ""
            utterances = polling_response.json().get('utterances', [])
            
            # Create a mapping from speaker numbers to letters (0 -> A, 1 -> B, etc.)
            speaker_mapping = {}
            current_speaker_letter = ord('A')
            
            for utterance in utterances:
                speaker_num = utterance['speaker']
                
                # If we haven't seen this speaker before, add them to the mapping
                if speaker_num not in speaker_mapping:
                    speaker_mapping[speaker_num] = chr(current_speaker_letter)
                    current_speaker_letter += 1
                    
                speaker_label = f"Speaker {speaker_mapping[speaker_num]}"
                
                # Format timestamps (convert from milliseconds to seconds)
                start_sec = utterance.get('start', 0) / 1000
                end_sec = utterance.get('end', 0) / 1000
                
                # Format as [00:00:00 - 00:00:00]
                start_time = time.strftime('%H:%M:%S', time.gmtime(start_sec))
                end_time = time.strftime('%H:%M:%S', time.gmtime(end_sec))
                
                # Calculate duration
                duration = end_sec - start_sec
                
                transcription_text += f"[{start_time} - {end_time}] {speaker_label} [Duration: {duration:.2f}s]:\n{utterance['text']}\n\n"
            
            self.transcription = transcription_text.strip()
            
            # Update UI
            self.root.after(0, lambda: self.text_input.delete("1.0", tk.END))
            self.root.after(0, lambda: self.text_input.insert(tk.END, self.transcription))
            self.root.after(0, lambda: self.summary_output.delete("1.0", tk.END))
            self.root.after(0, lambda: self.summary_output.insert(tk.END, "Transcription completed!"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Transcription Error", f"Error: {str(e)}"))
        finally:
            self.is_transcribing = False
    
    def summarize_text(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text to summarize.")
            return
            
        try:
            sentence_count = int(self.sentence_count_input.get())
            if sentence_count <= 0:
                messagebox.showwarning("Input Error", "Number of sentences must be greater than 0.")
                return
                
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LsaSummarizer(Stemmer("english"))
            summarizer.stop_words = get_stop_words("english")
            
            summary_sentences = summarizer(parser.document, sentence_count)
            self.summary = " ".join(str(sentence) for sentence in summary_sentences)
            
            self.summary_output.delete("1.0", tk.END)
            self.summary_output.insert(tk.END, self.summary)
            
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number of sentences.")
        except Exception as e:
            messagebox.showerror("Error", f"Summarization failed: {str(e)}")
    
    def download_summary(self):
        if not self.summary:
            messagebox.showwarning("No Summary", "No summary available to download.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            title="Save Summary As"
        )
        
        if file_path:
            try:
                with open(file_path, "w") as f:
                    f.write(self.summary)
                messagebox.showinfo("Success", "Summary downloaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
    
    def answer_question(self):
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showwarning("Question Required", "Please enter a question.")
            return
            
        if not self.summary:
            messagebox.showwarning("No Summary", "Please generate a summary first.")
            return
            
        # Simple Q&A implementation
        answer = self.simple_qa(self.summary, question)
        
        self.answer_output.config(state=tk.NORMAL)
        self.answer_output.delete("1.0", tk.END)
        self.answer_output.insert(tk.END, f"Q: {question}\n\nA: {answer}")
        self.answer_output.config(state=tk.DISABLED)
    
    def simple_qa(self, text, question):
        """Simple question answering implementation"""
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        question_lower = question.lower()
        
        # Check for direct matches
        if "who " in question_lower:
            # Look for names or people references
            sentences = nltk.sent_tokenize(text)
            for sent in sentences:
                if any(word in sent.lower() for word in ["mr.", "mrs.", "dr.", "professor"]):
                    return sent
                if any(word in sent.lower() for word in ["he", "she", "they"]):
                    return sent
            return "I couldn't find a specific person mentioned in the summary."
        
        elif "when " in question_lower:
            # Look for time references
            sentences = nltk.sent_tokenize(text)
            for sent in sentences:
                if any(word in sent.lower() for word in ["today", "yesterday", "tomorrow", "month", "year", "day"]):
                    return sent
                if any(word in sent.lower() for word in ["january", "february", "march", "april", "may", "june", 
                                                       "july", "august", "september", "october", "november", "december"]):
                    return sent
            return "I couldn't find a specific time mentioned in the summary."
        
        elif "where " in question_lower:
            # Look for location references
            sentences = nltk.sent_tokenize(text)
            for sent in sentences:
                if any(word in sent.lower() for word in ["in ", "at ", "on ", "near ", "city", "country", "place"]):
                    return sent
            return "I couldn't find a specific location mentioned in the summary."
        
        else:
            # Try to find a sentence that contains some question words
            sentences = nltk.sent_tokenize(text)
            question_words = set(question_lower.split())
            
            for sent in sentences:
                sent_words = set(sent.lower().split())
                if len(question_words & sent_words) > 2:  # At least 2 matching words
                    return sent
            
            return "I couldn't find a specific answer in the summary. The summary content may not contain the information you're looking for."
    
    def speak_summary(self):
        if not self.summary:
            messagebox.showwarning("No Summary", "Please generate a summary first.")
            return
            
        if self.engine:
            try:
                self.engine.say(self.summary)
                self.engine.runAndWait()
            except Exception as e:
                messagebox.showerror("TTS Error", f"Speech synthesis failed: {str(e)}")
        else:
            messagebox.showwarning("TTS Unavailable", "Text-to-speech engine is not available.")
    
    def speak_transcription(self):
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("No Text", "There is no text to speak.")
            return
            
        if self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                messagebox.showerror("TTS Error", f"Speech synthesis failed: {str(e)}")
        else:
            messagebox.showwarning("TTS Unavailable", "Text-to-speech engine is not available.")
    
    def clear_input(self):
        self.text_input.delete("1.0", tk.END)
        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, self.audio_url)
    
    def clear_output(self):
        self.summary_output.delete("1.0", tk.END)
        self.summary = ""
        self.answer_output.config(state=tk.NORMAL)
        self.answer_output.delete("1.0", tk.END)
        self.answer_output.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = TextSummarizerApp(root)
    root.mainloop()
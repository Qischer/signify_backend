from pathlib import Path
from openai import OpenAI

client = OpenAI()

def getCompleteSentece(arr):

    input = " ".join(arr);
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful ASL interpreter. You are given a list of words translated from sign language and you should produce a complete sentence."},
            {
                "role": "user",
                "content": input,
            }
        ],
    )

    s = stream.choices[0].message.content
    return s

def getSpeechAudio(s):
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
      model="tts-1",
      voice="alloy",
      input=s,
    )

    response.stream_to_file(speech_file_path)


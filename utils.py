from pathlib import Path
from openai import OpenAI

client = OpenAI()

def getCompleteSentece(arr):

    input = " ".join(arr);
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful ASL translator. You will be given a list of words that have been detected using an ASL detector. The words do not follow correct grammar. Use them to produce a complete sentence that makes sense. Only include the sentence and not any explanations."},
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


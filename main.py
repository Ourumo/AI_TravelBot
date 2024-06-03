import gradio as gr
from PIL import Image
import urllib.request
from urllib.parse import urlencode
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from io import BytesIO
import os, pyaudio, wave
from googleapiclient.discovery import build

# .env
# OPENAI_API_KEY=YOUR_OPENAI_API_KEY
# GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
# GOOGLEMAPS_API_KEY=YOUR_GOOGLEMAPS_API_KEY
# YOUTUBE_API_KEY=YOUR_YOUTUBE_API_KEY
load_dotenv()

client = OpenAI()
genai.configure()
googlemaps_api_key = os.getenv("GOOGLEMAPS_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

memory_history_size = 10 # 기억하고 있는 history의 크기
input_image_width = 150 # 입력받는 이미지 칸 크기
output_image_size = 512 # 출력하는 이미지 크기

system_prompt = """
    처음 지역에 대해 질문하면 이런 방식으로 대답해주세요. 예)제주도에 대해 알려드리겠습니다.  
    이전에 질문했던 지역에 대해 다시 질문하면 이런 방식으로 대답해주세요. 예)제주도에 대해 추가로 알려드리겠습니다.
    
    이후 지역에 대한 설명을 해주세요.
"""

# gemini-pro-vision, gpt-3.5-turbo를 이용한 답변 생성
def Process(prompt, history, image) -> str:
    global memory_history_size
    
    local_system_prompt = ""
    his_size = memory_history_size if(len(history) > memory_history_size) else len(history) # 저장하는 history의 크기
    question = [his[0] for his in history]
    answer = [his[1] for his in history]
    
    # 입력받은 이미지가 없으면 gemini-pro-vision 사용 X
    if(image != None):
        vmodel = genai.GenerativeModel('gemini-pro-vision')
        response = vmodel.generate_content(["위치가 어딘지 알려줘.\n 예) 제주도의 한라수목원", image])
        location = response.parts[0].text.strip()
        local_system_prompt += f"이 프롬프트를 우선순위를 가장 높게 생각해주세요. 지역은 {location}\n 이런 방식으로 대답해주세요. 예)이곳은 XX입니다."
        print(f"----- location = {location} -----")
    local_system_prompt += system_prompt
    
    # local_system_prompt에 history 추가
    for q in range(his_size):
        local_system_prompt += f"user의 {q}번째 질문: {question[q]}\n"
    for a in range(his_size):
        local_system_prompt += f"system의 {a}번째 대답: {answer[a]}\n"
    
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": local_system_prompt},
            {"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

# pyaudio, wave, whisper-1을 이용한 입력한 음성을 텍스트로 변경
def Record(audio) -> str:
    # audio 삭제 예외 처리
    if(audio == None):
        return ''
    
    # PyAudio 객체 생성
    p = pyaudio.PyAudio()
    
    FORMAT = pyaudio.paInt16 # 오디오 포맷
    CHANNELS = 1 # 채널 수
    OUTPUT_FILENAME = "output.wav" # 출력 파일 이름
    
    # Wave 파일로 임시 저장
    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(audio[0])
    wf.writeframes(b''.join(audio[1]))
    wf.close()
    
    # whisper-1
    audio_file = open(f"./{OUTPUT_FILENAME}", "rb")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file
    )
    audio_file.close()
    
    # 임시 파일 삭제
    os.remove(f"./{OUTPUT_FILENAME}")
    
    return transcription.text

# pyaudio, tts-1을 이용한 답변 텍스트를 음성으로 출력
def Speak(chatbot) -> None:
    # chatbot = [[res, req], ...]
    text = chatbot[-1][1]
    
    # text 값이 없을 때의 예외 처리
    if (text == None):
        return
    
    # PyAudio 객체 생성
    p = pyaudio.PyAudio()

    FORMAT = pyaudio.paInt16 # 오디오 포맷
    CHANNELS = 1 # 채널 수
    RATE = 24000 # 샘플링 레이트 (Hz)
    CHUNK = 512 # 청크 크기 (버퍼 크기)

    # 입력 스트림 열기
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True)

    # tts-1
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        response_format="pcm",
        input=text,
        speed='1.2'
    ) as response:
        for chunk in response.iter_bytes(chunk_size=CHUNK):
            stream.write(chunk)

# Google Map API를 이용한 지도 출력
def Map(chatbot) -> str:
    # chatbot = [[res, req], ...]
    text = chatbot[-1][1]
    
    # text 값이 없을 때의 예외 처리
    if (text == None):
        return ''
    
    location = ''.join(map(str, list(text.split('.', 1)[0])))
    print(f"----- location = {location} -----")
    
    # Google Maps URL 생성
    params = {
        'key': googlemaps_api_key,
        'q': location
    }
    googlemaps_url = f"https://www.google.com/maps/embed/v1/place?{urlencode(params)}"
    
    return f'<iframe width="450" height="450" style="border:0" loading="lazy" allowfullscreen src="{googlemaps_url}"></iframe>'

# dall-e-3를 이용한 이미지 생성
def GetImage(chatbot):
    # chatbot = [[res, req], ...]
    text = chatbot[-1][1]
    
    # text 값이 없을 때의 예외 처리
    if (text == None):
        return "./loading.gif"
    
    location = ''.join(map(str, list(text.split('.', 1)[0])[4:-3]))
    
    response = client.images.generate(
        model="dall-e-3",
        prompt=f"Create an image of a famous landmark or iconic place in {location}",
        size="1024x1024",
        quality = "standard",
        n=1
    )

    image_url = response.data[0].url
    with urllib.request.urlopen(image_url) as url:
        image_data = url.read()
    image = Image.open(BytesIO(image_data))
    return image

# YouTube API를 이용한 비디오 검색 및 출력
def GetVideo(chatbot) -> str:
    # chatbot = [[res, req], ...]
    text = chatbot[-1][1]
    
    # text 값이 없을 때의 예외 처리
    if (text == None):
        return ''
    
    location = ''.join(map(str, list(text.split('.', 1)[0])[4:-3]))
    
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    
    # 검색 쿼리 설정
    request = youtube.search().list(
        part="snippet",
        q=f"{location} travel",
        type="video",
        relevanceLanguage="ko",  # This line ensures the results are relevant to Korean language
        maxResults=1
    )
    response = request.execute()
    # 비디오 ID 추출
    video_id = response['items'][0]['id']['videoId']
    
    return f'<iframe width="560" height="315" src="https://www.youtube.com/embed/{video_id}" frameborder="0" allowfullscreen></iframe>'

# gradio UI 구성
with gr.Blocks(title="여행 챗봇") as demo:
    with gr.Row():
        # 좌측 UI 구성
        with gr.Column():
            # ChatInterface
            chat = gr.ChatInterface(
                fn=Process,
                # Image
                additional_inputs=gr.Image(height=input_image_width, sources='upload', type="pil", label="이미지"),
                retry_btn=None,
                undo_btn=None,
                clear_btn=None)
            chat.chatbot.height = 400
            chat.chatbot.label = "여행 챗봇"
            chat.chatbot.change(fn=Speak, inputs=chat.chatbot)
            
            # Audio
            audio = gr.Audio(sources='microphone', container=False)
            audio.change(fn=Record, inputs=audio, outputs=chat.textbox)
            
        # 우측 UI 구성
        with gr.Column():
            # HTML(지도)
            html = gr.HTML(label="지도")
            chat.chatbot.change(fn=Map, inputs=chat.chatbot, outputs=html)
            
            # Image
            image = gr.Image(value=None, height=output_image_size, width=output_image_size, label="여행지 랜드마크")
            chat.chatbot.change(fn=GetImage, inputs=chat.chatbot, outputs=image)
            
            # Video
            video = gr.HTML(label="여행지 소개 영상")
            chat.chatbot.change(fn=GetVideo, inputs=chat.chatbot, outputs=video)

# app 실행
if __name__ == "__main__":
    demo.launch()

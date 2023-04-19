import pytube
from pytube import YouTube
def Progressing(stream, chunk, remains):
    total = stream.filesize                    
    percent = (total-remains) / total * 100     
    print(f'下載中… {percent:05.2f}', end='\r')
while(True):
    url=input('請輸入YouTube影片網址:')
    try:
        yt=YouTube(url)
    except Exception:
        print('輸入錯誤的網址!')
        print('請重新輸入')
    else:
        break
filename=input('請輸入下載檔名:')
print('1.Mp4')
print('2.Mp3(僅音樂)')
while(True):
    mode=input('請選擇格式:')
    if mode=='1':  
        print('download...')
        yt = YouTube(url, on_progress_callback=Progressing)
        yt.streams.filter().get_highest_resolution().download(filename=filename+'.mp4')
        print()
        print(filename+'.mp4 下載完成!!')
        break
    elif mode=='2':
        print('download...')
        yt = YouTube(url, on_progress_callback=Progressing)
        yt.streams.filter().get_audio_only().download(filename=filename+'.mp3')
        print()
        print(filename+'.mp3 下載完成!!')
        break
    else:
        print('請重新輸入!')
exit()


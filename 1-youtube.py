from yt_dlp import YoutubeDL
print('動画url')
url = input()


ydl_opts = {'outtmpl':'J:/musicgame/down_load_video/%(title)s.mp4',
            'format': 'bestvideo',}
with YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])
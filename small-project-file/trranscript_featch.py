from modules.youtube import fetch_youtube_transcript, extract_youtube_video_id
from modules.helpers import save_response_as_file
from modules.helpers import num_tokens_from_string

video_url = "https://youtu.be/qe6dSDq5GV0?si=wN13pRWZPqzeyrCz"
video_id = extract_youtube_video_id(video_url)
transcript = fetch_youtube_transcript(video_url)

from modules.youtube import get_video_metadata

meta = get_video_metadata(video_url)
video_title = meta['name']
print(video_title)
save_response_as_file("../transcripts", video_title, transcript)

video = Video.create(
    yt_video_id = video_id,
    title = video_title,
    channel = meta['channel']
)

from datetime import datetime

Video.update({Video.saved_on: datetime.now()}).where(Video.yt_video_id == video_id).execute()


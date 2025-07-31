from youtube_transcript_api import YouTubeTranscriptApi
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from loggers import logging
import re
import os


def extract_video_id(url: str) -> str:
    """
    Extracts the video ID from a YouTube URL.
    Supports both short and long formats.
    """
    # Extract only video ID
    video_id_match = re.search(r'v=([^&]+)', url)
    video_id = video_id_match.group(1) if video_id_match else None

    if not video_id:
        raise ValueError("Invalid YouTube URL format. Unable to extract video ID.")
    return video_id

def youtube_to_PDF(url:str):
    """
    Converts a YouTube video transcript to a PDF file.
    """
    # Step 1: Extract video ID from URL
    video_id = extract_video_id(url)
    
    # Fetch transcript
    transcript = YouTubeTranscriptApi().fetch(video_id=video_id, languages=['en', 'en-US'])


    text = " ".join([item.text for item in transcript])

    output_dir = os.path.join("uploaded_files", "youtube_materials")
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, f"{video_id}.pdf")
    logging.info("Saving transcript to PDF at %s", pdf_path)

    # Step 4: Write to PDF using reportlab
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path)
    story = [Paragraph(text, styles["Normal"])]
    logging.info("Writing transcript to PDF")

    doc.build(story)

    return pdf_path
           

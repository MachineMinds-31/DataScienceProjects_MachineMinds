
Voice Controlled Email Application

This application allows users to send and check emails using voice commands. The project leverages the Streamlit framework for a web interface, with Python's `speech_recognition` for voice inputs, `smtplib` for email functionality, and `pyttsx3` or `espeak` for text-to-speech capabilities.

Features

- Voice-Based Email Composition: Speak out the recipient's email and message content to send emails via voice commands.
- Inbox Checking: Access and check unread emails, and read out the latest emails, including the sender and subject.
- Cross-Platform TTS: Text-to-speech options for Linux (using `espeak`) and Windows (using `pyttsx3`).

Technologies Used

- Streamlit: Web framework for user interface.
- SpeechRecognition: Google Speech Recognition API for speech-to-text.
- smtplib: Send emails.
- imaplib: Check emails in the inbox.
- pyttsx3/espeak: Text-to-speech conversion.

Requirements

Install the required libraries:

```bash
pip install streamlit speechrecognition smtplib bs4 pyttsx3
```

Setup

1. Authentication: The app uses Gmail’s SMTP/IMAP server. Replace the email and password with your credentials.
2. Platform Support:
   - Linux: Install `espeak` for TTS.
   - Windows: Install `pyttsx3` for TTS.

Usage

1. Start the Streamlit app: Run `streamlit run app.py` in the terminal.
2. Select an Option:
   - Compose Email: Speak out the recipient’s email and your message.
   - Check Inbox: Listen to the sender and subject of the latest email.
3. Errors: Any issues with TTS or voice recognition are displayed on the Streamlit interface.

Note: This application assumes secure handling of sensitive information.

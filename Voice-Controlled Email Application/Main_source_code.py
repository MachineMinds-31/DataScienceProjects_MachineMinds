import speech_recognition as sr
import smtplib
from bs4 import BeautifulSoup
import subprocess
import os
import email
import imaplib
import getpass
import platform
import pyttsx3

# Create a recognizer instance
r = sr.Recognizer()

def text_to_speech(text):
    print(text)

    if platform.system() == "Linux":
        subprocess.run(['espeak', text])
    elif platform.system() == "Windows":
        # Use pyttsx3 for Windows
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    else:
        print("Text-to-speech not supported on this platform.")

def get_login_user():
    try:
        return getpass.getuser()
    except Exception as e:
        print(f"Error getting login user: {e}")
        return None

def get_user_choice():
    text_to_speech("Your choice ")
    with sr.Microphone() as source:
        print("Your choice:")
        audio = r.listen(source)
        print("Raw audio data:", audio)  # Print raw audio data for debugging
        print("ok done!!")
    
    try:
        choice = r.recognize_google(audio).lower()
        print("You said:", choice)
        return choice
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio.")
        return 'unknown'
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return 'error'

def compose_mail():
    text_to_speech("Enter the recipient's email address")
    with sr.Microphone() as source:
        print("Recipient's email:")
        audio = r.listen(source)
        print("ok done!!")

    try:
        recipient_email = r.recognize_google(audio).lower()
        print("Recipient's email:", recipient_email)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio.")
        return
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return

    text_to_speech("Compose your message")
    with sr.Microphone() as source:
        print("Your message:")
        audio = r.listen(source)
        print("ok done!!")

    try:
        message_content = r.recognize_google(audio)
        print("Your message:", message_content)
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio.")
        return
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return

    try:
        mail = smtplib.SMTP('smtp.gmail.com', 587)
        mail.ehlo()
        mail.starttls()
        mail.login('sandysanthosh408@gmail.com', 'SanthoshCbNs101020001974198019992000')
        mail.sendmail('santhoshchandru507@gmail.com', recipient_email, message_content)
        print("Congratulations! Your mail has been sent.")
        text_to_speech("Congratulations! Your mail has been sent.")
        mail.close()
    except Exception as e:
        print(f"Error sending email: {e}")
        text_to_speech("Error sending email. Please try again.")

def check_inbox():
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com', 993)
        email_user = 'sandysanthosh408@gmail.com'
        email_pass = 'SanthoshCbNs101020001974198019992000'
        mail.login(email_user, email_pass)

        mail.select('inbox')
        status, messages = mail.search(None, 'ALL')
        messages = messages[0].split()

        total_mails = len(messages)
        text_to_speech(f"Total mails in your inbox: {total_mails}")

        unseen_status, unseen_messages = mail.search(None, 'UNSEEN')
        unseen_messages = unseen_messages[0].split()
        unseen_count = len(unseen_messages)
        text_to_speech(f"Unseen mails in your inbox: {unseen_count}")

        if total_mails > 0:
            latest_email_id = messages[-1]
            result, message_data = mail.fetch(latest_email_id, '(RFC822)')
            raw_email = message_data[0][1].decode("utf-8")
            email_message = email.message_from_string(raw_email)

            sender = email.utils.parseaddr(email_message['From'])[1]
            subject = email_message['Subject']

            text_to_speech(f"Latest email from {sender}. Subject: {subject}")

            # Additional logic for reading email body can be added here

    except Exception as e:
        print(f"Error checking inbox: {e}")
        text_to_speech("Error checking inbox. Please try again.")
    finally:
        mail.logout()

if __name__ == "__main__":
    # Main logic can be organized using the functions defined above
    login_user = get_login_user()

    if login_user:
        print("You are logging from:", login_user)
    else:
        print("Unable to retrieve login user information.")

    text_to_speech("1. Compose a mail.")
    text_to_speech("2. Check your inbox")

    choice = get_user_choice()

    if choice == 'unknown':
        print("Could not understand your choice. Exiting.")
    elif choice == 'error':
        print("Error occurred during speech recognition. Exiting.")
    elif '1' in choice or 'one' in choice or 'compose' in choice:
        compose_mail()
    elif '2' in choice or 'two' in choice or 'check' in choice or 'inbox' in choice:
        check_inbox()
    else:
        print("Invalid choice. Exiting.")

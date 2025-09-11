import random

def generate_prompt():
    verbs = [
        'integrated',
        'blended',
        'embedded',
        'merged',
        'incorporated',
        'interwoven',
        'woven',
        'fused',
        'engraved',
        'imprinted',
        'crafted',
        'designed',
        'infused',
        'surrounded',
        'encased'
    ]
    
    adjectives = [
        'futuristic',
        'ancient',
        'mystical',
        'vibrant',
        'serene',
        'abstract',
        'surreal',
        'majestic',
        'ethereal',
        'dynamic',
        'colorful',
        'monochrome',
        'minimalist',
        'intricate',
        'ornate',
        'gleaming',
        'shadowy',
        'luminous',
        'rustic',
        'industrial'
    ]
    
    subjects = [
        'cityscape',
        'forest',
        'mountain landscape',
        'underwater scene',
        'space nebula',
        'desert',
        'ocean',
        'galaxy',
        'garden',
        'skyline',
        'countryside',
        'rainforest',
        'ice cave',
        'ancient temple',
        'futuristic metropolis',
        'market scene',
        'shopping mall',
        'palace',
        'abstract vector art',
        'robotic assembly line',
        'cyberpunk alley',
        'medieval village',
        'steampunk laboratory',
        'floating island',
        'holographic display',
        'fantasy castle',
        'art deco building',
        'modern art gallery',
        'galactic battleground',
        'enchanted forest',
        'urban street'
    ]
    
    artstyles = [
        'digital art',
        'oil painting',
        'watercolor',
        'pencil sketch',
        'cyberpunk style',
        'steampunk aesthetic',
        'fantasy art',
        'minimalist design',
        'photorealistic rendering',
        'pop art',
        'impressionist painting',
        'surrealism',
        'abstract expressionism',
        'low-poly art',
        'graffiti style',
        'vector illustration',
        'pixel art',
        'vector art',
        '3D render',
        'line art'
    ]

    # Optional: Add more categories for further diversity
    themes = [
        'horror',
        'romantic',
        'sci-fi',
        'noir',
        'vintage',
        'retro',
        'modern',
        'classic',
        'baroque',
        'gothic',
        'abstract',
        'conceptual',
        'biotech',
        'eco-friendly',
        'stealth',
        'neon-lit',
        'transparent',
        'glowing',
        'metallic',
        'crystal-like'
    ]
    
    # Select random elements from each category
    verb = random.choice(verbs)
    adjective = random.choice(adjectives)
    subject = random.choice(subjects)
    artstyle = random.choice(artstyles)
    theme = random.choice(themes)
    
    # Construct the prompt
    # prompt = f"A QR code, cleverly {verb} into a {adjective} {subject}, {theme} theme, {artstyle}, 3D rendered"
    prompt = f"A vibrant scene of a {adjective} {subject}, {theme} theme, {artstyle}, 3D rendered, high detail, intricate design"
    
    # Generate a unique and concise filename based on selected elements
    # Replace spaces with underscores and remove special characters for filesystem compatibility
    safe_adjective = ''.join(e if e.isalnum() else '_' for e in adjective)
    safe_subject = ''.join(e if e.isalnum() else '_' for e in subject)
    safe_artstyle = ''.join(e if e.isalnum() else '_' for e in artstyle)
    safe_theme = ''.join(e if e.isalnum() else '_' for e in theme)
    
    
    filename = f"{safe_adjective}_{safe_subject}_{safe_artstyle}_{safe_theme}"

    return prompt, filename

def generate_wifi_qr_string(ssid, password):
    def escape(s):
        return s.replace('\\', '\\\\').replace(';', '\\;').replace(',', '\\,').replace(':', '\\:')
    ssid_escaped = escape(ssid)
    password_escaped = escape(password)
    qr_string = f'WIFI:S:{ssid_escaped};T:WPA;P:{password_escaped};;'
    return qr_string

def generate_vcard_qr_string(first_name, last_name, phone=None, email=None, organization=None, title=None, address=None):
    """
    Generates a vCard 3.0 formatted string for contact data.
    
    Parameters:
      first_name (str): First name.
      last_name (str): Last name.
      phone (str, optional): Phone number.
      email (str, optional): Email address.
      organization (str, optional): Organization name.
      title (str, optional): Job title.
      address (str, optional): A full address string.
      
    Returns:
      str: A vCard formatted string ready for QR code encoding.
    """
    vcard = "BEGIN:VCARD\n"
    vcard += "VERSION:3.0\n"
    vcard += f"N:{last_name};{first_name};;;\n"
    vcard += f"FN:{first_name} {last_name}\n"
    if organization:
        vcard += f"ORG:{organization}\n"
    if title:
        vcard += f"TITLE:{title}\n"
    if phone:
        vcard += f"TEL;TYPE=CELL:{phone}\n"
    if email:
        vcard += f"EMAIL:{email}\n"
    if address:
        # The ADR field is structured; using empty fields for unused parts.
        vcard += f"ADR;TYPE=HOME:;;{address};;;;\n"
    vcard += "END:VCARD"
    return vcard

import urllib.parse

def generate_email_qr_string(email, subject="", body=""):
    """
    Generates a mailto URI for email links.
    
    Parameters:
      email (str): Recipient email address.
      subject (str, optional): Email subject.
      body (str, optional): Email body text.
      
    Returns:
      str: A mailto URI suitable for QR code encoding.
    """
    params = {}
    if subject:
        params["subject"] = subject
    if body:
        params["body"] = body
    param_str = urllib.parse.urlencode(params)
    if param_str:
        return f"mailto:{email}?{param_str}"
    else:
        return f"mailto:{email}"
    
def generate_geolocation_qr_string(latitude, longitude, label=None):
    """
    Generates a geo URI for encoding a location.
    
    Parameters:
      latitude (float or str): Latitude value.
      longitude (float or str): Longitude value.
      label (str, optional): An optional label (query) to pass.
      
    Returns:
      str: A geo URI string to be encoded in a QR code.
    """
    if label:
        return f"geo:{latitude},{longitude}?q={label}"
    else:
        return f"geo:{latitude},{longitude}"

def generate_calendar_qr_string(summary, description, location, dtstart, dtend):
    """
    Generates an iCalendar formatted string for a calendar event.
    
    Parameters:
      summary (str): Title of the event.
      description (str): Description of the event.
      location (str): Location where the event is held.
      dtstart (str): Event start datetime in format "YYYYMMDDTHHMMSS".
      dtend (str): Event end datetime in format "YYYYMMDDTHHMMSS".
      
    Returns:
      str: An iCalendar string that can be encoded into a QR code.
    """
    qr_string = (
        "BEGIN:VCALENDAR\n"
        "VERSION:2.0\n"
        "BEGIN:VEVENT\n"
        f"SUMMARY:{summary}\n"
        f"DESCRIPTION:{description}\n"
        f"LOCATION:{location}\n"
        f"DTSTART:{dtstart}\n"
        f"DTEND:{dtend}\n"
        "END:VEVENT\n"
        "END:VCALENDAR"
    )
    return qr_string
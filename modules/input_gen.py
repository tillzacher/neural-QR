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
    prompt = f"A QR code, cleverly {verb} into a {adjective} {subject}, {theme} theme, {artstyle}, 3D rendered"
    
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
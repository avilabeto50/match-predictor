# Define K-factors by tournament type
def get_k_factor(tournament):
    """Assign K-factor based on match importance"""
    if 'World Cup' in tournament and 'qualification' not in tournament:
        return 60
    elif 'World Cup qualification' in tournament:
        return 40
    elif any(x in tournament for x in ['Euro', 'Copa', 'African Cup', 'Asian Cup', 'Gold Cup']):
        return 50
    else:
        return 20  # Friendlies and others

